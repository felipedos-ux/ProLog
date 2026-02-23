"""
Detector TURBO com Sliding Window - Máxima Saturação de GPU.

Processa múltiplas janelas de múltiplas sessões em paralelo.
"""

import torch
import torch.nn.functional as F
import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional
import json
import os
from torch.utils.data import Dataset, DataLoader
from datetime import timedelta

from model import LogGPT
from sklearn.model_selection import train_test_split

from config import (
    MODEL_NAME, MODEL_DIR, DEVICE, DATA_FILE, 
    SKIP_START_LOGS, INFER_SCHEMA_LENGTH, 
    TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT, LOG_DESC_MAX_LEN,
    SESSION_ID_COL, LABEL_COL, TIMESTAMP_COL
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# CONFIGURAÇÕES TURBO
WINDOW_SIZE_HOURS = 2
BATCH_SIZE = 256  # Batch grande para saturar GPU
NUM_WORKERS = 0

class WindowDataset(Dataset):
    """Dataset de janelas pré-processadas."""
    def __init__(self, windows_data, max_len=128):
        self.windows = windows_data
        self.max_len = max_len
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx]

def preprocess_all_windows(df, session_ids, labels, tokenizer, window_hours=2, max_context=128):
    """
    Pré-processa TODAS as janelas de TODAS as sessões de uma vez.
    Retorna lista de dicts com dados prontos para inferência.
    """
    all_windows = []
    
    for tid, label in tqdm(zip(session_ids, labels), total=len(session_ids), desc="Preprocessing Windows"):
        session_df = df.filter(pl.col(SESSION_ID_COL) == tid)
        session_df = session_df.sort(TIMESTAMP_COL)
        
        templates = session_df["EventTemplate"].to_list()
        raw_ts = session_df[TIMESTAMP_COL].to_list()
        
        if len(templates) < 2:
            continue
        
        try:
            ts_datetime = [pd.to_datetime(ts, unit='s') for ts in raw_ts]
        except:
            continue
        
        # Dividir em janelas
        window_delta = timedelta(hours=window_hours)
        window_start = ts_datetime[0]
        current_indices = []
        
        for i, ts in enumerate(ts_datetime):
            if ts <= window_start + window_delta:
                current_indices.append(i)
            else:
                if len(current_indices) >= 2:
                    # Processar janela
                    window_templates = [templates[j] for j in current_indices]
                    window_ts = [ts_datetime[j] for j in current_indices]
                    
                    window_data = prepare_window_sequences(
                        window_templates, window_ts, tokenizer, max_context, label
                    )
                    if window_data:
                        all_windows.append(window_data)
                
                window_start = ts
                current_indices = [i]
        
        # Última janela
        if len(current_indices) >= 2:
            window_templates = [templates[j] for j in current_indices]
            window_ts = [ts_datetime[j] for j in current_indices]
            
            window_data = prepare_window_sequences(
                window_templates, window_ts, tokenizer, max_context, label
            )
            if window_data:
                all_windows.append(window_data)
    
    return all_windows

def prepare_window_sequences(templates, timestamps, tokenizer, max_context, label):
    """Prepara sequências de uma janela para inferência."""
    sequences = []
    context_ids = []
    
    for i, current_log in enumerate(templates):
        if current_log is None:
            current_log = ""
        text = (" \n " if i > 0 else "") + str(current_log)
        new_ids = tokenizer.encode(text)
        
        if i < SKIP_START_LOGS:
            context_ids.extend(new_ids)
            if len(context_ids) > max_context:
                context_ids = context_ids[-max_context:]
            continue
        
        if i == 0:
            context_ids.extend(new_ids)
            continue
        
        full_seq = context_ids + new_ids
        if len(full_seq) > max_context:
            input_seq = full_seq[-max_context:]
            target_start_idx = len(input_seq) - len(new_ids)
        else:
            input_seq = full_seq
            target_start_idx = len(context_ids)
        
        sequences.append({
            'input_ids': input_seq,
            'target_start': target_start_idx,
            'log_idx': i,
            'timestamp': timestamps[i]
        })
        
        context_ids.extend(new_ids)
        if len(context_ids) > max_context:
            context_ids = context_ids[-max_context:]
    
    if not sequences:
        return None
    
    return {
        'sequences': sequences,
        'label': label,
        'failure_ts': timestamps[-1],
        'window_start': timestamps[0]
    }

def evaluate_windows_batch(windows, model, threshold, device, max_context=128):
    """
    Avalia múltiplas janelas em batch para máxima saturação de GPU.
    """
    results = []
    
    # Flatten todas as sequências de todas as janelas
    all_seqs = []
    seq_to_window = []  # Mapeia sequência -> janela
    
    for w_idx, window in enumerate(windows):
        for seq in window['sequences']:
            all_seqs.append(seq)
            seq_to_window.append(w_idx)
    
    if not all_seqs:
        return results
    
    # Preparar batches
    n_seqs = len(all_seqs)
    all_losses = [None] * n_seqs
    
    for batch_start in range(0, n_seqs, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_seqs)
        batch_seqs = all_seqs[batch_start:batch_end]
        
        # Pad e criar tensor
        max_len = max(len(s['input_ids']) for s in batch_seqs)
        batch_tensor = torch.zeros(len(batch_seqs), max_len, dtype=torch.long, device=device)
        
        for i, seq in enumerate(batch_seqs):
            ids = seq['input_ids']
            batch_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        
        with torch.no_grad():
            logits, _ = model(batch_tensor)
        
        # Calcular loss para cada sequência
        for i, seq in enumerate(batch_seqs):
            input_ids = seq['input_ids']
            target_start = seq['target_start']
            
            target_indices = range(target_start, len(input_ids))
            logit_indices = [idx - 1 for idx in target_indices]
            
            if not logit_indices or logit_indices[0] < 0 or logit_indices[-1] >= logits.size(1):
                continue
            
            relevant_logits = logits[i, logit_indices, :]
            relevant_targets = torch.tensor(input_ids[target_start:], dtype=torch.long, device=device)
            
            if relevant_logits.shape[0] == relevant_targets.shape[0]:
                loss = F.cross_entropy(relevant_logits, relevant_targets).item()
                all_losses[batch_start + i] = (loss, seq['log_idx'], seq['timestamp'])
    
    # Agregar resultados por janela
    window_results = {i: {'max_loss': 0, 'first_alert_ts': None, 'first_alert_loss': 0, 'detected': False} 
                      for i in range(len(windows))}
    
    for seq_idx, loss_data in enumerate(all_losses):
        if loss_data is None:
            continue
        
        loss, log_idx, ts = loss_data
        w_idx = seq_to_window[seq_idx]
        
        if loss > threshold:
            if not window_results[w_idx]['detected']:
                window_results[w_idx]['detected'] = True
                window_results[w_idx]['first_alert_ts'] = ts
                window_results[w_idx]['first_alert_loss'] = loss
        
        if loss > window_results[w_idx]['max_loss']:
            window_results[w_idx]['max_loss'] = loss
    
    # Converter para lista de resultados
    for w_idx, window in enumerate(windows):
        wr = window_results[w_idx]
        lead_time = 0.0
        
        if wr['detected'] and wr['first_alert_ts']:
            lead_time = (window['failure_ts'] - wr['first_alert_ts']).total_seconds() / 60
            if lead_time < 0 or lead_time > WINDOW_SIZE_HOURS * 60:
                lead_time = 0.0
        
        results.append({
            'is_detected': wr['detected'],
            'label': window['label'],
            'lead_time': lead_time,
            'alert_loss': wr['first_alert_loss']
        })
    
    return results

def main():
    logger.info("🚀 TURBO Sliding Window (GPU 100% Saturation)")
    logger.info(f"📏 Window: {WINDOW_SIZE_HOURS}h | Batch: {BATCH_SIZE}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = torch.load(f"{MODEL_DIR}/config.pt", weights_only=False)
    if not hasattr(config, 'dropout'):
        config.dropout = 0.0
    
    model = LogGPT(config)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/loggpt_weights.pt", weights_only=False))
    model.to(DEVICE)
    model.eval()
    
    THRESHOLD = 34.41
    if os.path.exists("threshold_config.json"):
        with open("threshold_config.json", "r") as f:
            THRESHOLD = json.load(f).get("threshold", 34.41)
    logger.info(f"Threshold: {THRESHOLD:.4f}")
    
    logger.info("Loading Data...")
    df = pl.read_csv(str(DATA_FILE), infer_schema_length=INFER_SCHEMA_LENGTH)
    
    normal_ids = df.filter(pl.col(LABEL_COL) == 0)[SESSION_ID_COL].unique().to_list()
    anom_ids = df.filter(pl.col(LABEL_COL) == 1)[SESSION_ID_COL].unique().to_list()
    
    _, test_val_ids = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    _, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)
    
    session_ids = anom_ids + test_norm_ids
    labels = [1] * len(anom_ids) + [0] * len(test_norm_ids)
    
    logger.info(f"Sessions: {len(session_ids)} | Anomalies: {len(anom_ids)}, Normal: {len(test_norm_ids)}")
    
    # Pré-processar todas as janelas (CPU intensive, mas feito uma vez)
    logger.info("📦 Pre-processing ALL windows...")
    all_windows = preprocess_all_windows(
        df, session_ids, labels, tokenizer, WINDOW_SIZE_HOURS, model.config.block_size
    )
    logger.info(f"Total windows: {len(all_windows)}")
    
    # Processar em mega-batches (GPU intensive)
    logger.info("⚡ Evaluating with GPU saturation...")
    
    all_results = []
    MEGA_BATCH = 1000  # Processar 1000 janelas por vez
    
    for i in tqdm(range(0, len(all_windows), MEGA_BATCH), desc="Mega-Batch Processing"):
        batch_windows = all_windows[i:i+MEGA_BATCH]
        batch_results = evaluate_windows_batch(batch_windows, model, THRESHOLD, DEVICE, model.config.block_size)
        all_results.extend(batch_results)
    
    # Métricas
    tp = sum(1 for r in all_results if r['label'] == 1 and r['is_detected'])
    fn = sum(1 for r in all_results if r['label'] == 1 and not r['is_detected'])
    fp = sum(1 for r in all_results if r['label'] == 0 and r['is_detected'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    valid_leads = [r['lead_time'] for r in all_results if r['label'] == 1 and r['is_detected'] and r['lead_time'] > 0]
    avg_lead = np.mean(valid_leads) if valid_leads else 0
    median_lead = np.median(valid_leads) if valid_leads else 0
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 RESULTADOS TURBO SLIDING WINDOW")
    logger.info("=" * 60)
    logger.info(f"   Precision:    {precision:.4f}")
    logger.info(f"   Recall:       {recall:.4f}")
    logger.info(f"   F1 Score:     {f1:.4f}")
    logger.info("")
    logger.info("⏱️  LEAD TIME (Corrigido):")
    logger.info(f"   Avg:    {avg_lead:.2f} min")
    logger.info(f"   Median: {median_lead:.2f} min")
    logger.info("=" * 60)
    
    with open("results_turbo_sliding.txt", "w") as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Avg Lead Time: {avg_lead:.2f} min\n")
        f.write(f"Median Lead Time: {median_lead:.2f} min\n")
    
    logger.info("✅ Saved to results_turbo_sliding.txt")

if __name__ == "__main__":
    main()
