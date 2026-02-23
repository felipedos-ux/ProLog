"""
Detector com Sliding Window Temporal para Lead Time Correto.

Este script divide cada sessão (node_id) em sub-sessões de 2 horas,
permitindo calcular lead times realistas (minutos, não dias).
"""

import torch
import torch.nn.functional as F
import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import List, Tuple, Optional, Dict, Any
import json
import os
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta

from model import LogGPT, GPTConfig
from sklearn.model_selection import train_test_split

from config import (
    MODEL_NAME, MODEL_DIR, DEVICE, DATA_FILE, 
    SKIP_START_LOGS, THRESHOLD as DEFAULT_THRESHOLD,
    INFER_SCHEMA_LENGTH, TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT, LOG_DESC_MAX_LEN,
    SESSION_ID_COL, LABEL_COL, TIMESTAMP_COL
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# CONFIGURAÇÕES DE SLIDING WINDOW
WINDOW_SIZE_HOURS = 2  # Tamanho da janela em horas
INFERENCE_BATCH_SIZE = 128

class LogInferenceDataset(Dataset):
    def __init__(self, log_sequences, max_len=128):
        self.sequences = log_sequences
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
        return torch.tensor(seq, dtype=torch.long)

def collate_fn(batch):
    max_len = max(len(x) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded

def create_sliding_windows(session_df: pl.DataFrame, window_hours: int = 2) -> List[pl.DataFrame]:
    """
    Divide uma sessão em sub-sessões usando sliding window temporal.
    
    Args:
        session_df: DataFrame da sessão completa
        window_hours: Tamanho da janela em horas
        
    Returns:
        Lista de DataFrames, cada um representando uma sub-sessão
    """
    session_df = session_df.sort(TIMESTAMP_COL)
    timestamps = session_df[TIMESTAMP_COL].to_list()
    
    if len(timestamps) == 0:
        return []
    
    # Converter para datetime
    try:
        ts_datetime = [pd.to_datetime(ts, unit='s') for ts in timestamps]
    except:
        return [session_df]  # Fallback: retornar sessão inteira
    
    windows = []
    window_start = ts_datetime[0]
    window_delta = timedelta(hours=window_hours)
    
    current_window_indices = []
    
    for i, ts in enumerate(ts_datetime):
        if ts <= window_start + window_delta:
            current_window_indices.append(i)
        else:
            # Salvar janela atual e iniciar nova
            if current_window_indices:
                window_df = session_df.slice(current_window_indices[0], len(current_window_indices))
                windows.append(window_df)
            
            # Nova janela
            window_start = ts
            current_window_indices = [i]
    
    # Última janela
    if current_window_indices:
        window_df = session_df.slice(current_window_indices[0], len(current_window_indices))
        windows.append(window_df)
    
    return windows

def evaluate_window(
    window_df: pl.DataFrame,
    model: LogGPT,
    tokenizer: PreTrainedTokenizer,
    threshold: float,
    device: str,
    window_label: int
) -> Optional[Dict[str, Any]]:
    """Avalia uma sub-sessão (janela temporal)."""
    
    templates = window_df["EventTemplate"].to_list()
    raw_ts = window_df[TIMESTAMP_COL].to_list()
    
    if len(templates) < 2:
        return None
    
    try:
        timestamps = [pd.to_datetime(ts, unit='s') for ts in raw_ts]
    except ValueError:
        return None
    
    # O "failure" é o último log da janela
    failure_ts = timestamps[-1]
    window_start_ts = timestamps[0]
    
    # Preparar sequências
    sequences = []
    context_ids = []
    MAX_CONTEXT_LEN = model.config.block_size
    
    for i, current_log in enumerate(templates):
        if current_log is None:
            current_log = ""
        text = (" \n " if i > 0 else "") + str(current_log)
        new_ids = tokenizer.encode(text)
        
        if i < SKIP_START_LOGS:
            context_ids.extend(new_ids)
            if len(context_ids) > MAX_CONTEXT_LEN:
                context_ids = context_ids[-MAX_CONTEXT_LEN:]
            continue
        
        if i == 0:
            context_ids.extend(new_ids)
            continue
        
        full_seq = context_ids + new_ids
        if len(full_seq) > MAX_CONTEXT_LEN:
            input_seq = full_seq[-MAX_CONTEXT_LEN:]
            target_start_idx = len(input_seq) - len(new_ids)
        else:
            input_seq = full_seq
            target_start_idx = len(context_ids)
        
        sequences.append((i, input_seq, target_start_idx, new_ids))
        
        context_ids.extend(new_ids)
        if len(context_ids) > MAX_CONTEXT_LEN:
            context_ids = context_ids[-MAX_CONTEXT_LEN:]
    
    if len(sequences) == 0:
        return None
    
    # Processar em batch
    input_seqs = [s[1] for s in sequences]
    dataset = LogInferenceDataset(input_seqs, MAX_CONTEXT_LEN)
    loader = DataLoader(dataset, batch_size=INFERENCE_BATCH_SIZE, collate_fn=collate_fn, num_workers=0)
    
    is_detected = False
    first_alert_ts = None
    first_alert_loss = 0.0
    
    batch_idx = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            
            for i in range(batch.size(0)):
                global_idx = batch_idx + i
                if global_idx >= len(sequences):
                    break
                
                log_idx, input_seq, target_start_idx, new_ids = sequences[global_idx]
                
                target_indices = range(target_start_idx, len(input_seq))
                logit_indices = [idx - 1 for idx in target_indices]
                
                if not logit_indices:
                    continue
                if logit_indices[0] < 0 or logit_indices[-1] >= logits.size(1):
                    continue
                
                relevant_logits = logits[i, logit_indices, :]
                relevant_targets = torch.tensor(input_seq[target_start_idx:], dtype=torch.long, device=device)
                
                if relevant_logits.shape[0] != relevant_targets.shape[0]:
                    continue
                
                loss_val = F.cross_entropy(relevant_logits, relevant_targets).item()
                
                if loss_val > threshold:
                    is_detected = True
                    first_alert_ts = timestamps[log_idx]
                    first_alert_loss = loss_val
                    break
            
            if is_detected:
                break
            
            batch_idx += batch.size(0)
    
    # Calcular lead time (agora dentro da janela de 2h, será realista)
    lead_time = 0.0
    if is_detected and first_alert_ts is not None:
        lead_time = (failure_ts - first_alert_ts).total_seconds() / 60  # Em minutos
        
        # Sanity check: lead time não pode ser maior que a janela
        max_lead = WINDOW_SIZE_HOURS * 60  # Em minutos
        if lead_time > max_lead or lead_time < 0:
            lead_time = 0.0  # Inválido
    
    window_duration = (failure_ts - window_start_ts).total_seconds() / 60
    
    return {
        "is_detected": is_detected,
        "label": window_label,
        "lead_time": lead_time,
        "alert_loss": first_alert_loss,
        "window_duration_min": window_duration,
        "n_logs": len(templates)
    }

def main():
    logger.info("🚀 LogGPT Sliding Window Evaluation (Lead Time Correto)")
    logger.info(f"📏 Window Size: {WINDOW_SIZE_HOURS} hours")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = torch.load(f"{MODEL_DIR}/config.pt", weights_only=False)
    if not hasattr(config, 'dropout'):
        config.dropout = 0.0
    
    model = LogGPT(config)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/loggpt_weights.pt", weights_only=False))
    model.to(DEVICE)
    model.eval()

    # Load Threshold
    THRESHOLD = 5.0
    if os.path.exists("threshold_config.json"):
        with open("threshold_config.json", "r") as f:
            conf = json.load(f)
            THRESHOLD = conf.get("threshold", 5.0)
            logger.info(f"Threshold: {THRESHOLD:.4f}")
    
    logger.info("Loading Data...")
    df = pl.read_csv(str(DATA_FILE), infer_schema_length=INFER_SCHEMA_LENGTH)
    
    normal_ids = df.filter(pl.col(LABEL_COL) == 0)[SESSION_ID_COL].unique().to_list()
    anom_ids = df.filter(pl.col(LABEL_COL) == 1)[SESSION_ID_COL].unique().to_list()
    
    _, test_val_ids = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    _, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)
    
    eval_list = [(tid, 1) for tid in anom_ids] + [(tid, 0) for tid in test_norm_ids]
    
    logger.info(f"Processing {len(eval_list)} sessions with sliding window...")
    
    results = []
    total_windows = 0
    
    for tid, label in tqdm(eval_list, desc="Sliding Window Evaluation"):
        session_df = df.filter(pl.col(SESSION_ID_COL) == tid)
        windows = create_sliding_windows(session_df, WINDOW_SIZE_HOURS)
        
        for window_df in windows:
            res = evaluate_window(window_df, model, tokenizer, THRESHOLD, DEVICE, label)
            if res:
                results.append(res)
                total_windows += 1
    
    logger.info(f"Total windows processed: {total_windows}")
    
    # Metrics
    tp = sum(1 for r in results if r['label'] == 1 and r['is_detected'])
    fn = sum(1 for r in results if r['label'] == 1 and not r['is_detected'])
    fp = sum(1 for r in results if r['label'] == 0 and r['is_detected'])
    tn = sum(1 for r in results if r['label'] == 0 and not r['is_detected'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Lead time (apenas para detecções válidas)
    valid_leads = [r['lead_time'] for r in results 
                   if r['label'] == 1 and r['is_detected'] and r['lead_time'] > 0]
    
    avg_lead = np.mean(valid_leads) if valid_leads else 0.0
    median_lead = np.median(valid_leads) if valid_leads else 0.0
    max_lead = np.max(valid_leads) if valid_leads else 0.0
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 RESULTADOS COM SLIDING WINDOW")
    logger.info("=" * 60)
    logger.info(f"   Window Size:  {WINDOW_SIZE_HOURS}h")
    logger.info(f"   Total Windows: {total_windows}")
    logger.info("")
    logger.info(f"   Precision:    {precision:.4f}")
    logger.info(f"   Recall:       {recall:.4f}")
    logger.info(f"   F1 Score:     {f1:.4f}")
    logger.info("")
    logger.info("⏱️  LEAD TIME (Corrigido):")
    logger.info(f"   Detecções válidas: {len(valid_leads)}")
    logger.info(f"   Avg Lead Time:     {avg_lead:.2f} min")
    logger.info(f"   Median Lead Time:  {median_lead:.2f} min")
    logger.info(f"   Max Lead Time:     {max_lead:.2f} min")
    logger.info("=" * 60)
    
    # Save report
    report = [
        "=" * 60,
        "RESULTADOS COM SLIDING WINDOW - LEAD TIME CORRIGIDO",
        "=" * 60,
        f"Window Size: {WINDOW_SIZE_HOURS}h",
        f"Total Windows: {total_windows}",
        "",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"F1: {f1:.4f}",
        "",
        "LEAD TIME (Corrigido):",
        f"Detecções válidas: {len(valid_leads)}",
        f"Avg Lead Time: {avg_lead:.2f} min",
        f"Median Lead Time: {median_lead:.2f} min",
        f"Max Lead Time: {max_lead:.2f} min",
        "=" * 60
    ]
    
    with open("results_sliding_window.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    logger.info("✅ Report saved to results_sliding_window.txt")

if __name__ == "__main__":
    main()
