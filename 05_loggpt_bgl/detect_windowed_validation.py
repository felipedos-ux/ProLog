"""
Script de Validação do Modelo Re-treinado (Recall Improvement).

Objetivo:
Avaliar se o modelo treinado com Janelas (Train Windowed) tem melhor performance
que o modelo original (Train Session).

Diferenças do detect_hybrid_leadtime.py:
- Usa `config_retrain` (pesos novos).
- Foca em comparar métricas com o baseline.
"""

import torch
import torch.nn.functional as F
import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from transformers import AutoTokenizer
import json
import os

from model import LogGPT
from sklearn.model_selection import train_test_split

# USE RETRAIN CONFIG
import config_retrain as config
from config_retrain import (
    MODEL_NAME, MODEL_DIR, DEVICE, DATA_FILE, 
    SKIP_START_LOGS, INFER_SCHEMA_LENGTH, 
    TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT,
    SESSION_ID_COL, LABEL_COL, TIMESTAMP_COL
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# CONFIGURAÇÕES (Identicas ao baseline para comparação justa)
WINDOW_SIZE_HOURS = 2
BATCH_SIZE = 128
THRESHOLD = 17.17 # Optimized for >90% Recall 

def preprocess_all_windows(df, session_ids, tokenizer, window_hours=2, max_context=128):
    all_windows = []
    
    # Filtrar apenas sessões que vamos usar
    target_df = df.filter(pl.col(SESSION_ID_COL).is_in(session_ids))
    
    # Sort
    target_df = target_df.sort([SESSION_ID_COL, TIMESTAMP_COL])
    
    session_groups = target_df.group_by(SESSION_ID_COL, maintain_order=True)
    
    for tid, session_df in tqdm(session_groups, desc="Preprocessing Windows", total=len(session_ids)):
        
        templates = session_df["EventTemplate"].to_list()
        raw_ts = session_df[TIMESTAMP_COL].to_list()
        log_labels = session_df[LABEL_COL].to_list()
        
        if len(templates) < 2: continue
        
        try:
            ts_datetime = [pd.to_datetime(ts, unit='s') for ts in raw_ts]
        except: continue
        
        # Sliding Window Logic
        window_delta = timedelta(hours=window_hours)
        window_start = ts_datetime[0]
        current_indices = []
        
        error_timestamps = [pd.to_datetime(ts, unit='s') for ts, lbl in zip(raw_ts, log_labels) if lbl == 1]
        
        for i, ts in enumerate(ts_datetime):
            if ts <= window_start + window_delta:
                current_indices.append(i)
            else:
                if len(current_indices) >= 2:
                    process_window(all_windows, current_indices, templates, ts_datetime, log_labels, error_timestamps, tokenizer, max_context)
                
                window_start = ts
                current_indices = [i]
        
        # Last window
        if len(current_indices) >= 2:
            process_window(all_windows, current_indices, templates, ts_datetime, log_labels, error_timestamps, tokenizer, max_context)
    
    return all_windows

def process_window(all_windows, indices, templates, timestamps, labels, error_timestamps, tokenizer, max_context):
    w_templates = [templates[j] for j in indices]
    w_labels = [labels[j] for j in indices]
    w_ts = [timestamps[j] for j in indices]
    
    # HYBRID LABEL: Janela é anomala se TIVER erro DENTRO dela
    w_anom_label = 1 if sum(w_labels) > 0 else 0
    
    sequences = []
    
    # Gerar sequencias (igual train)
    context_ids = []
    
    for i, current_log in enumerate(w_templates):
        text = (" \n " if i > 0 else "") + str(current_log)
        new_ids = tokenizer.encode(text)
        
        if i == 0:
            context_ids.extend(new_ids)
            continue
            
        full_seq = context_ids + new_ids
        if len(full_seq) > max_context:
            input_seq = full_seq[-max_context:]
            target_start_idx = max(0, len(input_seq) - len(new_ids))
        else:
            input_seq = full_seq
            target_start_idx = len(context_ids)
            
        sequences.append({
            'input_ids': input_seq,
            'target_start': target_start_idx,
            'timestamp': w_ts[i]
        })
        
        context_ids.extend(new_ids)
        if len(context_ids) > max_context: context_ids = context_ids[-max_context:]
        
    if sequences:
        all_windows.append({
            'sequences': sequences,
            'label': w_anom_label,
            'error_timestamps': error_timestamps # Passamos TODOS os erros da sessão para calcular Next Real Error
        })

def evaluate_windows_batch(windows, model, threshold, device):
    results = []
    all_seqs = []
    seq_to_window = []
    
    for w_idx, window in enumerate(windows):
        for seq in window['sequences']:
            all_seqs.append(seq)
            seq_to_window.append(w_idx)
            
    if not all_seqs: return results
    
    # Batch Processing
    n_seqs = len(all_seqs)
    all_losses = [None] * n_seqs
    
    # Otimização: inferencia em batch
    for batch_start in tqdm(range(0, n_seqs, BATCH_SIZE), desc="Inference"):
        batch_end = min(batch_start + BATCH_SIZE, n_seqs)
        batch_seqs = all_seqs[batch_start:batch_end]
        
        max_len = max(len(s['input_ids']) for s in batch_seqs)
        batch_tensor = torch.zeros(len(batch_seqs), max_len, dtype=torch.long, device=device)
        
        for i, seq in enumerate(batch_seqs):
            ids = seq['input_ids']
            batch_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            
        with torch.no_grad():
            logits, _ = model(batch_tensor)
            
        for i, seq in enumerate(batch_seqs):
            input_ids = seq['input_ids']
            target_start = seq['target_start']
            
            target_indices = range(target_start, len(input_ids))
            logit_indices = [idx - 1 for idx in target_indices]
            
            if not logit_indices: continue
            
            relevant_logits = logits[i, logit_indices, :]
            relevant_targets = torch.tensor(input_ids[target_start:], dtype=torch.long, device=device)
            
            loss = F.cross_entropy(relevant_logits, relevant_targets).item()
            all_losses[batch_start + i] = (loss, seq['timestamp'])
            
    # Aggregate results
    window_results = {i: {'first_alert_ts': None, 'detected': False, 'max_loss': 0} for i in range(len(windows))}
    
    for seq_idx, loss_data in enumerate(all_losses):
        if loss_data is None: continue
        loss, ts = loss_data
        w_idx = seq_to_window[seq_idx]
        
        if loss > window_results[w_idx]['max_loss']:
             window_results[w_idx]['max_loss'] = loss
        
        if loss > threshold and not window_results[w_idx]['detected']:
            window_results[w_idx]['detected'] = True
            window_results[w_idx]['first_alert_ts'] = ts
            
    # Calculate Metrics
    for w_idx, window in enumerate(windows):
        wr = window_results[w_idx]
        lead_time = 0.0
        is_anticipated = False
        
        if wr['detected'] and wr['first_alert_ts']:
            error_timestamps = window['error_timestamps']
            first_alert = wr['first_alert_ts']
            
            future_errors = [t for t in error_timestamps if t > first_alert]
            
            if future_errors:
                next_error = future_errors[0]
                lead_time = (next_error - first_alert).total_seconds() / 60
                is_anticipated = True
                
        results.append({
            'is_detected': wr['detected'],
            'label': window['label'],
            'lead_time': lead_time,
            'is_anticipated': is_anticipated,
            'max_loss': wr['max_loss']
        })
        
    return results

def main():
    logger.info("🚀 Validating RETRAINED Model (Windowed)")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # WARN: Load from RETRAIN DIR
    logger.info(f"Loading weights from {MODEL_DIR}")
    config_path = f"{MODEL_DIR}/config.pt"
    weights_path = f"{MODEL_DIR}/loggpt_weights.pt"
    
    if not os.path.exists(config_path):
        logger.error(f"Config not found at {config_path}. Training finished?")
        return

    model_config = torch.load(config_path, weights_only=False)
    model = LogGPT(model_config)
    model.load_state_dict(torch.load(weights_path, weights_only=False))
    model.to(DEVICE)
    model.eval()
    
    # Load Data (Original Source, not windowed CSV)
    # Why? To ensure fair comparison with detect_hybrid_leadtime.py logic
    # We use DATA_FILE from config_retrain which is BGL_windowed_train.csv??
    # NO. We verify on the RAW data (or processed raw).
    
    # Force use of original processed CSV for validation
    ORIG_DATA = "D:/ProLog/data/BGL_processed.csv"
    logger.info(f"Loading raw data from {ORIG_DATA}")
    df = pl.read_csv(ORIG_DATA, infer_schema_length=INFER_SCHEMA_LENGTH)
    
    # Split (Same seed)
    normal_ids = df.filter(pl.col(LABEL_COL) == 0)[SESSION_ID_COL].unique().to_list()
    anom_ids = df.filter(pl.col(LABEL_COL) == 1)[SESSION_ID_COL].unique().to_list()
    _, test_val_ids = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    _, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)
    
    # Validation Set (500 sessions mixed)
    sample_norm = test_norm_ids[:300]
    sample_anom = anom_ids[:300] 
    session_ids = sample_anom + sample_norm
    
    logger.info(f"Validating on {len(session_ids)} sessions...")
    
    windows = preprocess_all_windows(df, session_ids, tokenizer)
    logger.info(f"Generated {len(windows)} windows.")
    
    results = evaluate_windows_batch(windows, model, THRESHOLD, DEVICE)
    
    # Metrics
    tp = sum(1 for r in results if r['label'] == 1 and r['is_detected'])
    fp = sum(1 for r in results if r['label'] == 0 and r['is_detected'])
    fn = sum(1 for r in results if r['label'] == 1 and not r['is_detected'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*40)
    print(f"📊 RESULTS (Retrained Model)")
    print("="*40)
    print(f"Threshold: {THRESHOLD}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    
    if recall > 0.8:
        print("\n✅ SUCCESS: Recall significantly improved!")
    else:
        print("\n⚠️ Note: If Recall is low, adjust THRESHOLD.")

if __name__ == "__main__":
    main()
