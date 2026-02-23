"""
Detector HÍBRIDO com Sliding Window e Lead Time Realista.

Diferenças do TURBO:
1. Label da janela é dinâmico (baseado nos logs CONTIDOS nela).
2. Lead time é calculado até o PRÓXIMO ERRO REAL na janela, não até o fim da janela.
"""

import numpy as np
import torch
import torch.nn.functional as F
import polars as pl
import pandas as pd
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

# CONFIGURAÇÕES
WINDOW_SIZE_HOURS = 2
BATCH_SIZE = 256
OVERLAP = False # Future improvement

def preprocess_all_windows(df, session_ids, tokenizer, window_hours=2, max_context=128):
    """
    Pré-processa janelas calculando labels DINÂMICOS.
    """
    all_windows = []
    
    # Filtrar apenas sessões que vamos usar
    target_df = df.filter(pl.col(SESSION_ID_COL).is_in(session_ids))
    
    # OTIMIZAÇÃO: Iterar por grupos ao invés de filtrar repetidamente
    # target_df.group_by(SESSION_ID_COL) é muito mais rápido
    
    # Ordenar globalmente primeiro para garantir ordem nos grupos (opcional mas bom)
    target_df = target_df.sort([SESSION_ID_COL, TIMESTAMP_COL])
    
    # Group By retorna (tuple_key, dataframe)
    # Como agrupamos por uma col, key é o valor direto se usarmos iter direto? 
    # Polars group_by iterator yields (key, df)
    
    session_groups = target_df.group_by(SESSION_ID_COL, maintain_order=True)
    
    for tid, session_df in tqdm(session_groups, desc="Preprocessing Windows", total=len(unique_sessions)):
        
        templates = session_df["EventTemplate"].to_list()
        raw_ts = session_df[TIMESTAMP_COL].to_list()
        log_labels = session_df[LABEL_COL].to_list() # 0 ou 1 por log
        
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
                    process_window(all_windows, current_indices, templates, ts_datetime, log_labels, tokenizer, max_context)
                
                window_start = ts
                current_indices = [i]
        
        # Última janela
        if len(current_indices) >= 2:
            process_window(all_windows, current_indices, templates, ts_datetime, log_labels, tokenizer, max_context)
    
    return all_windows

def process_window(all_windows, indices, templates, timestamps, labels, tokenizer, max_context):
    window_templates = [templates[j] for j in indices]
    window_ts = [timestamps[j] for j in indices]
    window_labels = [labels[j] for j in indices]
    
    # Label da janela: 1 se tiver QUALQUER erro
    window_anom_label = 1 if sum(window_labels) > 0 else 0
    
    window_data = prepare_window_sequences(
        window_templates, window_ts, window_labels, window_anom_label, tokenizer, max_context
    )
    if window_data:
        all_windows.append(window_data)

def prepare_window_sequences(templates, timestamps, log_labels, window_label, tokenizer, max_context):
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
    
    # Identificar timestamps de erros reais
    error_timestamps = [timestamps[i] for i, label in enumerate(log_labels) if label == 1]
    
    return {
        'sequences': sequences,
        'label': window_label, # 1 se tiver erro na janela
        'error_timestamps': error_timestamps, # Lista de momentos exatos dos erros
        'window_start': timestamps[0],
        'window_end': timestamps[-1]
    }

def evaluate_windows_batch(windows, model, threshold, device):
    results = []
    all_seqs = []
    seq_to_window = []
    
    for w_idx, window in enumerate(windows):
        for seq in window['sequences']:
            all_seqs.append(seq)
            seq_to_window.append(w_idx)
    
    if not all_seqs:
        # print("DEBUG: No sequences in this batch of windows")
        return results
    
    # Batch Processing
    n_seqs = len(all_seqs)
    all_losses = [None] * n_seqs
    
    for batch_start in range(0, n_seqs, BATCH_SIZE):
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
            
            if not logit_indices or logit_indices[0] < 0 or logit_indices[-1] >= logits.size(1):
                continue
            
            relevant_logits = logits[i, logit_indices, :]
            relevant_targets = torch.tensor(input_ids[target_start:], dtype=torch.long, device=device)
            
            if relevant_logits.shape[0] == relevant_targets.shape[0]:
                loss = F.cross_entropy(relevant_logits, relevant_targets).item()
                all_losses[batch_start + i] = (loss, seq['timestamp'])
    
    # Agregar por janela
    window_results = {i: {'first_alert_ts': None, 'detected': False} for i in range(len(windows))}
    
    for seq_idx, loss_data in enumerate(all_losses):
        if loss_data is None: continue
        loss, ts = loss_data
        w_idx = seq_to_window[seq_idx]
        
        if loss > threshold and not window_results[w_idx]['detected']:
            # print(f"DEBUG: Detection! Window {w_idx} Loss {loss:.4f} > {threshold}")
            window_results[w_idx]['detected'] = True
            window_results[w_idx]['first_alert_ts'] = ts
            
    # Calcular Lead Time Híbrido
    for w_idx, window in enumerate(windows):
        wr = window_results[w_idx]
        lead_time = 0.0
        is_anticipated = False
        
        if wr['detected'] and wr['first_alert_ts']:
            # Lógica HÍBRIDA: Procurar o próximo erro real
            error_timestamps = window['error_timestamps']
            first_alert = wr['first_alert_ts']
            
            # Filtrar erros que ocorreram DEPOIS da detecção
            future_errors = [t for t in error_timestamps if t > first_alert]
            
            if future_errors:
                # Pegar o mais próximo
                next_error = future_errors[0]
                lead_time = (next_error - first_alert).total_seconds() / 60
                is_anticipated = True
            else:
                # Se não tem erro futuro, mas a janela TEM erro (passado), lead time é negativo?
                # Ou consideramos 0? Vamos considerar 0 para métricas de "antecipação".
                # Se a janela é normal (FP), lead time é 0.
                pass
                
        results.append({
            'is_detected': wr['detected'],
            'label': window['label'],
            'lead_time': lead_time,
            'is_anticipated': is_anticipated
        })
        
    return results

def main():
    logger.info("🚀 HYBRID Lead Time Detector (Sliding Window + Precision Logic)")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = torch.load(f"{MODEL_DIR}/config.pt", weights_only=False)
    if not hasattr(config, 'dropout'): config.dropout = 0.0
    
    model = LogGPT(config)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/loggpt_weights.pt", weights_only=False))
    model.to(DEVICE)
    model.eval()
    
    THRESHOLD = 5.0
    if os.path.exists("threshold_config.json"):
        with open("threshold_config.json", "r") as f:
            THRESHOLD = json.load(f).get("threshold", 5.0)
    logger.info(f"Threshold: {THRESHOLD:.4f}")
    
    logger.info("Loading Data...")
    df = pl.read_csv(str(DATA_FILE), infer_schema_length=INFER_SCHEMA_LENGTH)
    
    # Selecionar sessões para teste (mesmo split)
    normal_ids = df.filter(pl.col(LABEL_COL) == 0)[SESSION_ID_COL].unique().to_list()
    anom_ids = df.filter(pl.col(LABEL_COL) == 1)[SESSION_ID_COL].unique().to_list()
    
    _, test_val_ids = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    _, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)
    
    session_ids = anom_ids + test_norm_ids
    
    logger.info("📦 Preprocessing Windows (Dynamic Labeling)...")
    all_windows = preprocess_all_windows(
        df, session_ids, tokenizer, WINDOW_SIZE_HOURS, model.config.block_size
    )
    
    logger.info("⚡ Evaluating...")
    all_results = []
    MEGA_BATCH = 1000
    
    for i in tqdm(range(0, len(all_windows), MEGA_BATCH), desc="Evaluating"):
        batch_windows = all_windows[i:i+MEGA_BATCH]
        batch_results = evaluate_windows_batch(batch_windows, model, THRESHOLD, DEVICE)
    logger.info(f"⚡ Evaluation complete. Total results: {len(all_results)}")
    
    # Métricas
    tp = sum(1 for r in all_results if r['label'] == 1 and r['is_detected'])
    fp = sum(1 for r in all_results if r['label'] == 0 and r['is_detected'])
    fn = sum(1 for r in all_results if r['label'] == 1 and not r['is_detected'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Anticipation Metrics (Apenas Lead Time > 0)
    anticipated_leads = [r['lead_time'] for r in all_results if r['is_anticipated']]
    
    avg_lead = np.mean(anticipated_leads) if anticipated_leads else 0
    median_lead = np.median(anticipated_leads) if anticipated_leads else 0
    max_lead = np.max(anticipated_leads) if anticipated_leads else 0
    
    print("\n" + "="*50)
    print("📊 RESULTADOS HÍBRIDOS (Next Real Error Logic)")
    print("="*50)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 30)
    print(f"✅ Anticipated (Lead > 0): {len(anticipated_leads)} / {tp} detectados")
    print(f"Avg Lead Time:    {avg_lead:.2f} min")
    print(f"Median Lead Time: {median_lead:.2f} min")
    print(f"Max Lead Time:    {max_lead:.2f} min")
    print("="*50)
    
    with open("results_hybrid.txt", "w") as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Avg Lead Time: {avg_lead:.2f}\n")
        f.write(f"Median Lead Time: {median_lead:.2f}\n")

if __name__ == "__main__":
    main()
