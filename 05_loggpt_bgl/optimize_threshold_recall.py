"""
Script de Otimização de Threshold para Recall.

Objetivo: Encontrar o threshold que garante Recall > 90% (e analisar a perda de precisão).
Estratégia:
1. Carregar modelo e dados.
2. Gerar "Anomaly Scores" para cada janela (max loss).
3. Calcular Precision/Recall para múltiplos thresholds sem re-rodar o modelo.
"""

import numpy as np
import torch
import torch.nn.functional as F
import polars as pl
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import os
from datetime import timedelta

from model import LogGPT
from sklearn.model_selection import train_test_split
from config import (
    MODEL_NAME, MODEL_DIR, DEVICE, DATA_FILE, 
    SKIP_START_LOGS, INFER_SCHEMA_LENGTH, 
    TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT,
    SESSION_ID_COL, LABEL_COL, TIMESTAMP_COL
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# CONFIGURAÇÕES
WINDOW_SIZE_HOURS = 2
BATCH_SIZE = 256 # Pode ser maior aqui? Não, manter seguro.

def get_window_scores(df, session_ids, tokenizer, model, device, window_hours=2, max_context=128):
    """
    Gera (score, label) para todas as janelas.
    Score = Max Loss na janela.
    """
    window_scores = []
    window_labels = []
    
    # Filtrar sessões
    target_df = df.filter(pl.col(SESSION_ID_COL).is_in(session_ids))
    target_df = target_df.sort([SESSION_ID_COL, TIMESTAMP_COL])
    
    session_groups = target_df.group_by(SESSION_ID_COL, maintain_order=True)
    
    logger.info("⚡ Generating scores for optimization...")
    
    all_windows_seqs = []
    window_meta = [] # (window_idx, label)
    
    # 1. Preprocessamento (Gerar sequencias)
    # Fazemos em blocos para não estourar RAM
    
    current_batch_seqs = []
    current_batch_meta = []
    
    w_idx_global = 0
    
    for tid, session_df in tqdm(session_groups, desc="Preprocessing", total=len(session_ids)):
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
        
        for i, ts in enumerate(ts_datetime):
            if ts <= window_start + window_delta:
                current_indices.append(i)
            else:
                if len(current_indices) >= 2:
                    # Process window
                    w_seqs, w_label = prepare_window(current_indices, templates, log_labels, tokenizer, max_context)
                    if w_seqs:
                        all_windows_seqs.extend(w_seqs)
                        # Mapear quais seqs pertencem a esta janela
                        # Vamos simplificar: salvar (start_seq_idx, end_seq_idx, label)
                        start_idx = len(all_windows_seqs) - len(w_seqs)
                        end_idx = len(all_windows_seqs)
                        window_meta.append({'start': start_idx, 'end': end_idx, 'label': w_label})
                
                window_start = ts
                current_indices = [i]
                
        # Last window
        if len(current_indices) >= 2:
             w_seqs, w_label = prepare_window(current_indices, templates, log_labels, tokenizer, max_context)
             if w_seqs:
                all_windows_seqs.extend(w_seqs)
                start_idx = len(all_windows_seqs) - len(w_seqs)
                end_idx = len(all_windows_seqs)
                window_meta.append({'start': start_idx, 'end': end_idx, 'label': w_label})

    # 2. Inference (Calcular Loss)
    logger.info(f"⚡ Running Inference on {len(all_windows_seqs)} sequences...")
    all_losses = run_inference(all_windows_seqs, model, device)
    
    # 3. Aggregate per Window (Max Loss strategy)
    final_scores = []
    final_labels = []
    
    for meta in window_meta:
        w_losses = all_losses[meta['start']:meta['end']]
        if not w_losses: max_l = 0
        else: max_l = max(w_losses)
        
        final_scores.append(max_l)
        final_labels.append(meta['label'])
        
    return final_scores, final_labels

def prepare_window(indices, templates, labels, tokenizer, max_context):
    w_templates = [templates[j] for j in indices]
    w_labels = [labels[j] for j in indices]
    
    # Label da janela: 1 se tiver erro
    w_anom_label = 1 if sum(w_labels) > 0 else 0
    
    # Se janela é normal, podemos pular downsampling?
    # Para otimização, queremos ver tudo.
    
    sequences = []
    context_ids = []
    
    for i, current_log in enumerate(w_templates):
        if current_log is None: current_log = ""
        text = (" \n " if i > 0 else "") + str(current_log)
        new_ids = tokenizer.encode(text)
        
        if i < SKIP_START_LOGS:
            context_ids.extend(new_ids)
            if len(context_ids) > max_context: context_ids = context_ids[-max_context:]
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
            'target_start': target_start_idx
        })
        
        context_ids.extend(new_ids)
        if len(context_ids) > max_context: context_ids = context_ids[-max_context:]
        
    return sequences, w_anom_label

import pandas as pd # Reimport for safety in function

def run_inference(all_seqs, model, device):
    all_losses = []
    
    for batch_start in tqdm(range(0, len(all_seqs), BATCH_SIZE), desc="Inference"):
        batch_end = min(batch_start + BATCH_SIZE, len(all_seqs))
        batch_seqs = all_seqs[batch_start:batch_end]
        
        if not batch_seqs: continue

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
            
            if not logit_indices: 
                all_losses.append(0.0)
                continue
                
            relevant_logits = logits[i, logit_indices, :]
            relevant_targets = torch.tensor(input_ids[target_start:], dtype=torch.long, device=device)
            
            loss = F.cross_entropy(relevant_logits, relevant_targets).item()
            all_losses.append(loss)
            
    return all_losses

def analyze_thresholds(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)
    
    thresholds = np.linspace(0, 50, 100) # De 0 a 50
    results = []
    
    for th in thresholds:
        preds = scores > th
        
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': th,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn
        })
        
    return results

def main():
    logger.info("🔍 Threshold Optimization for High Recall")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = torch.load(f"{MODEL_DIR}/config.pt", weights_only=False)
    model = LogGPT(config)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/loggpt_weights.pt", weights_only=False))
    model.to(DEVICE)
    model.eval()
    
    df = pl.read_csv(str(DATA_FILE), infer_schema_length=INFER_SCHEMA_LENGTH)
    
    # Usar same split logic
    normal_ids = df.filter(pl.col(LABEL_COL) == 0)[SESSION_ID_COL].unique().to_list()
    anom_ids = df.filter(pl.col(LABEL_COL) == 1)[SESSION_ID_COL].unique().to_list()
    _, test_val_ids = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    _, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)
    
    # Usar SAMPLE para otimização (não precisa ser full dataset 35k sessions, demora muito)
    # Vamos usar as 500 do debug anterior (já vimos resultados) + mais algumas
    # Mix: 200 normal + 200 anomalias
    
    sample_norm = test_norm_ids[:200]
    sample_anom = anom_ids[:200] # Anomalias são poucas? Usar todas disponíveis
    session_ids = sample_anom + sample_norm
    
    logger.info(f"Using {len(session_ids)} sessions for threshold analysis.")
    
    scores, labels = get_window_scores(df, session_ids, tokenizer, model, DEVICE)
    
    results = analyze_thresholds(scores, labels)
    
    print("\n=== THRESHOLD ANALYSIS ===")
    print(f"{'Threshold':<10} | {'Recall':<10} | {'Precision':<10} | {'F1':<10}")
    print("-" * 50)
    
    best_f1 = 0
    best_recall_th = 0
    target_recall_found = False
    
    for r in results:
        # Imprimir alguns pontos chave
        if r['threshold'] % 5 == 0 or r['recall'] > 0.9:
             print(f"{r['threshold']:<10.2f} | {r['recall']:<10.4f} | {r['precision']:<10.4f} | {r['f1']:<10.4f}")
        
        if r['f1'] > best_f1:
            best_f1 = r['f1']
            
        if not target_recall_found and r['recall'] < 0.9:
            # Acabamos de descer de 0.9
            # O anterior era o bom?
            pass
            
    # Find optimal for Recall > 0.9
    candidates = [r for r in results if r['recall'] >= 0.90]
    if candidates:
        # Get one with best precision
        best = max(candidates, key=lambda x: x['precision'])
        logger.info(f"\n✅ FOUND OPTIMAL THRESHOLD FOR RECALL > 90%:")
        logger.info(f"Threshold: {best['threshold']:.4f}")
        logger.info(f"Recall: {best['recall']:.4f}")
        logger.info(f"Precision: {best['precision']:.4f}")
        
        # Salvar
        with open("threshold_recall_90.json", "w") as f:
            json.dump({"threshold": best['threshold']}, f)
    else:
        logger.warning("❌ Could not reach 90% Recall within range 0-50.")

if __name__ == "__main__":
    main()
