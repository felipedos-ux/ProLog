
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

from dataset import load_bgl_data
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

# OTIMIZAÇÃO: Batch size para saturar GPU
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

def evaluate_session_ultra_fast(
    tid: str,
    label: int,
    session_df: pl.DataFrame,
    model: LogGPT,
    tokenizer: PreTrainedTokenizer,
    threshold: float,
    device: str
) -> Optional[Dict[str, Any]]:
    """Versão ULTRA otimizada com batch processing."""
    ts_col = TIMESTAMP_COL
    session_df = session_df.sort(ts_col)
    templates = session_df["EventTemplate"].to_list()
    raw_ts = session_df[ts_col].to_list()
    
    try:
        timestamps = [pd.to_datetime(ts, unit='s') for ts in raw_ts]
    except ValueError as e:
        logger.error(f"❌ Error parsing timestamps for session {tid}: {e}")
        return None
    
    failure_ts = timestamps[-1]
    
    # Preparar todas as sequências
    sequences = []
    context_ids = []
    MAX_CONTEXT_LEN = model.config.block_size
    
    for i, current_log in enumerate(templates):
        if current_log is None:
            current_log = ""
        text = (" \n " if i > 0 else "") + current_log
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
        return {
            "test_id": tid,
            "is_detected": False,
            "label": label,
            "lead_time": 0.0,
            "alert_loss": 0.0,
            "final_log": templates[-1][:LOG_DESC_MAX_LEN] + "..."
        }
    
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
    
    result = {
        "test_id": tid,
        "is_detected": is_detected,
        "label": label,
        "lead_time": 0.0,
        "alert_loss": first_alert_loss,
        "final_log": templates[-1][:LOG_DESC_MAX_LEN] + "..."
    }
    
    if is_detected:
        lead = (failure_ts - first_alert_ts).total_seconds() / 60
        result["lead_time"] = lead
    
    return result

def main():
    logger.info("🚀 LogGPT ULTRA-FAST Evaluation (GPU Optimized)")
    
    logger.info("Loading Tokenizer & Model...")
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
            logger.info(f"Using Adaptive Threshold (K={conf.get('k_sigma')}): {THRESHOLD:.4f}")
    else:
        logger.warning(f"No adaptive config found. Using default threshold: {THRESHOLD}")
    
    logger.info("Loading Data...")
    df = pl.read_csv(str(DATA_FILE), infer_schema_length=INFER_SCHEMA_LENGTH)
    
    normal_ids = df.filter(pl.col(LABEL_COL) == 0)[SESSION_ID_COL].unique().to_list()
    anom_ids = df.filter(pl.col(LABEL_COL) == 1)[SESSION_ID_COL].unique().to_list()
    
    _, test_val_ids = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    _, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)
    
    logger.info(f"Test Set: {len(test_norm_ids)} Normal, {len(anom_ids)} Anomalies")
    
    eval_list = [(tid, 1) for tid in anom_ids] + [(tid, 0) for tid in test_norm_ids]
    
    results = []
    
    for tid, label in tqdm(eval_list, desc="Ultra-Fast Evaluation"):
        session_df = df.filter(pl.col(SESSION_ID_COL) == tid)
        res = evaluate_session_ultra_fast(tid, label, session_df, model, tokenizer, THRESHOLD, DEVICE)
        if res:
            results.append(res)
    
    # Metrics
    tp = sum(1 for r in results if r['label'] == 1 and r['is_detected'])
    fn = sum(1 for r in results if r['label'] == 1 and not r['is_detected'])
    fp = sum(1 for r in results if r['label'] == 0 and r['is_detected'])
    tn = sum(1 for r in results if r['label'] == 0 and not r['is_detected'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    positive_leads = [r for r in results if r['label'] == 1 and r['is_detected'] and r['lead_time'] > 0]
    
    avg_lead = np.mean([r['lead_time'] for r in positive_leads]) if positive_leads else 0.0
    
    logger.info("")
    logger.info(f"📊 Final Results (Threshold {THRESHOLD:.4f}):")
    logger.info(f"   Precision: {precision:.4f}")
    logger.info(f"   Recall:    {recall:.4f}")
    logger.info(f"   F1 Score:  {f1:.4f}")
    logger.info(f"   Avg Lead:  {avg_lead:.2f} min")
    
    # Save detailed report (simplified)
    report = []
    report.append(f"Precision: {precision:.4f}")
    report.append(f"Recall: {recall:.4f}")
    report.append(f"F1: {f1:.4f}")
    report.append(f"Avg Lead Time: {avg_lead:.2f} min")
    
    with open("results_metrics_detailed.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    logger.info("✅ Report saved to results_metrics_detailed.txt")

if __name__ == "__main__":
    main()
