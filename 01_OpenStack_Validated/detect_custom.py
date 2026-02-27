"""
LogGPT Detection for OpenStack — Identical to HDFS detect_hdfs.py.

Detection Logic (Top-K):
1. Feed WHOLE session through model in one pass
2. Softmax → probabilities → Top-K predictions at each position  
3. If the ACTUAL next token is NOT in Top-K → that step is anomalous
4. If ANY step is anomalous → session is anomalous
"""

import torch
import torch.nn.functional as F
import polars as pl
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import json
import os
import hashlib

from model import LogGPT, GPTConfig
from dataset import load_openstack_data, prepare_session_strings
from config import (
    MODEL_NAME, MODEL_DIR, DEVICE, DATA_FILE,
    BLOCK_SIZE, BATCH_SIZE, SEED,
    SKIP_START_LOGS, LOG_COLUMN,
    INFER_SCHEMA_LENGTH, TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT,
    LOG_DESC_MAX_LEN,
    set_seeds
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

K = 5  # Top-K parameter (same as HDFS)
PAD_TOKEN_ID = 50256  # GPT2 EOS/PAD token


class OpenStackTestDataset(Dataset):
    """Test dataset — same structure as HDFS HDFSTestDataset."""
    
    def __init__(self, sessions_df, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = sessions_df
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.row(idx, named=True)
        seq_str = row['EventTemplate']
        label = row['label']
        test_id = row['test_id']
        
        # Truncate to maximum standard GPT2 length to prevent token length warnings
        tokens = self.tokenizer.encode(seq_str, truncation=True, max_length=1024)
        
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'test_id': test_id,
            'seq_len': len(tokens),
            'event_template': seq_str,
        }


def collate_fn(batch):
    """Dynamic padding (same as HDFS)."""
    max_len = max(x['seq_len'] for x in batch)
    
    padded_ids = torch.full((len(batch), max_len), PAD_TOKEN_ID, dtype=torch.long)
    labels = []
    test_ids = []
    event_templates = []
    
    for i, x in enumerate(batch):
        l = x['seq_len']
        padded_ids[i, :l] = x['input_ids']
        labels.append(x['label'])
        test_ids.append(x['test_id'])
        event_templates.append(x['event_template'])
    
    return {
        'input_ids': padded_ids,
        'label': torch.tensor(labels, dtype=torch.long),
        'test_id': test_ids,
        'event_template': event_templates,
    }


def main():
    logger.info(f"🚀 LogGPT Detection (Top-{K} — HDFS-style)")
    set_seeds()
    
    # 1. Load Model & Tokenizer
    logger.info("Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = torch.load(f"{MODEL_DIR}/config.pt", weights_only=False)
    if not hasattr(config, 'dropout'):
        config.dropout = 0.0
    
    model = LogGPT(config)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/loggpt_weights.pt", weights_only=False))
    model.to(DEVICE)
    model.eval()
    
    # 2. Prepare Test Data
    logger.info("Preparing Test Data...")
    df = load_openstack_data()
    
    # Get ALL sessions (both normal and anomaly)
    all_sessions = prepare_session_strings(df)
    
    # Extract error information for anomalous sessions
    error_info = (
        df.sort("timestamp")
        .group_by("test_id")
        .agg([
            pl.col("anom_label").max().alias("is_anomaly"),
            pl.col("EventTemplate").filter(pl.col("anom_label") == 1).first().alias("error_template"),
            pl.col("EventTemplate").filter(pl.col("anom_label") == 1).first().alias("first_error_template"),
            pl.col("timestamp").filter(pl.col("anom_label") == 1).first().alias("first_error_timestamp"),
            pl.col("EventId").filter(pl.col("anom_label") == 1).first().alias("first_error_eventid"),
            # Index (0-based) of the first anomalous log within the sorted session
            pl.col("anom_label").cum_sum().eq(1).arg_true().first().alias("first_error_index"),
        ])
    )
    error_info_dict = {row["test_id"]: row for row in error_info.iter_rows(named=True)}

    # Build per-session ordered timestamp list so we can map token-step → real timestamp.
    # The tokenizer joins EventTemplates with spaces, so token steps correspond to event indices.
    session_timestamps_dict = {}
    for tid, grp in df.sort("timestamp").group_by("test_id", maintain_order=True):
        # grp is a Polars DataFrame for this session, already sorted by timestamp
        timestamps = grp["timestamp"].to_list()
        session_timestamps_dict[tid] = timestamps
    
    # Split normal IDs for test (same split as training used)
    normal_ids = df.filter(pl.col("anom_label") == 0)["test_id"].unique().to_list()
    anom_ids = df.filter(pl.col("anom_label") == 1)["test_id"].unique().to_list()
    
    _, test_val_ids = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    _, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)
    
    # Build test set: test_norm_ids (label=0) + all anom_ids (label=1)
    test_ids_set = set(test_norm_ids + anom_ids)
    test_sessions = all_sessions.filter(pl.col("test_id").is_in(list(test_ids_set)))
    
    n_normal_test = len([tid for tid in test_norm_ids if tid in set(test_sessions["test_id"].to_list())])
    n_anom_test = len([tid for tid in anom_ids if tid in set(test_sessions["test_id"].to_list())])
    
    logger.info(f"Test Set: {n_normal_test} Normal, {n_anom_test} Anomaly sessions")
    
    # Create test dataset & loader (same as HDFS)
    test_ds = OpenStackTestDataset(test_sessions, tokenizer, BLOCK_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 3. Detection (Top-K — IDENTICAL to HDFS detect_hdfs.py lines 100-170)
    results = []
    
    logger.info(f"Running Detection (Top-{K})...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Detecting"):
            input_ids = batch['input_ids'].to(DEVICE)
            batch_labels = batch['label'].cpu().numpy()
            test_ids = batch['test_id']
            event_templates = batch['event_template']
            
            # Forward pass
            logits, _ = model(input_ids)
            
            # Targets = input shifted by 1 (same as HDFS)
            targets = input_ids[:, 1:]
            preds = logits[:, :-1, :]
            
            # Top-K check (same as HDFS)
            probs = torch.softmax(preds, dim=-1)
            _, topk_inds = torch.topk(probs, K, dim=-1)
            
            # Check if target is in Top-K
            matches = (topk_inds == targets.unsqueeze(-1)).any(dim=-1)
            
            # Mask out padding
            target_mask = (targets != PAD_TOKEN_ID)
            
            # Anomaly = NOT in Top-K AND not padding
            valid_anomalies = (~matches) & target_mask
            
            # Session is anomalous if ANY step is anomalous
            is_anom_pred = valid_anomalies.any(dim=1).cpu().numpy()
            
            # First anomaly index (lead time proxy)
            first_indices = valid_anomalies.int().argmax(dim=1).cpu().numpy()
            
            for i in range(len(test_ids)):
                tid = test_ids[i]
                true_label = batch_labels[i]
                pred_label = 1 if is_anom_pred[i] else 0
                first_step = int(first_indices[i]) if is_anom_pred[i] else -1
                n_events = event_templates[i].count(' ') + 1
                
                # Get error information from error_info_dict
                err_info = error_info_dict.get(tid, {})
                error_template = err_info.get('error_template', None)
                first_error_template = err_info.get('first_error_template', None)
                first_error_timestamp = err_info.get('first_error_timestamp', None)
                first_error_eventid = err_info.get('first_error_eventid', None)
                first_error_index = err_info.get('first_error_index', None)
                
                log_desc = event_templates[i][:LOG_DESC_MAX_LEN]
                log_hash = hashlib.md5(log_desc.encode()).hexdigest()[:8]
                
                # ── Lead Time Calculation ────────────────────────────────────────
                # We use the real per-event timestamps so that:
                #   alert_timestamp  = timestamp of the event at first_anomaly_step
                #   lead_time        = first_error_timestamp - alert_timestamp
                # Positive lead_time → model detected the anomaly BEFORE the real error.
                # Negative lead_time → model detected AFTER (reactive).
                alert_timestamp_str = None
                lead_time_seconds = None
                lead_time_minutes = None
                alert_step_before_error = None

                if pred_label == 1 and first_step >= 0 and first_error_timestamp is not None:
                    session_ts = session_timestamps_dict.get(tid, [])
                    if first_step < len(session_ts):
                        # Map top-k token step to real event timestamp
                        # first_step is 0-indexed position in the token sequence.
                        # Because we join EventTemplates with spaces, step N corresponds
                        # to event N (the token that follows N words/events of context).
                        alert_ts = pd.to_datetime(session_ts[first_step])
                        error_ts = pd.to_datetime(first_error_timestamp)
                        delta_sec = (error_ts - alert_ts).total_seconds()
                        alert_timestamp_str = str(alert_ts)
                        lead_time_seconds = delta_sec
                        lead_time_minutes = delta_sec / 60.0

                    if first_error_index is not None and first_step >= 0:
                        # Positive = model detected N events before the real error
                        alert_step_before_error = int(first_error_index) - first_step
                # ─────────────────────────────────────────────────────────────────
                
                results.append({
                    'test_id': tid,
                    'label': int(true_label),
                    'predicted': pred_label,
                    'first_anomaly_step': first_step,
                    'n_events': n_events,
                    'log_desc': log_desc,
                    'log_hash': log_hash,
                    'error_template': error_template,
                    'first_error_template': first_error_template,
                    'first_error_timestamp': str(first_error_timestamp) if first_error_timestamp is not None else None,
                    'first_error_index': first_error_index,
                    'first_error_eventid': first_error_eventid,
                    # ── Lead time fields (corrected) ──
                    'alert_timestamp': alert_timestamp_str,        # Timestamp real da detecção
                    'lead_time_seconds': lead_time_seconds,        # >0 = antecipou, <0 = reativo
                    'lead_time_minutes': lead_time_minutes,
                    'alert_step_before_error': alert_step_before_error,  # em passos de evento
                })
    
    # 4. Metrics (same as HDFS)
    y_true = [r['label'] for r in results]
    y_pred = [r['predicted'] for r in results]
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    acc = accuracy_score(y_true, y_pred)
    
    tp = sum(1 for r in results if r['label'] == 1 and r['predicted'] == 1)
    fn = sum(1 for r in results if r['label'] == 1 and r['predicted'] == 0)
    fp = sum(1 for r in results if r['label'] == 0 and r['predicted'] == 1)
    tn = sum(1 for r in results if r['label'] == 0 and r['predicted'] == 0)
    
    logger.info("")
    logger.info(f"� Results (Top-{K}):")
    logger.info(f"   Precision: {precision:.4f}")
    logger.info(f"   Recall:    {recall:.4f}")
    logger.info(f"   F1 Score:  {f1:.4f}")
    logger.info(f"   Accuracy:  {acc:.4f}")
    logger.info(f"   TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"   Confusion Matrix:\n{cm}")
    
    # 5. Report
    report_lines = []
    report_lines.append("📊 LOGGPT FAILURE DIVERSITY REPORT")
    report_lines.append("===================================\n")
    report_lines.append(f"Detection Method: Top-{K}")
    report_lines.append(f"Precision: {precision:.4f}")
    report_lines.append(f"Recall:    {recall:.4f}")
    report_lines.append(f"F1 Score:  {f1:.4f}")
    report_lines.append(f"Accuracy:  {acc:.4f}")
    report_lines.append(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}\n")
    
    # Detected anomaly patterns
    detected = [r for r in results if r['label'] == 1 and r['predicted'] == 1]
    hashes = set(r['log_hash'] for r in detected)
    report_lines.append(f"🔹 Found {len(hashes)} distinct failure patterns detected.\n")
    
    for h in sorted(hashes):
        group = [r for r in detected if r['log_hash'] == h]
        report_lines.append(f"📌 Pattern: {h}...")
        report_lines.append(f"   Count: {len(group)} sessions")
        report_lines.append(f"   Example: ID {group[0]['test_id']} (Step {group[0]['first_anomaly_step']})")
        report_lines.append(f"   ------------------------------------------------")
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_metrics_detailed.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"✅ Report saved to {report_path}")
    
    # Save results to JSON for advanced report generation
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_metrics_detailed.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            'detection_method': f"Top-{K}",
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': acc,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ JSON results saved to {json_path}")


if __name__ == "__main__":
    main()
