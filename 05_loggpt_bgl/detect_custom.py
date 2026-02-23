
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

from dataset import load_bgl_data
from model import LogGPT, GPTConfig
from sklearn.model_selection import train_test_split

from config import (
    MODEL_NAME, MODEL_DIR, DEVICE, DATA_FILE, 
    SKIP_START_LOGS, THRESHOLD as DEFAULT_THRESHOLD,
    INFER_SCHEMA_LENGTH, TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT, LOG_DESC_MAX_LEN,
    SESSION_ID_COL, LABEL_COL, TIMESTAMP_COL  # BGL-specific
)
from utils.logger import setup_logger

# Setup Logger
logger = setup_logger(__name__)

# Constants
THRESHOLD = DEFAULT_THRESHOLD # Will be overwritten by optimal_threshold.txt

def evaluate_session(
    tid: str,
    label: int,
    session_df: pl.DataFrame,
    model: LogGPT,
    tokenizer: PreTrainedTokenizer,
    threshold: float,
    device: str
) -> Optional[Dict[str, Any]]:
    """
    Evaluates a single session for anomalies.
    
    Args:
        tid: Test ID
        label: 1 for Anomaly, 0 for Normal
        session_df: Polars DataFrame for the session
        model: Trained LogGPT model
        tokenizer: Tokenizer
        threshold: Loss threshold for detection
        device: 'cuda' or 'cpu'
        
    Returns:
        Dict with detection metrics if detected, else None (or empty dict context)
    """
    # Timestamps (BGL always uses 'timestamp' - Unix epoch)
    ts_col = TIMESTAMP_COL
    
    # CRITICAL: Sort by timestamp to ensure causal order
    session_df = session_df.sort(ts_col)
    templates = session_df["EventTemplate"].to_list()
    
    raw_ts = session_df[ts_col].to_list()
    
    try:
        # BGL uses Unix epoch seconds
        timestamps = [pd.to_datetime(ts, unit='s') for ts in raw_ts]
    except ValueError as e:
        logger.error(f"❌ Error parsing timestamps for session {tid}: {e}")
        return None # Skip bad session
        
    # Ensure sorting (P1.4 / P2.4)
    if timestamps != sorted(timestamps):
        logger.warning(f"⚠️ Session {tid} timestamps NOT sorted! Potentially invalid causal analysis.")
        
    failure_ts = timestamps[-1]
    
    # Detection State
    is_detected = False
    first_alert_ts = None
    first_alert_loss = 0.0
    context_ids = []
    
    MAX_CONTEXT_LEN = model.config.block_size
    
    for i, current_log in enumerate(templates):
        if current_log is None:
            current_log = ""
        text = (" \n " if i > 0 else "") + current_log
        new_ids = tokenizer.encode(text)
        
        # Logic Update: Update context but skip inference for Start Logs
        if i < SKIP_START_LOGS:
            context_ids.extend(new_ids)
            if len(context_ids) > MAX_CONTEXT_LEN:
                context_ids = context_ids[-MAX_CONTEXT_LEN:]
            continue
        
        if i == 0:
            context_ids.extend(new_ids)
            continue
            
        # Prepare Input
        full_seq = context_ids + new_ids
        if len(full_seq) > MAX_CONTEXT_LEN:
            input_seq = full_seq[-MAX_CONTEXT_LEN:]
            target_start_idx = len(input_seq) - len(new_ids)
        else:
            input_seq = full_seq
            target_start_idx = len(context_ids)
            
        x = torch.tensor(input_seq, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = model(x)
        
        # Extract relevant logits
        target_indices = range(target_start_idx, len(input_seq))
        logit_indices = [idx - 1 for idx in target_indices]
        
        # Robust Shape Checks (P2.3)
        if not logit_indices:
             loss_val = 0.0
        else:
            # 1. Check Index Bounds
            if logit_indices[0] < 0 or logit_indices[-1] >= logits.size(1):
                 logger.error(f"Index out of bounds in session {tid} log {i}")
                 loss_val = 0.0
            else:
                relevant_logits = logits[0, logit_indices, :]
                relevant_targets = torch.tensor(input_seq[target_start_idx:], dtype=torch.long, device=device)
                
                # 2. Check Shape Match
                if relevant_logits.shape[0] != relevant_targets.shape[0]:
                    logger.error(f"Shape mismatch in session {tid}: {relevant_logits.shape} vs {relevant_targets.shape}")
                    loss_val = 0.0
                else:
                    loss_val = F.cross_entropy(relevant_logits, relevant_targets).item()
            
            # Check Threshold
            if loss_val > threshold:
                is_detected = True
                first_alert_ts = timestamps[i]
                first_alert_loss = loss_val
                break # Stop at first alert
        
        # Update Context
        context_ids.extend(new_ids)
        if len(context_ids) > MAX_CONTEXT_LEN:
             context_ids = context_ids[-MAX_CONTEXT_LEN:]
    
    # Result Construction
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
    logger.info("🚀 LogGPT Full Evaluation (Structure & Type Hints Applied)")
    
    # 1. Load Tokenizer & Model
    logger.info("Loading Tokenizer & Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = torch.load(f"{MODEL_DIR}/config.pt", weights_only=False)
    
    # Backwards Compatibility: Inject dropout if missing from old checkpoints
    if not hasattr(config, 'dropout'):
        logger.warning("Loaded config missing 'dropout' attribute. Defaulting to 0.0.")
        config.dropout = 0.0
    
    logger.info(f"Config Attributes: {dir(config)}")
    model = LogGPT(config)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/loggpt_weights.pt", weights_only=False))
    model.to(DEVICE)
    model.eval()

    # Load Adaptive Threshold
    THRESHOLD = 5.0 # Fallback
    if os.path.exists("threshold_config.json"):
        with open("threshold_config.json", "r") as f:
            conf = json.load(f)
            THRESHOLD = conf.get("threshold", 5.0)
            logger.info(f"Using Adaptive Threshold (K={conf.get('k_sigma')}): {THRESHOLD:.4f}")
    else:
        logger.warning(f"No adaptive config found. Using default threshold: {THRESHOLD}")
    
    # 2. Load Data
    logger.info("Loading Data...")
    df = pl.read_csv(str(DATA_FILE), infer_schema_length=INFER_SCHEMA_LENGTH)
    


    # Split Logic
    normal_ids = df.filter(pl.col(LABEL_COL) == 0)[SESSION_ID_COL].unique().to_list()
    anom_ids = df.filter(pl.col(LABEL_COL) == 1)[SESSION_ID_COL].unique().to_list()
    
    # Split Normal
    _, test_val_ids = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    _, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)
    
    logger.info(f"Test Set: {len(test_norm_ids)} Normal, {len(anom_ids)} Anomalies")
    
    # Evaluation
    eval_list = [(tid, 1) for tid in anom_ids] + [(tid, 0) for tid in test_norm_ids]
    
    results = []
    
    for tid, label in tqdm(eval_list, desc="Evaluating"):
        session_df = df.filter(pl.col(SESSION_ID_COL) == tid)
        res = evaluate_session(tid, label, session_df, model, tokenizer, THRESHOLD, DEVICE)
        if res:
            results.append(res)
            
    # Metrics Calculation
    tp = sum(1 for r in results if r['label'] == 1 and r['is_detected'])
    fn = sum(1 for r in results if r['label'] == 1 and not r['is_detected'])
    fp = sum(1 for r in results if r['label'] == 0 and r['is_detected'])
    tn = sum(1 for r in results if r['label'] == 0 and not r['is_detected'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Anticipation Metrics
    positive_leads = [r for r in results if r['label'] == 1 and r['is_detected'] and r['lead_time'] > 0]
    negative_leads = [r for r in results if r['label'] == 1 and r['is_detected'] and r['lead_time'] <= 0]
    
    tp_anticipated = len(positive_leads)
    tp_not_anticipated = len(negative_leads)
    
    avg_lead_positive = np.mean([r['lead_time'] for r in positive_leads]) if positive_leads else 0.0
    max_lead_positive = np.max([r['lead_time'] for r in positive_leads]) if positive_leads else 0.0
    median_lead_positive = np.median([r['lead_time'] for r in positive_leads]) if positive_leads else 0.0

    logger.info("")
    logger.info(f"📊 Final Results (Threshold {THRESHOLD}):")
    logger.info(f"   Precision: {precision:.4f}")
    logger.info(f"   Recall:    {recall:.4f}")
    logger.info(f"   F1 Score:  {f1:.4f}")
    
    # 6. Report Generation
    report = []
    report.append("📊 LOGGPT FAILURE DIVERSITY REPORT")
    report.append("===================================")
    
    report.append("")
    report.append("🎯 ANTICIPATION ANALYSIS (Lead Time > 0)")
    report.append("------------------------------------------")
    report.append(f"   Total Detections:        {tp} (100%)")
    report.append(f"   ✅ Anticipated (Lead>0):  {tp_anticipated} ({tp_anticipated/tp*100:.1f}%)" if tp > 0 else "   ✅ Anticipated: 0")
    report.append(f"   ⚠️  Not Anticipated (Lead≤0): {tp_not_anticipated} ({tp_not_anticipated/tp*100:.1f}%)" if tp > 0 else "   ⚠️ Not Anticipated: 0")
    report.append("")
    
    if tp_anticipated > 0:
        report.append(f"   📈 Anticipation Metrics (Only Lead > 0):")
        report.append(f"      Max Lead Time:    {max_lead_positive:.2f} min")
        report.append(f"      Avg Lead Time:    {avg_lead_positive:.2f} min")
        report.append(f"      Median Lead Time: {median_lead_positive:.2f} min")
    report.append("")
    
    # Group by Final Log
    signatures = {}
    for item in results:
        if item['label'] == 1 and item['is_detected']:
            sig = item['final_log']
            if sig not in signatures:
                signatures[sig] = []
            signatures[sig].append(item)
            
    report.append(f"🔹 Diversity check: Found {len(signatures)} distinct failure patterns detected.")
    report.append("")
    
    sorted_sigs = sorted(signatures.items(), key=lambda x: len(x[1]), reverse=True)
    
    for sig, items in sorted_sigs:
        best_lead = max(item['lead_time'] for item in items)
        avg_lead_sig = np.mean([item['lead_time'] for item in items])
        example_id = items[0]['test_id']
        
        report.append(f"📌 Pattern: {sig}")
        report.append(f"   Count:     {len(items)} sessions")
        report.append(f"   Best Lead: {best_lead:.2f} min")
        report.append(f"   Avg Lead:  {avg_lead_sig:.2f} min")
        report.append(f"   Example:   ID {example_id} (Alert Loss: {items[0]['alert_loss']:.2f})")
        report.append("   ------------------------------------------------")

    # Top 10 Best
    if positive_leads:
        # Sort by lead time desc
        top_best = sorted(positive_leads, key=lambda x: x['lead_time'], reverse=True)[:10]
        report.append("")
        report.append("🏆 Top 10 Absolute Best Lead Times")
        report.append("------------------------------------------")
        for i, row in enumerate(top_best, 1):
             report.append(f"{i}. [ID {row['test_id']}] Lead: {row['lead_time']:.2f} min | Loss: {row['alert_loss']:.2f} | Log: {row['final_log']}")

    full_report = "\n".join(report)
    print(full_report)
    
    with open("results_metrics_detailed.txt", "w", encoding="utf-8") as f:
        f.write(full_report)
        
    logger.info("✅ Report saved to results_metrics_detailed.txt")

if __name__ == "__main__":
    main()
