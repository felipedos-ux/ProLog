"""
Optimized Detection with Checkpointing and Chunked Processing.
Designed to handle long-running processes without interruption.
"""

import torch
import torch.nn.functional as F
import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import os
import pickle
from sklearn.model_selection import train_test_split
from datetime import datetime

from model import LogGPT
from config import (
    MODEL_NAME, MODEL_DIR, DEVICE, DATA_FILE,
    SKIP_START_LOGS, THRESHOLD as DEFAULT_THRESHOLD,
    INFER_SCHEMA_LENGTH, TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT,
    SESSION_ID_COL, TIMESTAMP_COL, TEMPLATE_COL, LABEL_COL
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Configuration (Optimized for RTX 3080 Ti + Ryzen 3600)
CHUNK_SIZE = 1000  # Process 1000 sessions at a time (2x increase)
SAMPLE_RATIO = 1.0  # Use 100% of sessions for complete results
CHECKPOINT_FILE = "detection_checkpoint.pkl"
RESULTS_FILE = "detection_results_partial.pkl"


def evaluate_session(tid, label, session_df, model, tokenizer, threshold, device):
    """Evaluates a single session for anomalies."""
    session_df = session_df.sort(TIMESTAMP_COL)
    templates = session_df[TEMPLATE_COL].to_list()
    raw_ts = session_df[TIMESTAMP_COL].to_list()
    
    try:
        timestamps = [pd.to_datetime(ts) for ts in raw_ts]
    except ValueError:
        return None
        
    failure_ts = timestamps[-1]
    
    is_detected = False
    first_alert_ts = None
    first_alert_loss = 0.0
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
            
        x = torch.tensor(input_seq, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = model(x)
        
        target_indices = range(target_start_idx, len(input_seq))
        logit_indices = [idx - 1 for idx in target_indices]
        
        if not logit_indices or logit_indices[0] < 0 or logit_indices[-1] >= logits.size(1):
            loss_val = 0.0
        else:
            relevant_logits = logits[0, logit_indices, :]
            relevant_targets = torch.tensor(input_seq[target_start_idx:], dtype=torch.long, device=device)
            
            if relevant_logits.shape[0] != relevant_targets.shape[0]:
                loss_val = 0.0
            else:
                loss_val = F.cross_entropy(relevant_logits, relevant_targets).item()
        
        if loss_val > threshold:
            is_detected = True
            first_alert_ts = timestamps[i]
            first_alert_loss = loss_val
            break
        
        context_ids.extend(new_ids)
        if len(context_ids) > MAX_CONTEXT_LEN:
            context_ids = context_ids[-MAX_CONTEXT_LEN:]
    
    result = {
        "session_id": tid,
        "is_detected": is_detected,
        "label": label,
        "lead_time": 0.0,
        "alert_loss": first_alert_loss,
        "final_log": str(templates[-1])[:50] + "..."
    }
    
    if is_detected:
        lead = (failure_ts - first_alert_ts).total_seconds() / 60
        result["lead_time"] = lead
        
    return result


def load_checkpoint():
    """Load checkpoint if exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return {"processed_sessions": set(), "results": []}


def save_checkpoint(checkpoint):
    """Save checkpoint."""
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)


def save_final_results(results):
    """Save final results."""
    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump(results, f)


def generate_report(results, threshold):
    """Generate final report from results."""
    tp = sum(1 for r in results if r['label'] == 1 and r['is_detected'])
    fn = sum(1 for r in results if r['label'] == 1 and not r['is_detected'])
    fp = sum(1 for r in results if r['label'] == 0 and r['is_detected'])
    tn = sum(1 for r in results if r['label'] == 0 and not r['is_detected'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    positive_leads = [r for r in results if r['label'] == 1 and r['is_detected'] and r['lead_time'] > 0]
    
    tp_anticipated = len(positive_leads)
    avg_lead = np.mean([r['lead_time'] for r in positive_leads]) if positive_leads else 0.0
    max_lead = np.max([r['lead_time'] for r in positive_leads]) if positive_leads else 0.0
    median_lead = np.median([r['lead_time'] for r in positive_leads]) if positive_leads else 0.0
    
    report = []
    report.append("=" * 60)
    report.append("📊 HDFS DETECTION RESULTS (Optimized Pipeline)")
    report.append("=" * 60)
    report.append("")
    report.append(f"⚙️  Configuration:")
    report.append(f"   Threshold: {threshold:.4f} (Fixed)")
    report.append(f"   Sample Ratio: {SAMPLE_RATIO*100:.0f}%")
    report.append(f"   Total Sessions Evaluated: {len(results)}")
    report.append("")
    report.append(f"📈 METRICS:")
    report.append(f"   Precision:  {precision:.4f}")
    report.append(f"   Recall:     {recall:.4f}")
    report.append(f"   F1 Score:   {f1:.4f}")
    report.append(f"   TP={tp} | FN={fn} | FP={fp} | TN={tn}")
    report.append("")
    report.append(f"🎯 LEAD TIME ANALYSIS:")
    report.append(f"   Anomalies Detected: {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)" if (tp+fn) > 0 else "   No anomalies")
    report.append(f"   Anticipated (Lead>0): {tp_anticipated}/{tp} ({tp_anticipated/tp*100:.1f}%)" if tp > 0 else "   N/A")
    
    if tp_anticipated > 0:
        report.append("")
        report.append(f"   📊 Lead Time Statistics (positive leads only):")
        report.append(f"      Max:    {max_lead:.2f} min")
        report.append(f"      Avg:    {avg_lead:.2f} min")
        report.append(f"      Median: {median_lead:.2f} min")
    
    report.append("")
    report.append("=" * 60)
    
    full_report = "\n".join(report)
    print(full_report)
    
    with open("results_chunked.txt", "w", encoding="utf-8") as f:
        f.write(full_report)
    
    logger.info("✅ Report saved to results_chunked.txt")
    return precision, recall, f1


def main():
    logger.info("🚀 Optimized HDFS Detection (Chunked Processing)")
    logger.info(f"   Chunk Size: {CHUNK_SIZE}")
    logger.info(f"   Sample Ratio: {SAMPLE_RATIO*100:.0f}%")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_sessions = checkpoint["processed_sessions"]
    results = checkpoint["results"]
    
    if processed_sessions:
        logger.info(f"📂 Resuming from checkpoint: {len(processed_sessions)} sessions already processed")
    
    # Load Model
    logger.info("Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = torch.load(f"{MODEL_DIR}/config.pt", weights_only=False)
    if not hasattr(config, 'dropout'):
        config.dropout = 0.0
    
    model = LogGPT(config)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/hdfs_loggpt.pt", weights_only=False))
    model.to(DEVICE)
    model.eval()
    
    THRESHOLD = DEFAULT_THRESHOLD
    if os.path.exists("threshold_config.json"):
        with open("threshold_config.json", "r") as f:
            conf = json.load(f)
            THRESHOLD = conf.get("threshold", DEFAULT_THRESHOLD)
    logger.info(f"Using Threshold: {THRESHOLD:.4f}")
    
    # Load Data
    logger.info("Loading Data...")
    df = pl.read_csv(str(DATA_FILE), infer_schema_length=INFER_SCHEMA_LENGTH)
    
    normal_ids = df.filter(pl.col(LABEL_COL) == 0)[SESSION_ID_COL].unique().to_list()
    anom_ids = df.filter(pl.col(LABEL_COL) == 1)[SESSION_ID_COL].unique().to_list()
    
    _, test_val_ids = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    _, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)
    
    # Sample for faster execution
    import random
    random.seed(42)
    test_norm_sample = random.sample(test_norm_ids, int(len(test_norm_ids) * SAMPLE_RATIO))
    anom_sample = random.sample(anom_ids, int(len(anom_ids) * SAMPLE_RATIO))
    
    logger.info(f"Test Set (Sampled): {len(test_norm_sample)} Normal, {len(anom_sample)} Anomalies")
    
    eval_list = [(tid, 1) for tid in anom_sample] + [(tid, 0) for tid in test_norm_sample]
    
    # Filter out already processed
    eval_list = [(tid, label) for tid, label in eval_list if tid not in processed_sessions]
    
    logger.info(f"Remaining to process: {len(eval_list)} sessions")
    
    # Process in chunks
    for chunk_start in range(0, len(eval_list), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(eval_list))
        chunk = eval_list[chunk_start:chunk_end]
        
        logger.info(f"Processing chunk {chunk_start//CHUNK_SIZE + 1}/{(len(eval_list)-1)//CHUNK_SIZE + 1} ({len(chunk)} sessions)")
        
        for tid, label in tqdm(chunk, desc=f"Chunk {chunk_start//CHUNK_SIZE + 1}"):
            session_df = df.filter(pl.col(SESSION_ID_COL) == tid)
            res = evaluate_session(tid, label, session_df, model, tokenizer, THRESHOLD, DEVICE)
            if res:
                results.append(res)
                processed_sessions.add(tid)
        
        # Save checkpoint after each chunk
        checkpoint = {"processed_sessions": processed_sessions, "results": results}
        save_checkpoint(checkpoint)
        logger.info(f"✅ Checkpoint saved ({len(results)} results)")
    
    # Generate final report
    logger.info("Generating final report...")
    generate_report(results, THRESHOLD)
    
    # Save final results
    save_final_results(results)
    
    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logger.info("🗑️  Checkpoint file cleaned up")
    
    logger.info("✅ Detection complete!")


if __name__ == "__main__":
    main()
