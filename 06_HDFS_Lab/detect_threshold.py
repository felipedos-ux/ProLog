"""
Threshold Detection: Baseline method (same as current 03_HDFS_Benchmark).
Uses cross-entropy loss vs a calibrated threshold.
Serves as the control for comparing with Top-K detection.
"""
import torch
import torch.nn.functional as F
import polars as pl
import pandas as pd
import numpy as np
import json
import time
from transformers import AutoTokenizer

from model import LogGPT
from config import (
    MODEL_NAME, DEVICE, LAB_DATA_DIR,
    SKIP_START_LOGS, LOG_DESC_MAX_LEN, DEFAULT_THRESHOLD,
    SESSION_ID_COL, TIMESTAMP_COL, TEMPLATE_COL, LABEL_COL
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

TEST_FILE = LAB_DATA_DIR / "hdfs_test_subset.csv"


def calibrate_threshold(model, tokenizer, train_path=None, k_sigma=2.0, is_regex=False):
    """
    Calibrate threshold from training data using mean + k*std of losses.
    """
    if is_regex and train_path is None:
        path = LAB_DATA_DIR / "hdfs_train_5k_regex.csv"
    else:
        path = train_path or (LAB_DATA_DIR / "hdfs_train_5k.csv")
    logger.info(f"Calibrating threshold (k_sigma={k_sigma})...")
    
    df = pl.read_csv(str(path), infer_schema_length=10000)
    normal_df = df.filter(pl.col(LABEL_COL) == 0)
    
    # Sample sessions for calibration
    session_ids = normal_df[SESSION_ID_COL].unique().to_list()
    import random
    random.seed(42)
    sample_ids = random.sample(session_ids, min(200, len(session_ids)))
    
    all_losses = []
    model.eval()
    
    MAX_CONTEXT_LEN = model.config.block_size
    
    for tid in sample_ids:
        session_df = normal_df.filter(pl.col(SESSION_ID_COL) == tid).sort(TIMESTAMP_COL)
        templates = session_df[TEMPLATE_COL].to_list()
        
        context_ids = []
        for i, tpl in enumerate(templates):
            if tpl is None:
                tpl = ""
            text = (" \n " if i > 0 else "") + str(tpl)
            new_ids = tokenizer.encode(text)
            
            if i < SKIP_START_LOGS:
                context_ids.extend(new_ids)
                if len(context_ids) > MAX_CONTEXT_LEN:
                    context_ids = context_ids[-MAX_CONTEXT_LEN:]
                continue
            
            if i == 0 or len(context_ids) == 0:
                context_ids.extend(new_ids)
                continue
            
            full_seq = context_ids + new_ids
            if len(full_seq) > MAX_CONTEXT_LEN:
                input_seq = full_seq[-MAX_CONTEXT_LEN:]
                target_start_idx = len(input_seq) - len(new_ids)
            else:
                input_seq = full_seq
                target_start_idx = len(context_ids)
            
            x = torch.tensor([input_seq], dtype=torch.long, device=DEVICE)
            
            with torch.no_grad():
                logits, _, _ = model(x)
            
            target_indices = range(target_start_idx, len(input_seq))
            logit_indices = [idx - 1 for idx in target_indices]
            
            if logit_indices and logit_indices[0] >= 0 and logit_indices[-1] < logits.size(1):
                relevant_logits = logits[0, logit_indices, :]
                relevant_targets = torch.tensor(
                    input_seq[target_start_idx:], dtype=torch.long, device=DEVICE
                )
                if relevant_logits.shape[0] == relevant_targets.shape[0]:
                    loss_val = F.cross_entropy(relevant_logits, relevant_targets).item()
                    all_losses.append(loss_val)
            
            context_ids.extend(new_ids)
            if len(context_ids) > MAX_CONTEXT_LEN:
                context_ids = context_ids[-MAX_CONTEXT_LEN:]
    
    if all_losses:
        mean_loss = np.mean(all_losses)
        std_loss = np.std(all_losses)
        threshold = mean_loss + k_sigma * std_loss
        logger.info(f"  Mean loss: {mean_loss:.4f}, Std: {std_loss:.4f}")
        logger.info(f"  Threshold (mean + {k_sigma}σ): {threshold:.4f}")
        return threshold
    else:
        logger.warning("No losses computed, using default threshold")
        return DEFAULT_THRESHOLD


def evaluate_session_threshold(
    tid, label, session_df, model, tokenizer, threshold, device
):
    """
    Evaluates a single session using cross-entropy threshold detection.
    Same logic as 03_HDFS_Benchmark/detect.py.
    """
    session_df = session_df.sort(TIMESTAMP_COL)
    templates = session_df[TEMPLATE_COL].to_list()
    raw_ts = session_df[TIMESTAMP_COL].to_list()
    
    try:
        timestamps = [pd.to_datetime(ts) for ts in raw_ts]
    except (ValueError, TypeError):
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
        
        x = torch.tensor([input_seq], dtype=torch.long, device=device).unsqueeze(0) if len(torch.tensor([input_seq]).shape) == 1 else torch.tensor([input_seq], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits, _, _ = model(x)
        
        target_indices = range(target_start_idx, len(input_seq))
        logit_indices = [idx - 1 for idx in target_indices]
        
        loss_val = 0.0
        if logit_indices and logit_indices[0] >= 0 and logit_indices[-1] < logits.size(1):
            relevant_logits = logits[0, logit_indices, :]
            relevant_targets = torch.tensor(
                input_seq[target_start_idx:], dtype=torch.long, device=device
            )
            if relevant_logits.shape[0] == relevant_targets.shape[0]:
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
        "final_log": str(templates[-1])[:LOG_DESC_MAX_LEN] + "...",
    }
    
    if is_detected and first_alert_ts:
        lead = (failure_ts - first_alert_ts).total_seconds() / 60
        result["lead_time"] = lead
    
    return result


def run_threshold_detection(
    model, config, threshold=None, test_file=None, experiment_id="",
    train_data_path=None, is_regex=False, k_sigma=2.0
):
    """
    Full threshold-based detection pipeline.
    """
    logger.info("=" * 60)
    logger.info(f"THRESHOLD DETECTION | Experiment {experiment_id}")
    logger.info("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Calibrate threshold if not provided
    if threshold is None:
        threshold = calibrate_threshold(model, tokenizer, train_data_path, k_sigma=k_sigma, is_regex=is_regex)
    
    logger.info(f"Using threshold: {threshold:.4f}")
    
    # Load test data
    if is_regex and test_file is None:
        test_path = LAB_DATA_DIR / "hdfs_test_subset_regex.csv"
    else:
        test_path = test_file or TEST_FILE
    logger.info(f"Loading test data from {test_path}...")
    df = pl.read_csv(str(test_path), infer_schema_length=10000)
    
    # Get session IDs with labels
    session_labels = (
        df.group_by(SESSION_ID_COL)
        .agg(pl.col(LABEL_COL).first())
    )
    
    eval_list = [(row[0], row[1]) for row in session_labels.rows()]
    
    normal_count = sum(1 for _, l in eval_list if l == 0)
    anom_count = sum(1 for _, l in eval_list if l == 1)
    logger.info(f"Evaluating {len(eval_list)} sessions ({normal_count} normal, {anom_count} anomalous)")
    
    # Evaluate
    model.eval()
    results = []
    t0 = time.time()
    
    for idx, (tid, label) in enumerate(eval_list):
        session_df = df.filter(pl.col(SESSION_ID_COL) == tid)
        res = evaluate_session_threshold(tid, label, session_df, model, tokenizer, threshold, DEVICE)
        if res:
            results.append(res)
        
        if (idx + 1) % 500 == 0:
            logger.info(f"  Evaluated {idx + 1}/{len(eval_list)} sessions...")
    
    eval_time = time.time() - t0
    
    # Metrics
    tp = sum(1 for r in results if r['label'] == 1 and r['is_detected'])
    fn = sum(1 for r in results if r['label'] == 1 and not r['is_detected'])
    fp = sum(1 for r in results if r['label'] == 0 and r['is_detected'])
    tn = sum(1 for r in results if r['label'] == 0 and not r['is_detected'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    positive_leads = [r for r in results if r['label'] == 1 and r['is_detected'] and r['lead_time'] > 0]
    
    metrics = {
        'f1': round(f1, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'accuracy': round(accuracy, 4),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'confusion_matrix': [[tn, fp], [fn, tp]],
        'detection_method': 'threshold',
        'threshold': round(threshold, 4),
        'eval_time_seconds': round(eval_time, 1),
        'sessions_evaluated': len(results),
        'lead_time': {
            'anticipated_count': len(positive_leads),
            'avg_minutes': round(np.mean([r['lead_time'] for r in positive_leads]), 2) if positive_leads else 0.0,
            'max_minutes': round(np.max([r['lead_time'] for r in positive_leads]), 2) if positive_leads else 0.0,
            'median_minutes': round(np.median([r['lead_time'] for r in positive_leads]), 2) if positive_leads else 0.0,
        },
    }
    
    logger.info(f"\n📊 Results (Threshold={threshold:.4f}):")
    logger.info(f"   F1:        {f1:.4f}")
    logger.info(f"   Precision: {precision:.4f}")
    logger.info(f"   Recall:    {recall:.4f}")
    logger.info(f"   TP={tp} | FP={fp} | FN={fn} | TN={tn}")
    logger.info(f"   Eval time: {eval_time:.0f}s")
    
    return metrics, results
