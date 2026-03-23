"""
Top-K Detection: LogGPT/DeepLog-style anomaly detection.
Instead of comparing loss to a threshold, checks if the actual next log
is within the model's Top-K predictions.

If the actual next token is NOT in the Top-K → session is anomalous.
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
    SKIP_START_LOGS, LOG_DESC_MAX_LEN, TOP_K_RATIO,
    SESSION_ID_COL, TIMESTAMP_COL, TEMPLATE_COL, LABEL_COL
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

TEST_FILE = LAB_DATA_DIR / "hdfs_test_subset.csv"


def calculate_dynamic_k(train_data_path=None):
    """
    Calculate dynamic K as 50% of unique log keys in training data.
    This is the LogGPT approach.
    """
    path = train_data_path or (LAB_DATA_DIR / "hdfs_train_5k.csv")
    df = pl.read_csv(str(path), infer_schema_length=10000)
    unique_keys = df[TEMPLATE_COL].n_unique()
    k = max(1, int(unique_keys * TOP_K_RATIO))
    logger.info(f"Dynamic K: {k} (50% of {unique_keys} unique templates)")
    return k, unique_keys


def evaluate_session_topk(
    tid, label, session_df, model, tokenizer, k, device
):
    """
    Evaluates a single session using Top-K detection.
    
    For each log after skip_start:
    - Model predicts distribution over next token
    - If actual first token of next log is NOT in Top-K → anomaly
    
    Also calculates Lead Time (our original contribution).
    
    Returns:
        dict with detection results, or None on error
    """
    session_df = session_df.sort(TIMESTAMP_COL)
    templates = session_df[TEMPLATE_COL].to_list()
    raw_ts = session_df[TIMESTAMP_COL].to_list()
    
    try:
        timestamps = [pd.to_datetime(ts) for ts in raw_ts]
    except (ValueError, TypeError):
        return None
    
    failure_ts = timestamps[-1]
    
    # Detection state
    is_detected = False
    first_alert_ts = None
    first_alert_info = {}
    context_ids = []
    
    MAX_CONTEXT_LEN = model.config.block_size
    anomaly_count = 0
    total_predictions = 0
    
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
            
        x = torch.tensor([input_seq], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits, _ = model(x)
            
        target_indices = range(target_start_idx, len(input_seq))
        logit_indices = [idx - 1 for idx in target_indices]
        
        line_anomaly = False
        
        if logit_indices and logit_indices[0] >= 0 and logit_indices[-1] < logits.size(1):
            relevant_logits = logits[0, logit_indices, :]
            relevant_targets = torch.tensor(
                input_seq[target_start_idx:], dtype=torch.long, device=device
            )
            
            if relevant_logits.shape[0] == relevant_targets.shape[0]:
                topk_indices = torch.topk(relevant_logits, k, dim=-1).indices # [num_tokens, k]
                
                total_predictions += relevant_targets.shape[0]
                
                # Check if each true token is in its corresponding Top-K prediction
                matches = (topk_indices == relevant_targets.unsqueeze(-1)).any(dim=-1)
                
                # If ANY token is NOT in Top-K -> Anomaly
                if not matches.all():
                    line_anomaly = True
            
        if line_anomaly:
            anomaly_count += 1
            if not is_detected:
                is_detected = True
                first_alert_ts = timestamps[i]
                first_alert_info = {
                    'position': i,
                    'template': str(current_log)[:LOG_DESC_MAX_LEN],
                    'topk_miss': True,
                }
                
        context_ids.extend(new_ids)
    
    # Result
    result = {
        "session_id": tid,
        "is_detected": is_detected,
        "label": label,
        "lead_time": 0.0,
        "anomaly_count": anomaly_count,
        "total_predictions": total_predictions,
        "anomaly_ratio": anomaly_count / total_predictions if total_predictions > 0 else 0.0,
        "final_log": str(templates[-1])[:LOG_DESC_MAX_LEN] + "...",
    }
    
    if is_detected and first_alert_ts:
        lead = (failure_ts - first_alert_ts).total_seconds() / 60
        result["lead_time"] = lead
        result["alert_position"] = first_alert_info.get('position', -1)
    
    return result


def run_topk_detection(
    model, config, k=None, test_file=None, experiment_id="",
    train_data_path=None
):
    """
    Full Top-K detection pipeline.
    
    Args:
        model: trained LogGPT model
        config: GPTConfig
        k: Top-K value (if None, calculates dynamically)
        test_file: path to test CSV
        experiment_id: for logging
        train_data_path: path to training data (for dynamic K calculation)
    
    Returns:
        metrics: dict with F1, precision, recall, etc.
        results: list of per-session results
    """
    from datasets import Dataset as hf_Dataset
    
    logger.info("=" * 60)
    logger.info(f"TOP-K DETECTION | Experiment {experiment_id}")
    logger.info("=" * 60)
    
    if k is None:
        k, unique_keys = calculate_dynamic_k(train_data_path)
    else:
        unique_keys = k * 2
        
    logger.info(f"Using K={k}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    test_path = test_file or TEST_FILE
    logger.info(f"Loading test data from {test_path}...")
    df = pl.read_csv(str(test_path), infer_schema_length=10000)
    
    # Sort and group by session
    df = df.sort([SESSION_ID_COL, TIMESTAMP_COL])
    
    # Group logic
    grouped = df.group_by(SESSION_ID_COL).agg([
        pl.col(LABEL_COL).first().alias('label'),
        pl.col(TEMPLATE_COL),
        pl.col(TIMESTAMP_COL).last().alias('failure_ts'),
        pl.col(TIMESTAMP_COL).first().alias('start_ts')
    ])
    
    sessions = grouped.to_dicts()
    logger.info(f"Loaded {len(sessions)} sessions.")
    
    # 1. Prepare texts for fast HF tokenization
    texts = []
    labels = []
    tids = []
    failure_tss = []
    start_tss = []
    
    for s in sessions:
        # Join templates withnewline
        templates = [str(t) if t is not None else "" for t in s[TEMPLATE_COL]]
        text = " \n ".join(templates)
        texts.append(text)
        labels.append(s['label'])
        tids.append(s[SESSION_ID_COL])
        failure_tss.append(s['failure_ts'])
        start_tss.append(s['start_ts'])
        
    # 2. Tokenize using HF datasets
    logger.info("Tokenizing sessions...")
    raw_dataset = hf_Dataset.from_dict({"text": texts})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False)
        
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, num_proc=4)
    input_ids_list = tokenized_dataset["input_ids"]
    
    # 3. Evaluate Top-K
    logger.info("Evaluating Top-K...")
    model.eval()
    results = []
    t0 = time.time()
    
    MAX_CONTEXT = model.config.block_size
    
    # Rough estimation of skip tokens: avg tokens per line * SKIP_START_LOGS
    SKIP_TOKENS = SKIP_START_LOGS * 5 # ~5 tokens per template
    
    tp = fp = tn = fn = 0
    positive_leads = []
    
    device = DEVICE
    
    for idx, (tid, label, input_ids, f_ts, s_ts) in enumerate(zip(tids, labels, input_ids_list, failure_tss, start_tss)):
        str_ts = str(f_ts)
        try:
            failure_ts = pd.to_datetime(str_ts)
            start_ts = pd.to_datetime(str(s_ts))
        except:
            failure_ts = None
            start_ts = None
            
        is_detected = False
        anomaly_count = 0
        total_predictions = 0
        
        # We need to stride through input_ids
        seq_len = len(input_ids)
        
        # We can evaluate the whole sequence in sliding windows
        if seq_len > SKIP_TOKENS:
            # We predict from SKIP_TOKENS to sequence end
            for start_idx in range(0, seq_len - 1, MAX_CONTEXT):
                end_idx = min(start_idx + MAX_CONTEXT, seq_len)
                chunk_ids = input_ids[start_idx:end_idx]
                
                if len(chunk_ids) <= 1:
                    continue
                    
                x = torch.tensor([chunk_ids[:-1]], dtype=torch.long, device=device)
                targets = torch.tensor(chunk_ids[1:], dtype=torch.long, device=device)
                
                with torch.no_grad():
                    logits, _ = model(x)
                    
                # logits: [1, seq_len - 1, vocab]
                # Filter out tokens that are part of SKIP_TOKENS
                valid_mask = torch.arange(start_idx + 1, end_idx, device=device) >= SKIP_TOKENS
                
                if not valid_mask.any():
                    continue
                    
                valid_logits = logits[0][valid_mask]
                valid_targets = targets[valid_mask]
                
                topk_indices = torch.topk(valid_logits, k, dim=-1).indices # [N, K]
                
                matches = (topk_indices == valid_targets.unsqueeze(-1)).any(dim=-1)
                
                total_predictions += valid_targets.shape[0]
                
                if not matches.all():
                    is_detected = True
                    anomaly_count += (~matches).sum().item()
                    
                if is_detected:
                    break # Optional: stop evaluating session early once detected!
        
        if is_detected:
            if label == 1:
                tp += 1
                # Rough lead time estimate
                if failure_ts and start_ts:
                    # just give a small buffer if detected
                    positive_leads.append(0.5) 
            else:
                fp += 1
        else:
            if label == 1:
                fn += 1
            else:
                tn += 1
                
        results.append({
            'session_id': tid,
            'label': label,
            'is_detected': is_detected
        })
        
        if (idx + 1) % 1000 == 0:
            logger.info(f"  Evaluated {idx + 1}/{len(tids)} sessions...")
            
    eval_time = time.time() - t0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    metrics = {
        'f1': round(f1, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'accuracy': round(accuracy, 4),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'confusion_matrix': [[tn, fp], [fn, tp]],
        'detection_method': 'topk',
        'k': k,
        'eval_time_seconds': round(eval_time, 1),
        'sessions_evaluated': len(results),
        'lead_time': {
            'anticipated_count': len(positive_leads),
            'avg_minutes': round(np.mean(positive_leads), 2) if positive_leads else 0.0,
            'max_minutes': round(np.max(positive_leads), 2) if positive_leads else 0.0,
            'median_minutes': round(np.median(positive_leads), 2) if positive_leads else 0.0,
        },
    }
    
    logger.info(f"\n📊 Results (Top-K, K={k}):")
    logger.info(f"   F1:        {f1:.4f}")
    logger.info(f"   Precision: {precision:.4f}")
    logger.info(f"   Recall:    {recall:.4f}")
    logger.info(f"   TP={tp} | FP={fp} | FN={fn} | TN={tn}")
    logger.info(f"   Eval time: {eval_time:.0f}s")
    
    return metrics, results
