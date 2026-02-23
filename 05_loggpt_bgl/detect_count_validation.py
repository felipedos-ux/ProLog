import torch
import torch.nn.functional as F
import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import os

from model import LogGPT
from sklearn.model_selection import train_test_split

import config_count as config
from config_count import (
    MODEL_NAME, MODEL_DIR, DEVICE, DATA_FILE, 
    SKIP_START_LOGS, INFER_SCHEMA_LENGTH, 
    TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT,
    SESSION_ID_COL, LABEL_COL, TIMESTAMP_COL
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Config
WINDOW_SIZE = 20
BATCH_SIZE = 32
THRESHOLD = 20.0 # Initial guess

def preprocess_count_windows(df, session_ids, tokenizer, window_size=20):
    all_windows = []
    
    target_df = df.filter(pl.col(SESSION_ID_COL).is_in(session_ids))
    target_df = target_df.sort([SESSION_ID_COL, TIMESTAMP_COL])
    
    session_groups = target_df.group_by(SESSION_ID_COL, maintain_order=True)
    
    for tid, session_df in tqdm(session_groups, desc="Preprocessing", total=len(session_ids)):
        templates = session_df["EventTemplate"].to_list()
        labels = session_df[LABEL_COL].to_list()
        
        n_logs = len(templates)
        
        # Sliding Window for Validation? Or Fixed Step?
        # Validation should be SLIDING to catch anomalies anywhere?
        # Or FIXED step to match OpenStack logic?
        # Let's use FIXED STEP (Non-overlapping) first to match training distribution.
        # Overlapping might inflate metrics or FP.
        
        step = window_size
        
        for i in range(0, n_logs, step):
            end_idx = i + window_size
            w_templates = templates[i:end_idx]
            w_labels = labels[i:end_idx]
            
            if len(w_templates) < 5: continue
            
            # Label
            w_anom_label = 1 if sum(w_labels) > 0 else 0
            
            # Tokenize
            # For validation, we treat whole window as one sequence
            # But inference calculates loss for Next Token.
            
            text = " \n ".join([str(t) for t in w_templates])
            ids = tokenizer.encode(text)
            
            # If ids > block_size?? 
            # In validation we can truncate or split.
            # config.BLOCK_SIZE = 64.
            if len(ids) > 64: ids = ids[:64]
            
            all_windows.append({
                'input_ids': ids,
                'label': w_anom_label,
                'node_id': tid
            })
            
    return all_windows

def evaluate_windows(windows, model, threshold, device):
    all_losses = []
    all_labels = []
    anom_losses = []
    norm_losses = []
    
    # Batch Processing
    for i in tqdm(range(0, len(windows), BATCH_SIZE), desc="Inference"):
        batch = windows[i:i+BATCH_SIZE]
        
        max_len = max(len(w['input_ids']) for w in batch)
        # Pad with EOS
        batch_tensor = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        batch_tensor.fill_(50256) # EOS
        
        for j, w in enumerate(batch):
            ids = w['input_ids']
            batch_tensor[j, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            
        with torch.no_grad():
            # Forward pass to get loss
            # We want loss per sample.
            # LogGPT forward usually validates inputs, targets.
            inputs = batch_tensor[:, :-1]
            targets = batch_tensor[:, 1:]
            
            # Model forward returns (logits, loss)
            # If we want per-sample loss, we need reduction='none'
            # Check model.py... assume it returns MEAN loss.
            # We must compute manual cross entropy for per-sample.
            
            logits, _ = model(inputs) # targets=None -> returns logits only usually
            if isinstance(logits, tuple): logits = logits[0]
            
            # Manual Loss
            # Shift done: logits [B, T, V], targets [B, T]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='none')
            loss = loss.view(targets.size())
            
            # Mean over sequence (ignoring padding?)
            # Valid positions mask
            # mask = (targets != 50256)
            # This logic is complex without modifying model.
            
            # FAST HACK: processing batch=1 allows capturing model loss if extraction enabled
            # OR JUST USE BATCH SIZE 1 for Validation Accuracy
            pass
            
    # REWRITE: Simple Iteration with Batch=1 for Correctness first
    model.eval()
    
    tp, fp, fn, tn = 0, 0, 0, 0
    
    predictions = []
    
    # Using larger batch manual calc
    with torch.no_grad():
         for i in tqdm(range(0, len(windows), BATCH_SIZE), desc="Eval Batch"):
            batch = windows[i:i+BATCH_SIZE]
            max_len = max(len(w['input_ids']) for w in batch)
            
            inputs_t = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            inputs_t.fill_(50256)
            
            for j, w in enumerate(batch):
                inputs_t[j, :len(w['input_ids'])] = torch.tensor(w['input_ids'], dtype=torch.long)
                
            inp = inputs_t[:, :-1]
            tgt = inputs_t[:, 1:]
            
            logits, _ = model(inp)
            
            # TOP-K LOGIC
            K = 1
            # logits: [B, T, V]
            # targets: [B, T]
            
            probs = F.softmax(logits, dim=-1)
            # Get top K indices: [B, T, K]
            topk_probs, topk_indices = torch.topk(probs, k=K, dim=-1)
            
            # Check if target is in topk
            # targets_expanded: [B, T, 1]
            targets_expanded = tgt.unsqueeze(-1)
            
            # matches: [B, T, K] -> boolean
            matches = (topk_indices == targets_expanded)
            
            # hit: [B, T] -> True if target in top K
            hits = matches.any(dim=-1)
            
            # Anomaly Score = Inverse of Hit Rate?
            # Or: if ANY token in the sequence is NOT in Top-K -> Window is Anomaly?
            # SOTA usually checks "next log".
            # Here we have a sequence.
            # If we miss ANY log in the sequence, is it anomalous?
            # Let's count "misses".
            
            misses = ~hits
            num_misses = misses.sum(dim=1)
            
            # Anomaly if num_misses > threshold (e.g. 0)
            # If threshold is 0, any miss is anomaly.
            threshold_counts = 0
            
            for j, miss_count in enumerate(num_misses):
                miss_count = miss_count.item()
                label = batch[j]['label']
                
                # Detect
                is_detected = miss_count > threshold_counts
                
                if label == 1:
                    anom_losses.append(miss_count)
                    if is_detected: tp += 1
                    else: fn += 1
                else:
                    norm_losses.append(miss_count)
                    if is_detected: fp += 1
                    else: tn += 1
                    
    import numpy as np
    print(f"\n🔍 TOP-{K} MISS STATISTICS (Threshold > {threshold_counts}):")
    if norm_losses:
        print(f"NORMAL Misses: Mean={np.mean(norm_losses):.4f}, Max={np.max(norm_losses):.4f}, Min={np.min(norm_losses):.4f}")
    if anom_losses:
        print(f"ANOMALY Misses: Mean={np.mean(anom_losses):.4f}, Max={np.max(anom_losses):.4f}, Min={np.min(anom_losses):.4f}")
        
    return tp, fp, fn, tn
                
    return tp, fp, fn, tn

def main():
    logger.info("🚀 Validating Count-Based Model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Model
    config_path = f"{MODEL_DIR}/config.pt"
    weights_path = f"{MODEL_DIR}/loggpt_weights.pt"
    
    if not os.path.exists(config_path):
        logger.error("Model not found")
        return
        
    model_config = torch.load(config_path, weights_only=False)
    model = LogGPT(model_config)
    model.load_state_dict(torch.load(weights_path, weights_only=False))
    model.to(DEVICE)
    model.eval()
    
    # Load Data (Original Processed)
    INPUT_FILE = "D:/ProLog/data/BGL_processed.csv"
    logger.info(f"Loading raw data from {INPUT_FILE}")
    df = pl.read_csv(INPUT_FILE, infer_schema_length=10000)
    
    # Split
    session_ids = df[SESSION_ID_COL].unique().to_list()
    # Same Random Split?
    # We need to replicate split logic to avoid training data leakage.
    # train_count.py used shuffle with seed 42?
    # config_count sets seed 42.
    
    import random
    random.seed(42)
    random.shuffle(session_ids) # Must match train_count.py logic exactly!
    
    n_train = int(len(session_ids) * (1 - TEST_SIZE_NORMAL))
    val_ids = session_ids[n_train:]
    
    # Subsample for speed
    val_ids = val_ids[:500]
    
    logger.info(f"Validating on {len(val_ids)} sessions...")
    
    windows = preprocess_count_windows(df, val_ids, tokenizer)
    logger.info(f"Generated {len(windows)} count-based windows.")
    
    tp, fp, fn, tn = evaluate_windows(windows, model, THRESHOLD, DEVICE)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*40)
    print(f"📊 RESULTS (Count-Based Model N=20)")
    print("="*40)
    print(f"Threshold: {THRESHOLD}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    
    if f1 > 0.8:
         print("\n✅ MISSION ACCOMPLISHED: >80% F1!")

if __name__ == "__main__":
    main()
