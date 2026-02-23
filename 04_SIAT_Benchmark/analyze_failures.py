# -*- coding: utf-8 -*-
"""
SIAT Failure Analysis & Leadtime
================================
1. Loads Test Data (Events + Timestamps)
2. Loads Model Predictions (Losses)
3. Identifies Ground Truth Anomaly Time (First 4xx/5xx)
4. Identifies Detection Time (First Loss > Threshold)
5. Calculates Leadtime (Error Time - Detection Time)
   - Positive: Early Warning
   - Negative: Reactive
6. Categorizes Failure Types (Endpoints involved)
"""
import pickle
import numpy as np
import pandas as pd
import torch
import json
from tqdm import tqdm
from pathlib import Path
from collections import Counter

from config import OUTPUT_DIR, SESSION_DATA, DEVICE
from dataset import load_data, SimpleTokenizer
from model import LogGPT, GPTConfig

THRESHOLD_CONFIG = OUTPUT_DIR / "threshold_optimized.json" # Use optimized!
if not THRESHOLD_CONFIG.exists():
    THRESHOLD_CONFIG = OUTPUT_DIR / "threshold_config.json"

def main():
    print("🕵️ Analyzing Failures & Leadtime...")
    
    # 1. Load Data (Need timestamps!)
    # Dataset loader gives input_ids. We need raw data for timestamps.
    print(f"📦 Loading {SESSION_DATA}...")
    with open(SESSION_DATA, "rb") as f:
        data = pickle.load(f)
    
    test_data = data['test'] # DataFrame with 'events', 'timestamp' (list of iso strings), 'label'
    
    # 2. Load Threshold
    with open(THRESHOLD_CONFIG, 'r') as f:
        t_config = json.load(f)
        threshold = t_config["threshold"]
    print(f"Using Threshold: {threshold:.4f}")

    # 3. Load Model
    # We need to run inference again? 
    # Or did we save per-token losses?
    # detect_siat.py saved `results_siat.pkl` but probably just max loss per session?
    # Let's check detect_siat.py... 
    # It calculated `sample_losses` (mean over session) and stored `loss`. 
    # It did NOT store per-token loss trend.
    # To calculate Leadtime, we need *when* inside the session the loss spiked.
    # So we must re-run inference on Anomalous Sessions (TP).
    
    print("🔄 Re-running inference on Anomalous Sessions for granule analysis...")
    
    # Load Model
    from config import CHECKPOINT_PATH, BATCH_SIZE
    tokenizer = SimpleTokenizer()
    tokenizer.fit(data['train']['events'].tolist()) # Re-fit correctly
    
    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=128)
    model = LogGPT(config).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    
    # Filter only True Positives (initially just Label=1)
    anom_df = test_data[test_data['label'] == 1].reset_index(drop=True)
    print(f"Analyzing {len(anom_df)} anomalous sessions...")
    
    leads = []
    failure_types = []
    
    # Inference Loop
    for idx, row in tqdm(anom_df.iterrows(), total=len(anom_df)):
        events = row['events']
        timestamps = pd.to_datetime(row['timestamp'])
        
        # Identify Ground Truth Error Index
        # We need status codes? test_data only has 'events' and 'label'.
        # 'events' are like "GET /api/foo".
        # Wait, preprocess_siat.py created 'is_error' column in df_clean but aggregation kept 'is_error' as max label.
        # It did NOT keep status codes per event in the aggregated list.
        # BUT... 
        # In preprocess_siat.py:
        # sessions = df_clean.groupby('global_session_id').agg({ 'event_token': list, 'timestamp': list, 'is_error': 'max' })
        # We lost the per-event status code!
        # However, we can infer it?
        # NO. "GET /api/foo" doesn't say if it was 200 or 500.
        # Wait, preprocess code: 
        # df_clean['event_token'] = df_clean['method'] + " " + df_clean['endpoint_norm']
        # It does NOT include status code in the token.
        # CRITICAL: We cannot know *which* event failed from the pickled data if we didn't save it.
        
        # Strategy Adjustment:
        # We assume the *last* event is the failure? No.
        # Or we rely on the implementation detail that usually anomalies are 4xx/5xx in the log.
        # But we stripped status from token.
        
        # Backtrack: We need to modify preprocess? Too slow.
        # Workaround: Re-load raw CSV? 
        # Yes, we have 'global_session_id' in siat_sessions.pkl? 
        # Let's check pickle structure.
        # 'test' is a DataFrame. Does it have 'global_session_id'? 
        # preprocess_siat.py: sessions.reset_index() -> global_session_id is a column!
        # YES. We can map back to raw CSV if needed.
        # BETTER: Preprocess stripped status codes from 'events'.
        # BUT... the user question implies we *can* know.
        # Let's check if we can infer "failure" from the token itself? No.
        
        # Okay, I will load the raw CSV, identify the error timestamps for these session IDs, and then compare.
        pass

    # Re-loading Raw Data for mapping
    # This is heavy but necessary for "Categorize Failures" and "Leadtime".
    print("📂 Loading Raw CSV to map errors...")
    raw_df = pd.read_csv("D:/ProLog/data/siat.csv", encoding='latin-1', header=None, low_memory=False)
    raw_df.columns = ['timestamp', 'status_code', 'method', 'endpoint', 'service', 'server', 'ip', 'city', 'country', 'user_agent']
    
    # Helper to calculate session ID (same logic as preprocess)
    # We can just match by unique IP + Time? Risks collisions.
    # Re-generating session IDs on the fly is risky.
    
    # Wait, the pickle `test` DF has `global_session_id`.
    # AND `preprocess_siat.py` calculated `df['global_session_id']`.
    # IF I strictly followed preprocess logic, I can regenerate or...
    # Did preprocess save the intermediate DF? No.
    
    # FAST PATH:
    # 1. We have `global_session_id` in `test_data`.
    # 2. We can't easily join with raw CSV without repeating the session logic.
    # 3. Let's repeat session logic briefly on raw data.
    
    # ... (Re-implement session logic from preprocess_siat.py) ...
    # This is safer.
    
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], errors='coerce')
    raw_df = raw_df.dropna(subset=['timestamp'])
    raw_df = raw_df.sort_values(['ip', 'timestamp'])
    raw_df['time_diff'] = raw_df.groupby('ip')['timestamp'].diff().dt.total_seconds()
    raw_df['new_session'] = (raw_df['time_diff'].isna()) | (raw_df['time_diff'] > 300)
    raw_df['session_id'] = raw_df.groupby('ip')['new_session'].cumsum()
    raw_df['global_session_id'] = raw_df['ip'].astype(str) + '_' + raw_df['session_id'].astype(str)
    
    # Now filter raw_df to only include session_ids in test_data[anom]
    target_sessions = set(anom_df['global_session_id'].unique())
    raw_anom = raw_df[raw_df['global_session_id'].isin(target_sessions)].copy()
    
    # Identify Error Timestamps per session
    raw_anom['status_code'] = pd.to_numeric(raw_anom['status_code'], errors='coerce').fillna(200).astype(int)
    raw_anom['is_error'] = raw_anom['status_code'] >= 400
    
    # Map session_id -> first_error_time, error_endpoint
    error_map = {}
    for sid, group in raw_anom.groupby('global_session_id'):
        errors = group[group['is_error']]
        if not errors.empty:
            first_err = errors.iloc[0]
            error_map[sid] = {
                'error_time': first_err['timestamp'],
                'error_type': f"{first_err['method']} {first_err['endpoint']} ({first_err['status_code']})",
                'error_idx': group.index.get_loc(first_err.name) # Relative index in session
            }
            
    # Now Inference
    results_list = []
    
    for idx, row in tqdm(anom_df.iterrows(), total=len(anom_df)):
        sid = row['global_session_id']
        if sid not in error_map:
            continue # Should not happen if label=1
            
        err_info = error_map[sid]
        
        # Tokenize
        tokens = tokenizer.encode(row['events'])
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
        
        # Loop inference (batch size 1) to avoid dimension hurdles
        with torch.no_grad():
            logits, _ = model(x)
            
            # Loss Calculation - Fully explicit to avoid shape mismatch
            # Logits: [1, seq_len-1, vocab]
            # Targets: [1, seq_len-1]
            
            # Logits: [1, seq_len, vocab] -> Shift to [1, seq_len-1, vocab]
            shift_logits = logits[:, :-1, :].contiguous()
            # Targets: [1, seq_len] -> Shift to [1, seq_len-1]
            shift_targets = x[:, 1:].contiguous()
            
            # Flatten to [seq_len-1, vocab] and [seq_len-1]
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_targets = shift_targets.view(-1)
            
            # Use cross_entropy with reduction='none'
            # Input: (N, C), Target: (N) -> Output: (N)
            loss = torch.nn.functional.cross_entropy(
                flat_logits, 
                flat_targets, 
                reduction='none'
            )
            
            # loss is now a 1D tensor [seq_len-1]
            losses = loss.cpu().numpy()
            
            # Now align losses with timestamps.
            # losses[t] corresponds to prediction of token at index t+1.
            # timestamps should be accessed at t+1.

            
            # Find first breach
            # We align loss index t with timestamp t+1
            breach_idx = -1
            breach_time = None
            
            for t, l_val in enumerate(losses):
                if l_val > threshold:
                    breach_idx = t
                    # Corresponds to timestamp[t+1]
                    if t+1 < len(row['timestamp']):
                        breach_time = pd.to_datetime(row['timestamp'][t+1])
                    break
            
            if breach_time:
                # Leadtime = Error Time - Alert Time
                # Positive = Alert BEFORE Error
                lead_sec = (err_info['error_time'] - breach_time).total_seconds()
                
                results_list.append({
                    'session_id': sid,
                    'leadtime': lead_sec,
                    'failure_type': err_info['error_type'],
                    'error_time': err_info['error_time'],
                    'alert_time': breach_time
                })
                
    # Analysis
    df_res = pd.DataFrame(results_list)
    if df_res.empty:
        print("No anomalies analyzed?")
        return

    print("\n📊 FAILURE CATEGORIES:")
    print(df_res['failure_type'].value_counts().head(10))
    
    early_detections = df_res[df_res['leadtime'] > 0]
    n_early = len(early_detections)
    pct_early = n_early / len(df_res) * 100
    
    print(f"\n⏰ LEADTIME ANALYSIS:")
    print(f"  Total Anomalies Analyzed: {len(df_res)}")
    print(f"  Early Detections (>0s):   {n_early} ({pct_early:.1f}%)")
    print(f"  Avg Leadtime (Early):     {early_detections['leadtime'].mean():.2f}s")
    print(f"  Max Leadtime:             {early_detections['leadtime'].max():.2f}s")
    
    # Save detailed report
    # Print detailed report to stdout for capture
    print("\n" + "="*40)
    print("FAILURE ANALYSIS REPORT")
    print("="*40)
    print(f"Threshold: {threshold}\n")
    print(f"Early Detections: {n_early}/{len(df_res)} ({pct_early:.1f}%)")
    print(f"Avg Leadtime: {early_detections['leadtime'].mean():.2f}s\n")
    print("TOP 20 FAILURE TYPES:")
    print(str(df_res['failure_type'].value_counts().head(20)))
    print("="*40 + "\n")
    
    # Save detailed report
    out_path = OUTPUT_DIR / "failure_analysis.txt"
    try:
        with open(out_path, "w", encoding='utf-8') as f:
            f.write("FAILURE ANALYSIS REPORT\n=======================\n")
            f.write(f"Threshold: {threshold}\n\n")
            f.write(f"Early Detections: {n_early}/{len(df_res)} ({pct_early:.1f}%)\n")
            f.write(f"Avg Leadtime: {early_detections['leadtime'].mean():.2f}s\n\n")
            f.write("TOP 20 FAILURE TYPES:\n")
            f.write(str(df_res['failure_type'].value_counts().head(20)))
        print(f"Saved analysis to {out_path.resolve()}")
    except Exception as e:
        print(f"Failed to write file: {e}")

if __name__ == "__main__":
    main()
