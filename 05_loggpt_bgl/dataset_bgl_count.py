import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import gc

# CONFIG
WINDOW_SIZE = 20 # SOTA for BGL
STEP_SIZE = 20   # Non-overlapping for training (maximizing data efficiency)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "BGL_processed.csv"
OUTPUT_FILE = DATA_DIR / "BGL_count_train.csv"

# CONSTANTS
SESSION_ID_COL = "node_id"
TIMESTAMP_COL = "timestamp"
LABEL_COL = "label"

def generate_count_dataset():
    print(f"🚀 Loading BGL data from {INPUT_FILE}...")
    
    # Check execution
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    # Load Data
    df = pl.read_csv(str(INPUT_FILE), infer_schema_length=10000)
    
    # Sort just in case
    df = df.sort([SESSION_ID_COL, TIMESTAMP_COL])
    
    print(f"📊 Total Rows: {len(df)}")
    print(f"🔹 Unique Nodes: {len(df[SESSION_ID_COL].unique())}")
    
    # Group by Node
    session_groups = df.group_by(SESSION_ID_COL, maintain_order=True)
    
    new_rows = []
    
    print("⚡ Processing Count-Based Windows (N=20)...")
    
    total_anom_windows = 0
    total_norm_windows = 0
    
    for tid, session_df in tqdm(session_groups, total=len(df[SESSION_ID_COL].unique())):
        templates = session_df["EventTemplate"].to_list()
        timestamps = session_df[TIMESTAMP_COL].to_list()
        labels = session_df[LABEL_COL].to_list()
        
        n_logs = len(templates)
        
        # Fixed Window Logic
        for i in range(0, n_logs, STEP_SIZE):
            end_idx = i + WINDOW_SIZE
            
            # Slice
            w_templates = templates[i:end_idx]
            w_timestamps = timestamps[i:end_idx]
            w_labels = labels[i:end_idx]
            
            # Ignore very short windows at the end?
            # Let's keep them if len >= 5 to avoid waste, but padding will handle it later.
            if len(w_templates) < 5:
                continue
                
            # Check Anomaly
            is_anom = sum(w_labels) > 0
            
            # FILTER: Unsupervised Training -> EXCLUDE Anomalies
            if is_anom:
                total_anom_windows += 1
                continue # SKIP ANOMALY FOR TRAINING
                
            total_norm_windows += 1
            
            # Add to dataset
            # We flatten here to save as CSV, compatible with LogGPT loader
            # Store Window ID to group later if needed, but Loader usually takes linear
            # Actually LogGPT loader groups by SessionId.
            # So we create a "Artificial Session ID" = NodeID_WindowIndex
            
            win_id = f"{tid}_w{i}"
            
            for k in range(len(w_templates)):
                new_rows.append({
                    SESSION_ID_COL: win_id,
                    'EventTemplate': w_templates[k],
                    TIMESTAMP_COL: w_timestamps[k],
                    LABEL_COL: w_labels[k]
                })

    print(f"\n📊 Generated Windows:")
    print(f"   - Normal (Used for Train): {total_norm_windows}")
    print(f"   - Anomalous (Skipped): {total_anom_windows}")
    print(f"   - Total Logs Output: {len(new_rows)}")
    
    # Save
    print(f"💾 Saving to {OUTPUT_FILE}...")
    final_df = pl.DataFrame(new_rows)
    final_df.write_csv(str(OUTPUT_FILE))
    print("✅ Done.")

if __name__ == "__main__":
    generate_count_dataset()
