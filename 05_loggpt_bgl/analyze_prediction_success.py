import torch
import polars as pl
import json
import numpy as np
from transformers import AutoTokenizer
import config_count as config
from model import LogGPT
from deploy_hybrid_prod import categorize_error  # Reuse our new function
from utils.logger import setup_logger

logger = setup_logger(__name__)

# CONFIG
MODEL_PATH = config.MODEL_DIR 
DEVICE = config.DEVICE
K_PROD = 5
FREQ_FILE = config.MODEL_DIR / "template_freq.json"
RARE_THRESHOLD = 5

def analyze_successes():
    # Load Model & Resources
    config_path = MODEL_PATH / "config.pt"
    model_config = torch.load(config_path, weights_only=False)
    model = LogGPT(model_config)
    model.load_state_dict(torch.load(MODEL_PATH / "loggpt_weights.pt", map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    
    with open(FREQ_FILE, 'r') as f:
        freq_map = json.load(f)
        
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Validation Data
    RAW_DATA = config.DATA_DIR / "BGL_processed.csv"
    logger.info(f"Loading Raw Data from {RAW_DATA}...")
    df = pl.read_csv(str(RAW_DATA), infer_schema_length=10000)
    
    # Validation Split
    session_ids = df[config.SESSION_ID_COL].unique().to_list()
    import random
    random.seed(config.SEED)
    random.shuffle(session_ids)
    n_train = int(len(session_ids) * (1 - config.TEST_SIZE_NORMAL))
    val_ids = set(session_ids[n_train:])
    val_df = df.filter(pl.col(config.SESSION_ID_COL).is_in(val_ids))
    
    anom_sessions = val_df.group_by(config.SESSION_ID_COL).agg(pl.col("label").sum().alias("total_anoms"))
    anom_ids = anom_sessions.filter(pl.col("total_anoms") > 0)[config.SESSION_ID_COL].to_list()
    
    logger.info(f"Scanning {len(anom_ids)} Anomalous Sessions for Precursors...")
    
    success_stories = []
    
    for sid in anom_ids:
        sess_df = val_df.filter(pl.col(config.SESSION_ID_COL) == sid).sort("timestamp")
        
        # Ground Truth
        failure_rows = sess_df.filter(pl.col("label") == 1)
        if len(failure_rows) == 0: continue
        t_fail = failure_rows["timestamp"][0]
        actual_error = failure_rows["EventTemplate"][0]
        
        templates = sess_df["EventTemplate"].to_list()
        timestamps = sess_df["timestamp"].to_list()
        
        # Check Windows
        for k in range(0, len(templates), config.BLOCK_SIZE):
            chunk_t = templates[k:k+20]
            chunk_time = timestamps[k:k+20]
            if len(chunk_t) < 5: continue
            
            curr_time = chunk_time[-1]
            if curr_time >= t_fail: break # Reached failure without early warning
            
            # Hybrid Check
            alert_reason = None
            alert_template = None
            
            # 1. Rare
            for t in chunk_t:
                if str(t) not in freq_map or freq_map[str(t)]['count'] < RARE_THRESHOLD:
                    alert_reason = "Rare Template"
                    alert_template = t
                    break
            
            # 2. LogGPT
            if not alert_reason:
                text = " \n ".join([str(t) for t in chunk_t])
                input_ids = tokenizer.encode(text)
                if len(input_ids) > config.BLOCK_SIZE: input_ids = input_ids[:config.BLOCK_SIZE]
                inp = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits, _ = model(inp[:, :-1])
                    probs = F.softmax(logits, dim=-1)
                    _, topk = torch.topk(probs, k=K_PROD, dim=-1)
                    hits = (topk == inp[:, 1:].unsqueeze(-1)).any(dim=-1)
                    if (~hits).sum() > 0:
                        alert_reason = "Sequence Anomaly"
                        # Approximation
                        alert_template = chunk_t[-1] 
            
            if alert_reason:
                lead_time = t_fail - curr_time
                if lead_time > 0:
                    success_stories.append({
                        "session": sid,
                        "lead_time": float(lead_time),
                        "lead_time_hours": float(lead_time / 3600),
                        "actual_error_type": categorize_error(actual_error),
                        "actual_error_template": str(actual_error),
                        "alert_reason": alert_reason,
                        "precursor_type": categorize_error(alert_template),
                        "precursor_template": str(alert_template)
                    })
                    break # Found earliest warning
    
    # Save & Print
    if not success_stories:
        print("No successful predictions found in this subset.")
        return
        
    print(f"\n🚀 Found {len(success_stories)} Successful Predictions!")
    
    # Sorting
    success_stories.sort(key=lambda x: x['lead_time'], reverse=True)
    
    print("\n🔹 Top 5 Best Predictions:")
    for i, s in enumerate(success_stories[:5]):
        print(f"{i+1}. Lead Time: {s['lead_time_hours']:.1f} hours")
        print(f"   Error: {s['actual_error_type']} ({s['actual_error_template'][:50]}...)")
        print(f"   Warn:  {s['precursor_type']} ({s['precursor_template'][:50]}...)")
        print(f"   Reason: {s['alert_reason']}")
        print("---")
        
    # Stats on Error Types
    types = [s['actual_error_type'] for s in success_stories]
    from collections import Counter
    print("\n🔹 Predictable Error Categories:")
    print(Counter(types))

if __name__ == "__main__":
    analyze_successes()
