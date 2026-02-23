import torch
import torch.nn.functional as F
import polars as pl
import json
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import config_count as config
from model import LogGPT
from utils.logger import setup_logger

logger = setup_logger(__name__)

# CONFIG
MODEL_PATH = config.MODEL_DIR 
DEVICE = config.DEVICE
K_PROD = 5
FREQ_FILE = config.MODEL_DIR / "template_freq.json"
RARE_THRESHOLD = 5

# ERROR CATEGORIES (Heuristic)
def categorize_error(content):
    content = str(content).lower()
    if 'mmfs' in content or 'file' in content or 'disk' in content: return "I/O (MMFS)"
    if 'ciod' in content or 'torus' in content or 'tree' in content or 'network' in content: return "Network"
    if 'kernel' in content or 'instruction' in content or 'program' in content: return "System/Kernel"
    if 'parity' in content or 'cache' in content or 'ddr' in content or 'memory' in content: return "Memory/Hardware"
    if 'power' in content or 'temp' in content or 'fan' in content: return "Hardware/Env"
    if 'app' in content or 'job' in content: return "Application"
    return "Unknown/Other"

def load_resources():
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
    
    return model, tokenizer, freq_map

def run_leadtime_analysis():
    model, tokenizer, freq_map = load_resources()
    
    RAW_DATA = config.DATA_DIR / "BGL_processed.csv"
    logger.info(f"Loading Raw Data from {RAW_DATA}...")
    df = pl.read_csv(str(RAW_DATA), infer_schema_length=10000)
    
    # 1. IDENTIFY ANOMALOUS SESSIONS (Validation Set Logic)
    # We need strictly VALIDATION sessions to be fair.
    session_ids = df[config.SESSION_ID_COL].unique().to_list()
    import random
    random.seed(config.SEED)
    random.shuffle(session_ids)
    n_train = int(len(session_ids) * (1 - config.TEST_SIZE_NORMAL))
    val_ids = set(session_ids[n_train:])
    
    val_df = df.filter(pl.col(config.SESSION_ID_COL).is_in(val_ids))
    
    # Filter only sessions that HAVE anomalies
    # (We can't measure lead time on normal sessions)
    anom_sessions = val_df.group_by(config.SESSION_ID_COL).agg(pl.col("label").sum().alias("total_anoms"))
    anom_ids = anom_sessions.filter(pl.col("total_anoms") > 0)[config.SESSION_ID_COL].to_list()
    
    logger.info(f"Analyzing {len(anom_ids)} Anomalous Validation Sessions for Lead Time...")
    
    lead_times = []
    categories = {}
    
    # Process each anomalous session individually
    for sid in tqdm(anom_ids, desc="Sessions"):
        sess_df = val_df.filter(pl.col(config.SESSION_ID_COL) == sid).sort("timestamp")
        
        # Ground Truth Failure Time (First Log with Label=1)
        failure_row = sess_df.filter(pl.col("label") == 1).head(1)
        if len(failure_row) == 0: continue
        t_fail = failure_row["timestamp"][0]
        fail_content = failure_row["EventTemplate"][0]
        
        # Categorize
        cat = categorize_error(fail_content)
        categories[cat] = categories.get(cat, 0) + 1
        
        # RUN DETECTION (Simulate Streaming)
        # We need to find the FIRST window that flags an anomaly.
        templates = sess_df["EventTemplate"].to_list()
        timestamps = sess_df["timestamp"].to_list()
        
        # Hybrid Detection Logic
        first_alert_time = None
        
        for k in range(0, len(templates), config.BLOCK_SIZE): # Step 20? No, usually streaming is log-by-log or sliding 1.
            # But trained on fixed 20. Let's use Fixed 20 to be consistent with accuracy benchmarks.
            # Or Sliding step 1 for maximum lead time accuracy?
            # Sliding step 5 is a good compromise.
            
            chunk_t = templates[k:k+20]
            chunk_time = timestamps[k:k+20]
            
            if len(chunk_t) < 5: continue
            
            # Current time (at end of window)
            curr_time = chunk_time[-1]
            if curr_time > t_fail: 
                # We passed the failure moment. 
                # If we haven't detected yet, it's a "Late Detection" (Lead Time likely 0 or negative)
                if first_alert_time is None: first_alert_time = curr_time
                break
                
            # 1. Frequency Check
            rare = False
            for t in chunk_t:
                if str(t) not in freq_map or freq_map[str(t)]['count'] < RARE_THRESHOLD:
                    rare = True
                    break
            
            if rare:
                first_alert_time = curr_time
                break
                
            # 2. Model Check
            text = " \n ".join([str(t) for t in chunk_t])
            input_ids = tokenizer.encode(text)
            if len(input_ids) > config.BLOCK_SIZE: input_ids = input_ids[:config.BLOCK_SIZE]
            
            inp_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                inputs = inp_tensor[:, :-1]
                targets = inp_tensor[:, 1:]
                logits, _ = model(inputs)
                probs = F.softmax(logits, dim=-1)
                _, topk = torch.topk(probs, k=K_PROD, dim=-1)
                hits = (topk == targets.unsqueeze(-1)).any(dim=-1)
                if (~hits).sum() > 0:
                    first_alert_time = curr_time
                    break
        
        if first_alert_time is not None:
            lt = t_fail - first_alert_time
            # If lt > 0: Predicted X seconds before failure.
            # If lt < 0: Predicted X seconds AFTER failure start.
            lead_times.append(lt)
        else:
            # Missed? Or detected way later?
            pass
            
    # REPORT
    lead_times = np.array(lead_times)
    # We only care about VALID predictions (where we detected something)
    
    print("\n========================================")
    print("⏱️ LEAD TIME & CATEGORIZATION ANALYSIS")
    print("========================================")
    
    print(f"Total Anomalous Sessions: {len(anom_ids)}")
    print(f"Detected Sessions: {len(lead_times)}")
    
    if len(lead_times) > 0:
        avg_lt = np.mean(lead_times)
        max_lt = np.max(lead_times)
        min_lt = np.min(lead_times)
        pos_lt = lead_times[lead_times > 0]
        
        print(f"\nTime-to-Failure Prediction:")
        print(f"   Avg Lead Time (All): {avg_lt:.2f} seconds")
        print(f"   Max Early Warning:   {max_lt:.2f} seconds")
        print(f"   Successful Pre-Warnings (>0s): {len(pos_lt)} ({len(pos_lt)/len(lead_times)*100:.1f}%)")
        if len(pos_lt) > 0:
             print(f"   Avg Lead Time (Successful): {np.mean(pos_lt):.2f} seconds")
             
    print(f"\nError Categories Discovered:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {cat}: {count} sessions")

if __name__ == "__main__":
    run_leadtime_analysis()
