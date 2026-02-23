import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import polars as pl
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm

from model import LogGPT
from utils.logger import setup_logger
import config_count as config

# HYBRID DEFAULTS
MODEL_PATH = config.MODEL_DIR 
BLOCK_SIZE = config.BLOCK_SIZE
DEVICE = config.DEVICE
K_PRODUCTION = 5 # LogGPT Component
FREQ_FILE = config.MODEL_DIR / "template_freq.json"
RARE_THRESHOLD = 5 # Absolute count. If seen < 5 times in training, it's rare.
                   # Alternatively use freq < 0.00001

logger = setup_logger(__name__)

def load_resources():
    """Loads Model, Tokenizer, AND Frequency Map."""
    logger.info(f"Loading Resources...")
    
    # 1. Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    config_path = MODEL_PATH / "config.pt"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
        
    model_config = torch.load(config_path, weights_only=False)
    model = LogGPT(model_config)
    
    weights_path = MODEL_PATH / "loggpt_weights.pt"
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    
    # 2. Frequency Map
    if not FREQ_FILE.exists():
        raise FileNotFoundError(f"Frequency map not found at {FREQ_FILE}. Run train_frequency.py first.")
        
    with open(FREQ_FILE, 'r') as f:
        freq_map = json.load(f)
        
    return model, tokenizer, freq_map

def is_rare_template(template_id, freq_map):
    """Checks if a template is rare based on training stats."""
    t_str = str(template_id)
    if t_str not in freq_map:
        return True # Unseen is inherently rare/anomalous
        
    count = freq_map[t_str]['count']
    return count < RARE_THRESHOLD

def preprocess_logs(log_file, tokenizer, window_size=20):
    """Same as deploy_bgl_prod.py but returns templates for frequency check."""
    logger.info(f"Processing logs from {log_file}...")
    
    df = pl.read_csv(str(log_file), infer_schema_length=10000)
    
    grouped = df.group_by(config.SESSION_ID_COL, maintain_order=True).agg(pl.col("EventTemplate"))
    rows = grouped.to_dict(as_series=False)
    
    windows = []
    
    for i, templates in enumerate(tqdm(rows["EventTemplate"], desc="Windowing")):
        session_id = rows[config.SESSION_ID_COL][i]
        
        for k in range(0, len(templates), window_size):
            chunk = templates[k:k+window_size]
            if len(chunk) < 2: continue 
            
            text = " \n ".join([str(t) for t in chunk])
            input_ids = tokenizer.encode(text)
            
            if len(input_ids) > BLOCK_SIZE:
                input_ids = input_ids[:BLOCK_SIZE]
                
            windows.append({
                'session_id': session_id,
                'chunk_index': k // window_size,
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'templates': chunk # Keep raw templates for frequency check
            })
            
    return windows


# ERROR CATEGORIES (Enhanced)
def categorize_error(content):
    content = str(content).lower()
    if 'mmfs' in content or 'file' in content or 'disk' in content or 'dfs' in content: return "I/O (Filesystem)"
    if 'ciod' in content or 'torus' in content or 'tree' in content or 'link' in content or 'network' in content: return "Network"
    if 'kernel' in content or 'instruction' in content or 'program' in content or 'ras' in content: return "System/Kernel"
    if 'parity' in content or 'cache' in content or 'ddr' in content or 'memory' in content: return "Memory"
    if 'power' in content or 'temp' in content or 'fan' in content or 'env' in content: return "Hardware/Environment"
    if 'app' in content or 'job' in content or 'load' in content: return "Application"
    if 'midplane' in content or 'node' in content or 'card' in content: return "Hardware (Node)"
    return "Unknown/Other"

def detect_hybrid(model, freq_map, windows, output_file):
    logger.info(f"Running Hybrid Inference (K={K_PRODUCTION}, Rare<{RARE_THRESHOLD})...")
    
    anomalies = []
    
    with torch.no_grad():
        for item in tqdm(windows, desc="Detecting"):
            # 1. Frequency Check (Point Anomaly)
            rare_templates = [t for t in item['templates'] if is_rare_template(t, freq_map)]
            
            if rare_templates:
                # Describe first rare template found
                cat = categorize_error(rare_templates[0])
                anomalies.append({
                    'timestamp': 'N/A', # In prod, would capture window start time
                    'session_id': item['session_id'],
                    'chunk_index': item['chunk_index'],
                    'reason': 'Rare Template',
                    'category': cat,
                    'details': rare_templates,
                    'severity': 'Critical'
                })
                continue 
            
            # 2. LogGPT Check (Sequence Anomaly)
            input_ids = item['input_ids'].unsqueeze(0).to(DEVICE)
            
            targets = input_ids[:, 1:]
            inputs = input_ids[:, :-1]
            
            logits, _ = model(inputs)
            
            probs = F.softmax(logits, dim=-1)
            _, topk_indices = torch.topk(probs, k=K_PRODUCTION, dim=-1)
            
            targets_expanded = targets.unsqueeze(-1)
            matches = (topk_indices == targets_expanded)
            hits = matches.any(dim=-1)
            
            misses = ~hits
            num_misses = misses.sum().item()
            
            if num_misses > 0:
                # Categorize based on the MISSED token (if possible) or the whole window context
                # Ideally we check which token was missed.
                # For simplicity, we categorize the whole window content.
                # Or better, the LAST token in window (often the trigger).
                last_template = item['templates'][-1]
                cat = categorize_error(last_template)
                
                anomalies.append({
                    'timestamp': 'N/A',
                    'session_id': item['session_id'],
                    'chunk_index': item['chunk_index'],
                    'reason': 'Sequence Anomaly (LogGPT)',
                    'category': cat,
                    'miss_count': num_misses,
                    'severity': 'Warning' if num_misses < 3 else 'Critical'
                })
                
    logger.info(f"⚠️ Found {len(anomalies)} anomalies.")
    with open(output_file, 'w') as f:
        json.dump(anomalies, f, indent=4)
        
    logger.info(f"Report report saved to {output_file}")
    return anomalies

def main():
    parser = argparse.ArgumentParser(description="LogGPT Hybrid Detection")
    parser.add_argument("--input", type=str, required=True, help="Path to logs.csv")
    parser.add_argument("--output", type=str, default="hybrid_anomalies.json", help="Output JSON")
    args = parser.parse_args()
    
    model, tokenizer, freq_map = load_resources()
    windows = preprocess_logs(args.input, tokenizer)
    detect_hybrid(model, freq_map, windows, args.output)

if __name__ == "__main__":
    main()
