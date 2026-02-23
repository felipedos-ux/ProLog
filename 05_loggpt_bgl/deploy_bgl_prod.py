import torch
import torch.nn.functional as F
from transformers import GPT2Config, AutoTokenizer
import polars as pl
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm

from model import LogGPT
from utils.logger import setup_logger
import config_count as config

# Hardcoded Production Defaults
MODEL_PATH = config.MODEL_DIR 
BLOCK_SIZE = config.BLOCK_SIZE
DEVICE = config.DEVICE
K_PRODUCTION = 5 # As recommended in Final Report

logger = setup_logger(__name__)

def load_production_model():
    """Loads the trained LogGPT model and tokenizer for production inference."""
    logger.info(f"Loading Production Model from {MODEL_PATH}...")
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Model Config
    config_path = MODEL_PATH / "config.pt"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
        
    model_config = torch.load(config_path, weights_only=False) # Safe load
    
    model = LogGPT(model_config)
    
    # Load Weights
    weights_path = MODEL_PATH / "loggpt_weights.pt"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")
        
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    
    return model, tokenizer

def preprocess_logs(log_file, tokenizer, window_size=20):
    """
    Reads a raw log file (or CSV), groups by Node/Session, and creates windows.
    For BGL production, we assume logs come in a stream or file.
    Here we handle the CSV format used in training for consistency.
    """
    logger.info(f"Processing logs from {log_file}...")
    
    # Check extension
    if str(log_file).endswith('.csv'):
        df = pl.read_csv(str(log_file), infer_schema_length=10000)
    else:
        # TODO: Implement raw parser if needed. 
        # For now assume structured CSV with 'EventTemplate' and 'Node'
        raise NotImplementedError("Production script currently requires structured CSV input.")

    # Group by Session/Node
    grouped = df.group_by(config.SESSION_ID_COL, maintain_order=True).agg(pl.col("EventTemplate"))
    rows = grouped.to_dict(as_series=False)
    
    windows = []
    
    for i, templates in enumerate(tqdm(rows["EventTemplate"], desc="Windowing")):
        session_id = rows[config.SESSION_ID_COL][i]
        
        # Sliding or Fixed? Count-based training used FIXED split.
        # Production should likely mirror this or be sliding.
        # Let's use FIXED NON-OVERLAPPING chunks of 20 to match training exactly.
        
        for k in range(0, len(templates), window_size):
            chunk = templates[k:k+window_size]
            if len(chunk) < 2: continue # Need at least 2 logs (input -> next)
            
            text = " \n ".join([str(t) for t in chunk])
            input_ids = tokenizer.encode(text)
            
            if len(input_ids) > BLOCK_SIZE:
                input_ids = input_ids[:BLOCK_SIZE]
                
            windows.append({
                'session_id': session_id,
                'chunk_index': k // window_size,
                'input_ids': torch.tensor(input_ids, dtype=torch.long)
            })
            
    return windows

def detect_anomalies(model, windows, output_file):
    """Runs inference and saves anomalies to output file."""
    logger.info(f"Running Inference on {len(windows)} windows (K={K_PRODUCTION})...")
    
    anomalies = []
    
    with torch.no_grad():
        for item in tqdm(windows, desc="Detecting"):
            input_ids = item['input_ids'].unsqueeze(0).to(DEVICE) # [1, T]
            
            if input_ids.size(1) < 2: continue
            
            targets = input_ids[:, 1:]
            inputs = input_ids[:, :-1]
            
            logits, _ = model(inputs)
            
            # Top-K Logic
            probs = F.softmax(logits, dim=-1)
            _, topk_indices = torch.topk(probs, k=K_PRODUCTION, dim=-1)
            
            # Check hits
            targets_expanded = targets.unsqueeze(-1)
            matches = (topk_indices == targets_expanded)
            hits = matches.any(dim=-1) # [1, T]
            
            misses = ~hits
            num_misses = misses.sum().item()
            
            # Anomaly Condition: ANY miss in the sequence (Threshold > 0)
            if num_misses > 0:
                anomalies.append({
                    'session_id': item['session_id'],
                    'chunk_index': item['chunk_index'],
                    'miss_count': num_misses,
                    'window_length': inputs.size(1),
                    'severity': 'Critical' if num_misses > 5 else 'Warning'
                })
                
    # Save Report
    logger.info(f"⚠️ Found {len(anomalies)} anomalies.")
    with open(output_file, 'w') as f:
        json.dump(anomalies, f, indent=4)
        
    logger.info(f"Report saved to {output_file}")
    return anomalies

def main():
    parser = argparse.ArgumentParser(description="LogGPT BGL Production Detection")
    parser.add_argument("--input", type=str, required=True, help="Path to input log file (CSV)")
    parser.add_argument("--output", type=str, default="bgl_anomalies.json", help="Output JSON report")
    args = parser.parse_args()
    
    model, tokenizer = load_production_model()
    windows = preprocess_logs(args.input, tokenizer)
    detect_anomalies(model, windows, args.output)

if __name__ == "__main__":
    main()
