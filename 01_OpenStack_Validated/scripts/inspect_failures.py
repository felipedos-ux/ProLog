import torch
import polars as pl
from model import LogGPT, GPTConfig
from transformers import AutoTokenizer
import os

from config import DATA_FILE, MODEL_DIR, THRESHOLD as DEFAULT_THRESHOLD, INFER_SCHEMA_LENGTH

# Config
# DATA_FILE imported from config
MODEL_PATH = f"{MODEL_DIR}/loggpt_weights.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load adaptive threshold if exists
import json
THRESHOLD = 19.3675 
if os.path.exists("threshold_config.json"):
    with open("threshold_config.json", "r") as f:
        THRESHOLD = json.load(f).get("threshold", 19.3675)

def inspect_session(tid):
    print(f"\n🔍 Inspecting Session {tid}...")
    
    # Load Data
    # Debug
    print(f"DEBUG: DATA_FILE from config is: {DATA_FILE}")
    
    path_to_load = str(DATA_FILE)
    if not os.path.exists(path_to_load):
        print(f"WARNING: {path_to_load} not found. Trying fallback.")
        path_to_load = "../data/OpenStack_data_original.csv"
        
    print(f"Loading data from: {path_to_load}")
    df = pl.read_csv(path_to_load, infer_schema_length=5000)
    session_df = df.filter(pl.col("test_id") == tid).sort("time_hour")
    
    if session_df.is_empty():
        print("Session not found!")
        return

    logs = session_df["EventTemplate"].to_list()
    timestamps = session_df["time_hour"].to_list()
    
    # Init Model
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    vocab_size = tokenizer.vocab_size + 100 # Match training buffer
    config = GPTConfig(vocab_size=vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=256)
    model = LogGPT(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval()
    
    context = []
    MAX_CONTEXT = 128
    
    print(f"{'Time':<25} | {'Loss':<8} | {'Th?':<3} | {'Log Content'}")
    print("-" * 100)
    
    for i, (ts, log) in enumerate(zip(timestamps, logs)):
        if log is None: log = ""
            
        # Prepare input
        text = (" \n " if i > 0 else "") + log
        new_ids = tokenizer.encode(text)
        
        # Eval only if we have context
        loss_val = 0.0
        is_alert = False
        
        if i > 5: # Skip start
            full_seq = context + new_ids
            # Truncate
            if len(full_seq) > MAX_CONTEXT:
                inp = torch.tensor(full_seq[-MAX_CONTEXT:], dtype=torch.long).unsqueeze(0).to(DEVICE)
                target_start = MAX_CONTEXT - len(new_ids)
            else:
                inp = torch.tensor(full_seq, dtype=torch.long).unsqueeze(0).to(DEVICE)
                target_start = len(context)
                
            with torch.no_grad():
                logits, _ = model(inp)
                # Calculate loss only on new tokens
                # logits: [1, seq_len, vocab]
                # target: inp
                
                # Align logic with detect_custom.py
                # We want loss for the *new* tokens, given previous
                # logits[t] predicts inp[t+1]
                
                # Slice logits to predict new_ids
                # positions: [target_start-1 ... end-1] predict [target_start ... end]
                
                shift_logits = logits[:, target_start-1:-1, :].contiguous()
                shift_labels = inp[:, target_start:].contiguous()
                
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='mean'
                )
                loss_val = loss.item()
                if loss_val > THRESHOLD:
                    is_alert = True
        
        alert_mark = "⚠️" if is_alert else " "
        print(f"{ts:<25} | {loss_val:<8.4f} | {alert_mark:<3} | {log[:60]}...")
        
        context.extend(new_ids)
        if len(context) > MAX_CONTEXT:
            context = context[-MAX_CONTEXT:]

if __name__ == "__main__":
    # Inspect the worst cases from the user's list
    # IDs: 6, 12, 3
    inspect_session(6)
    inspect_session(12)
    inspect_session(3)
