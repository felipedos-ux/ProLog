import torch
import torch.nn.functional as F
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from model import LogGPT
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, precision_recall_curve

# Configuration
MODEL_NAME = "distilgpt2"
MODEL_DIR = "model_weights"
DATA_PATH = "../data/OpenStack_data_original.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SKIP_START_LOGS = 3  # Based on Task 1.1 findings
SEED = 42

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pl.read_csv(DATA_PATH, infer_schema_length=10000)
    
    # Split IDs
    normal_ids = df.filter(pl.col("anom_label") == 0)["test_id"].unique().to_list()
    anomaly_ids = df.filter(pl.col("anom_label") == 1)["test_id"].unique().to_list()
    
    # Normal Split (Train/Val/Test)
    # 80% Train, 10% Val, 10% Test
    train_ids, test_val_ids = train_test_split(normal_ids, test_size=0.2, random_state=SEED)
    val_norm_ids, test_norm_ids = train_test_split(test_val_ids, test_size=0.5, random_state=SEED)
    
    # Anomaly Split (Val/Test)
    val_anom_ids, test_anom_ids = train_test_split(anomaly_ids, test_size=0.5, random_state=SEED)
    
    print("Split Stats:")
    print(f"   Normal Val: {len(val_norm_ids)} | Normal Test: {len(test_norm_ids)}")
    print(f"   Anom Val:   {len(val_anom_ids)}  | Anom Test:   {len(test_anom_ids)}")
    
    return df, val_norm_ids, val_anom_ids

def get_session_max_loss(model, tokenizer, session_text, config):
    # This logic must match detect_custom.py EXACTLY (minus the debug prints)
    templates = session_text.split(" \n ") # Use the FIXED separator
    if len(templates) <= SKIP_START_LOGS:
        return 0.0
        
    context_ids = []
    max_session_loss = 0.0
    
    MAX_CONTEXT_LEN = config.block_size
    
    for i in range(len(templates)):
        current_log = templates[i]
        # Same construction logic
        text = (" \n " if i > 0 else "") + current_log
        new_ids = tokenizer.encode(text)
        
        if i == 0:
            context_ids.extend(new_ids)
            continue
            
        # Prepare Input
        full_seq = context_ids + new_ids
        if len(full_seq) > MAX_CONTEXT_LEN:
            input_seq = full_seq[-MAX_CONTEXT_LEN:]
            target_start_idx = len(input_seq) - len(new_ids)
        else:
            input_seq = full_seq
            target_start_idx = len(context_ids)
            
        x = torch.tensor(input_seq, dtype=torch.long, device=DEVICE).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = model(x)
        
        # Extract relevant logits
        target_indices = range(target_start_idx, len(input_seq))
        logit_indices = [idx - 1 for idx in target_indices]
        
        if not logit_indices or logit_indices[0] < 0:
             loss_val = 0
        else:
            relevant_logits = logits[0, logit_indices, :]
            relevant_targets = torch.tensor(input_seq[target_start_idx:], dtype=torch.long, device=DEVICE)
            loss_val = F.cross_entropy(relevant_logits, relevant_targets).item()
        
        # SKIP START LOGS LOGIC
        if i < SKIP_START_LOGS:
            continue
            
        if loss_val > max_session_loss:
            max_session_loss = loss_val
            
        # Update context
        context_ids.extend(new_ids)
        if len(context_ids) > MAX_CONTEXT_LEN:
            context_ids = context_ids[-MAX_CONTEXT_LEN:]
            
    return max_session_loss

def calibrate():
    # 1. Load Model
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = torch.load(f"{MODEL_DIR}/config.pt", weights_only=False)
    model = LogGPT(config)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/loggpt_weights.pt", weights_only=False))
    model.to(DEVICE)
    model.eval()
    
    # 2. Load Data
    df, val_norm_ids, val_anom_ids = load_data()
    
    # 3. Calculate Losses
    print("Calculating Validation Losses...")
    y_true = []
    y_scores = []
    
    # Pre-group data
    sessions = (
        df.filter(pl.col("EventTemplate").is_not_null())
        .group_by("test_id")
        .agg(pl.col("EventTemplate"))
        .select(["test_id", "EventTemplate"])
    )
    
    session_map = {row[0]: " \n ".join(row[1]) for row in sessions.rows()}
    
    # Normal
    for tid in tqdm(val_norm_ids, desc="Normal Val"):
        text = session_map.get(tid)
        if text:
            loss = get_session_max_loss(model, tokenizer, text, config)
            y_true.append(0)
            y_scores.append(loss)
        
    # Anomaly
    for tid in tqdm(val_anom_ids, desc="Anomaly Val"):
        text = session_map.get(tid)
        if text:
            loss = get_session_max_loss(model, tokenizer, text, config)
            y_true.append(1)
            y_scores.append(loss)
    # 4. Grid Search Threshold
    print("Grid Searching Threshold...")
    thresholds = np.arange(1.0, 25.0, 0.1)
    best_f1 = 0
    best_th = 0
    best_metrics = (0, 0, 0) # P, R, F1
    
    precision_list = []
    recall_list = []
    f1_list = []
    
    for th in thresholds:
        y_pred = [1 if s > th else 0 for s in y_scores]
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_metrics = (p, r, f1)
            
    print(f"\n✅ BEST THRESHOLD: {best_th:.2f}")
    print(f"   Precision: {best_metrics[0]:.4f}")
    print(f"   Recall:    {best_metrics[1]:.4f}")
    print(f"   F1 Score:  {best_metrics[2]:.4f}")
    
    print("   f1_list:", f1_list) # Debug
    
    # Save Threshold
    with open("optimal_threshold.txt", "w") as f:
        f.write(str(best_th))
    print("Saved to optimal_threshold.txt")
    
    # Plots
    print("Generating Plots...")
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, f1_list, label='F1 Score')
    plt.plot(thresholds, precision_list, label='Precision', linestyle='--')
    plt.plot(thresholds, recall_list, label='Recall', linestyle='--')
    plt.axvline(best_th, color='r', linestyle=':', label=f'Best Th={best_th:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Threshold Calibration (SkipStart={SKIP_START_LOGS})')
    plt.legend()
    plt.grid(True)
    plt.savefig('calibration_curve.png')
    print("Saved calibration_curve.png")

if __name__ == "__main__":
    calibrate()
