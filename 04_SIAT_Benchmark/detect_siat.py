# -*- coding: utf-8 -*-
"""
SIAT Detection Script
=====================
Applies the calibrated model to the TEST set.
Outputs: results_siat.pkl
"""
import torch
import numpy as np
import pickle
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import (
    DEVICE, BATCH_SIZE, CHECKPOINT_PATH, 
    OUTPUT_DIR, SESSION_DATA
)
from model import LogGPT, GPTConfig
from dataset import load_data

THRESHOLD_CONFIG = OUTPUT_DIR / "threshold_config.json"
RESULTS_FILE = OUTPUT_DIR / "results_siat.pkl"

def main():
    print(f"🕵️ Starting Detection using {CHECKPOINT_PATH}")
    
    # 1. Load Threshold
    if not THRESHOLD_CONFIG.exists():
        print(f"Error: Threshold config not found at {THRESHOLD_CONFIG}")
        return
    
    with open(THRESHOLD_CONFIG, 'r') as f:
        t_config = json.load(f)
        threshold = t_config["threshold"]
    
    print(f"Threshold loaded: {threshold:.4f}")

    # 2. Load Data (Test Set)
    train_ds, test_ds, tokenizer = load_data()
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Load Model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size, 
        block_size=128
    )
    model = LogGPT(config).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    
    # 4. Detect
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Detecting")):
            x = batch["input_ids"].to(DEVICE)
            y_label = batch["anom_label"].cpu().numpy() # 0 or 1
            
            # Forward (we want loss per session)
            # LogGPT forward returns mean loss if labels provided.
            # But we need per-sample loss.
            # We pass targets=x to get logits, but ignore return loss (it's mean).
            logits, _ = model(x, targets=x)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = x[:, 1:].contiguous()
            
            # Loss per sample
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_targets.view(-1), 
                reduction='none'
            )
            loss = loss.view(x.size(0), -1)
            
            # Mask padding handling (simple tokenizer 0 is PAD? Or UNK?)
            # In dataset padding is 0. 
            padding_mask = (shift_targets != 0).float()
            seq_lens = padding_mask.sum(dim=1)
            seq_lens[seq_lens == 0] = 1.0
            
            sample_losses = (loss * padding_mask).sum(dim=1) / seq_lens
            sample_losses = sample_losses.cpu().numpy()
            
            # Store results
            for j in range(len(sample_losses)):
                loss_val = float(sample_losses[j])
                is_detected = loss_val > threshold
                true_label = int(y_label[j])
                
                results.append({
                    "loss": loss_val,
                    "label": true_label,
                    "pred": int(is_detected),
                    "threshold": threshold
                })
                
    # 5. Save Results
    with open(RESULTS_FILE, "wb") as f:
        pickle.dump(results, f)
        
    print(f"✅ Detection complete. Saved {len(results)} results to {RESULTS_FILE}")
    
    # Quick Stats
    res_labels = np.array([r['label'] for r in results])
    res_preds = np.array([r['pred'] for r in results])
    
    tp = ((res_labels == 1) & (res_preds == 1)).sum()
    fp = ((res_labels == 0) & (res_preds == 1)).sum()
    fn = ((res_labels == 1) & (res_preds == 0)).sum()
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    print(f"\nQuick Validation Stats:")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    main()
