# -*- coding: utf-8 -*-
"""
SIAT Calibration Script
=======================
Calculates k-sigma dynamic threshold on TRAINING set (normal sessions).
"""
import torch
import numpy as np
import json
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import (
    DEVICE, BATCH_SIZE, CHECKPOINT_PATH, 
    OUTPUT_DIR, SESSION_DATA, VOCAB_SIZE_ESTIMATE
)
from model import LogGPT, GPTConfig
from dataset import load_data, SimpleTokenizer

THRESHOLD_CONFIG = OUTPUT_DIR / "threshold_config.json"

def main():
    print(f"⚖️ Starting Calibration using {CHECKPOINT_PATH}")
    
    # 1. Load Data
    # For calibration, we need NORMAL sessions from TRAIN (or a holdout set).
    # dataset.SIATDataset('train') returns ONLY normal sessions.
    train_ds, test_ds, tokenizer = load_data()
    calib_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size, 
        block_size=128
    )
    model = LogGPT(config).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    
    # 3. Collect Losses on Normal Data
    normal_losses = []
    
    print("Collecting losses on normal sessions...")
    with torch.no_grad():
        for batch in tqdm(calib_loader, desc="Calibrating"):
            x = batch["input_ids"].to(DEVICE)
            y = batch["labels"].to(DEVICE) # same as x
            
            # Forward pass
            # We want per-sample loss. CrossEntropyLoss defaults to mean.
            # We must compute manually or use reduction='none' inside model?
            # Model returns scalar loss.
            # Workaround: For calibration, we can't easily change model signature without breaking trained weights?
            # Actually, model signature `forward(idx, targets)` returns `(logits, loss)`.
            # We can re-compute loss here with reduction='none'.
            
            logits, _ = model(x)
            
            # Shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = y[:, 1:].contiguous()
            
            # Compute cross entropy per token, then mean per sequence
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_targets.view(-1), 
                reduction='none'
            )
            
            loss = loss.view(x.size(0), -1)
            # Mean over sequence length (excluding padding? SimpleTokenizer uses 0 as PAD)
            # Mask padding
            padding_mask = (shift_targets != 0).float()
            seq_lens = padding_mask.sum(dim=1)
            
            # Avoid div by zero
            seq_lens[seq_lens == 0] = 1.0
            
            sample_losses = (loss * padding_mask).sum(dim=1) / seq_lens
            normal_losses.extend(sample_losses.cpu().numpy())

    normal_losses = np.array(normal_losses)
    
    # 4. Calculate K-Sigma Threshold from Normal Distribution
    mu = np.mean(normal_losses)
    sigma = np.std(normal_losses)
    
    # HDFS used k=8. For SIAT (cleaner data), maybe k=10 or adaptive?
    # Let's try to find a safe k that covers 99.9% of normal data
    # (3 sigma covers 99.7%, but distribution might be long-tailed)
    
    # Adaptive strategy: Set threshold at Max(Normal) + epsilon?
    # Or 99.9th percentile + margin
    p999 = np.percentile(normal_losses, 99.9)
    max_loss = np.max(normal_losses)
    
    # Conservative strategy: k=10 sigma
    k = 10
    threshold_k_sigma = mu + k * sigma
    
    # Hybrid strategy: Max of (k-sigma, max_normal_seen)
    final_threshold = max(threshold_k_sigma, max_loss * 1.05) # 5% margin above max seen
    
    print(f"\nStats for Normal Losses:")
    print(f"  Mean: {mu:.4f}")
    print(f"  Std:  {sigma:.4f}")
    print(f"  Max:  {max_loss:.4f}")
    print(f"  P99.9:{p999:.4f}")
    print(f"  K-Sigma ({k}): {threshold_k_sigma:.4f}")
    print(f"  Selected Threshold: {final_threshold:.4f}")
    
    config_data = {
        "mean": float(mu),
        "std": float(sigma),
        "k": k,
        "max_normal": float(max_loss),
        "threshold": float(final_threshold)
    }
    
    with open(THRESHOLD_CONFIG, 'w') as f:
        json.dump(config_data, f, indent=2)
        
    print(f"✅ Saved threshold to {THRESHOLD_CONFIG}")

if __name__ == "__main__":
    main()
