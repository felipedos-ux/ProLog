# -*- coding: utf-8 -*-
"""
SIAT Training Script
====================
Trains LogGPT-Small on SIAT normal logs.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path

from config import (
    DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, 
    OUTPUT_DIR, CHECKPOINT_PATH, SEED, EARLY_STOPPING_PATIENCE
)
# Add OUTPUT_DIR/CHECKPOINT_PATH to config if missing
if not hasattr(torch, 'CHECKPOINT_PATH'):
    CHECKPOINT_PATH = OUTPUT_DIR / "siat_loggpt.pt"

from dataset import load_data
from model import LogGPT, GPTConfig

def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seeds(SEED)
    print(f"🚀 Starting Training on {DEVICE}")
    
    # 1. Load Data
    train_ds, test_ds, tokenizer = load_data()
    print(f"Vocab Size: {tokenizer.vocab_size}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    # Validate on a subset of test (norm only? No, test has anomalies)
    # Validating loss on anomalous data is weird.
    # Usually we validate loss on hold-out NORMAL data.
    # But preprocess didn't give us validation split.
    # We will use training loss for convergence check.
    
    # 2. Config Model
    # Vocab size needs to match tokenizer
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=128,
        n_layer=4, 
        n_head=4, 
        n_embd=256, 
        dropout=0.1
    )
    model = LogGPT(config).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Train Loop
    best_loss = float('inf')
    patience = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for batch in pbar:
            x = batch["input_ids"].to(DEVICE)
            y = batch["labels"].to(DEVICE)
            
            # Forward
            _, loss = model(x, targets=y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / steps
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
        
        # Simple Early Stopping based on Training Loss (since no validation split)
        # Ideally we want loss to go down (~0.01 for log patterns)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            patience += 1
            
        if patience >= EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early stopping at epoch {epoch}")
            break
            
    print(f"✅ Training Complete. Model saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()
