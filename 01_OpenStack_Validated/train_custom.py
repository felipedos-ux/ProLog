"""
LogGPT Training for OpenStack — Identical to HDFS approach.

Key design decisions (matching HDFS train_hdfs.py):
1. Each session = 1 training example (NO group_texts / global concatenation)
2. Dynamic padding with EOS token 
3. inp = batch[:, :-1], tgt = batch[:, 1:] — proper causal LM shift
4. GPT2 tokenizer on space-separated EventIds ("E1 E2 E5")
"""

import torch
import math
import os
import random
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from model import LogGPT, GPTConfig
from dataset import (
    load_openstack_data, prepare_session_strings,
    LogSessionDataset, collate_fn
)
from config import (
    MODEL_NAME, MODEL_DIR as OUTPUT_DIR, 
    BLOCK_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE,
    N_LAYER, N_HEAD, N_EMBD, DROPOUT, SEED, set_seeds,
    TEST_SIZE_NORMAL
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for batch in pbar:
        # Proper causal LM shift (same as HDFS)
        inp = batch[:, :-1].to(device)
        tgt = batch[:, 1:].to(device)
        
        logits, loss = model(inp, targets=tgt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            inp = batch[:, :-1].to(device)
            tgt = batch[:, 1:].to(device)
            _, loss = model(inp, targets=tgt)
            total_loss += loss.item()
            
    return total_loss / len(loader)


def main():
    logger.info("🚀 LogGPT Training (OpenStack — HDFS-style approach)")
    set_seeds()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Tokenizer (GPT2 — same as HDFS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Prepare data — session-level (same as HDFS)
    logger.info("Loading data...")
    df = load_openstack_data()
    
    # Get normal sessions as space-separated EventId strings
    normal_sessions = prepare_session_strings(df, label_filter=0)
    session_texts = normal_sessions["EventTemplate"].to_list()
    
    logger.info(f"Total normal sessions: {len(session_texts)}")
    
    # Shuffle and split (same as HDFS: 90/10)
    random.seed(SEED)
    random.shuffle(session_texts)
    
    n_train = int(len(session_texts) * 0.9)
    train_seqs = session_texts[:n_train]
    val_seqs = session_texts[n_train:]
    
    logger.info(f"Train: {len(train_seqs)} | Val: {len(val_seqs)}")
    
    # Create datasets (each session = 1 example, same as HDFS HDFSDataset)
    train_ds = LogSessionDataset(train_seqs, tokenizer, BLOCK_SIZE)
    val_ds = LogSessionDataset(val_seqs, tokenizer, BLOCK_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 3. Model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT
    )
    model = LogGPT(config)
    model.to(DEVICE)
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model: {param_count:.2f}M params | Dropout={DROPOUT}")
    
    # 4. Optimizer (same as HDFS — plain AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training loop (same as HDFS)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, EPOCHS + 1):
        t_loss = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
        v_loss = evaluate(model, val_loader, DEVICE)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        
        t_ppl = math.exp(t_loss) if t_loss < 20 else float('inf')
        v_ppl = math.exp(v_loss) if v_loss < 20 else float('inf')
        
        logger.info(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train: {t_loss:.4f} (PPL {t_ppl:.2f}) | "
            f"Val: {v_loss:.4f} (PPL {v_ppl:.2f})"
        )
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            logger.info(f"✅ Saving Best Model to {OUTPUT_DIR}...")
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/loggpt_weights.pt")
            torch.save(config, f"{OUTPUT_DIR}/config.pt")
    
    # Save training curve
    curve_data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_losses),
        "config": {
            "n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD,
            "dropout": DROPOUT, "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE, "block_size": BLOCK_SIZE,
        }
    }
    with open(f"{OUTPUT_DIR}/training_curve.json", "w") as f:
        json.dump(curve_data, f, indent=2)
    
    logger.info(f"🎉 Done! Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
