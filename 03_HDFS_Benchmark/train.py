"""
LogGPT Training Script.
Identical logic to OpenStack: trains on Normal sessions only via Next-Token Prediction.
"""

import os
import torch
import math
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from tqdm import tqdm

from dataset import prepare_llm_dataset
from model import LogGPT, GPTConfig
from config import (
    MODEL_NAME, MODEL_DIR as OUTPUT_DIR, 
    BLOCK_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE,
    VOCAB_BUFFER, DROPOUT, SEED, set_seeds
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


def train_epoch(model, loader, optimizer, device, epoch_idx):
    """Trains the model for one epoch."""
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch_idx} [Train]")
    total_loss = 0.0
    steps = 0
    
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        _, loss = model(input_ids, targets=labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        steps += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / steps if steps > 0 else 0.0


def evaluate_epoch(model, loader, device):
    """Evaluates the model on validation set."""
    model.eval()
    total_loss = 0.0
    steps = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            _, loss = model(input_ids, targets=labels)
            total_loss += loss.item()
            steps += 1
            
    return total_loss / steps if steps > 0 else 0.0


def main():
    logger.info("🚀 LogGPT Training (HDFS - OpenStack Pipeline)")
    set_seeds()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Tokenizer
    logger.info("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    logger.info(f"Vocab Size: {vocab_size}")
    
    # 2. Prepare Dataset
    logger.info("Preparing Dataset...")
    lm_datasets = prepare_llm_dataset(tokenizer, block_size=BLOCK_SIZE)
    
    if isinstance(lm_datasets, dict):
         train_dataset = lm_datasets["train"]
         val_dataset = lm_datasets.get("test") or lm_datasets.get("validation")
    else:
         split = lm_datasets.train_test_split(test_size=0.1, seed=SEED)
         train_dataset = split["train"]
         val_dataset = split["test"]
         
    logger.info(f"Split Dataset: {len(train_dataset)} Train, {len(val_dataset)} Val")
    
    # Data Loader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)
    
    # 3. Initialize Model
    logger.info("Initializing LogGPT-Small...")
    config = GPTConfig(
        vocab_size=vocab_size + VOCAB_BUFFER,
        block_size=BLOCK_SIZE,
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=DROPOUT
    )
    model = LogGPT(config)
    model.to(DEVICE)
    
    param_count = sum(p.numel() for p in model.parameters())/1e6
    logger.info(f"Model params: {param_count:.2f}M")
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training Loop (with Early Stopping)
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    logger.info(f"🚀 Starting training for {EPOCHS} epochs (Patience={patience})...")
    
    for epoch in range(1, EPOCHS + 1):
        avg_train_loss = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
        avg_val_loss = evaluate_epoch(model, val_loader, DEVICE)
        
        ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float('inf')
        logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | PPL: {ppl:.2f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            logger.info(f"✅ New Best Model! Saving to {OUTPUT_DIR}...")
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/hdfs_loggpt.pt")
            torch.save(config, f"{OUTPUT_DIR}/config.pt")
        else:
            patience_counter += 1
            logger.info(f"⚠️ No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            logger.info(f"🛑 Early stopping triggered at Epoch {epoch}")
            break
            
    logger.info(f"🎉 Training Complete. Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
