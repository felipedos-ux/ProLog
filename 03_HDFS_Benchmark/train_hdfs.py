import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, AutoTokenizer
import polars as pl
from tqdm import tqdm
import os
import math
import random

# Import Model (copied locally)
from model import LogGPT
from utils.logger import setup_logger
import config_hdfs as config

logger = setup_logger(__name__)

class HDFSDataset(Dataset):
    def __init__(self, data_list, tokenizer, block_size):
        self.examples = []
        # Pre-tokenize all data
        # data_list is list of strings: "E1 E2 E5 ..."
        
        logger.info(f"Tokenizing {len(data_list)} sessions...")
        
        # Batch tokenization for speed
        # tokenizer(list_of_strings)
        batch_size = 1000
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i : i+batch_size]
            encodings = tokenizer(
                batch, 
                truncation=True, 
                max_length=block_size, 
                padding=False,
                return_attention_mask=False
            )
            
            for input_ids in encodings['input_ids']:
                self.examples.append(torch.tensor(input_ids, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

def collate_fn(batch):
    # Dynamic Padding with EOS/Pad token (GPT2 uses 50256 usually)
    max_len = max(len(x) for x in batch)
    padded = torch.full((len(batch), max_len), 50256, dtype=torch.long)
    
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
        
    return padded

def prepare_data(tokenizer):
    train_path = config.DATA_DIR / "HDFS_train.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found. Run dataset_hdfs.py first.")
        
    logger.info("Loading Training Data...")
    df = pl.read_csv(str(train_path))
    
    # Extract "EventTemplate" column which contains "E1 E2 ..."
    # Ensure it's string
    train_data = df["EventTemplate"].to_list()
    
    # Split Train/Val (Sessions)
    # Shuffle
    random.shuffle(train_data)
    
    n_total = len(train_data)
    n_train = int(n_total * 0.9) # 90% Train, 10% Val (since we have separate Test file)
    
    train_seqs = train_data[:n_train]
    val_seqs = train_data[n_train:]
    
    logger.info(f"Train: {len(train_seqs)} | Val: {len(val_seqs)}")
    
    train_ds = HDFSDataset(train_seqs, tokenizer, config.BLOCK_SIZE)
    val_ds = HDFSDataset(val_seqs, tokenizer, config.BLOCK_SIZE)
    
    return train_ds, val_ds

def train_epoch(model, loader, optimizer, device, epoch, scaler):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for batch in pbar:
        # GPT targets are inputs shifted by 1. 
        # But here inputs include the whole sequence.
        # LogGPT/GPT2LMHeadModel logic:
        # If we pass labels=input_ids, the model shifts internally (if using HF model)
        # BUT our model.py implements `loss = F.cross_entropy(logits..., targets...)`
        # It does NOT shift internally. We must shift manually or ensure model.py does it.
        # Checking model.py:
        # forward(idx, targets=None): if targets: loss = cross_entropy(logits, targets)
        # It calculates loss for ALL positions.
        # Usually for GPT: Input: x_0, ..., x_{n-1} -> Target: x_1, ..., x_n
        
        # Data preparation:
        # X = batch[:, :-1]
        # Y = batch[:, 1:]
        
        inp = batch[:, :-1].to(device)
        tgt = batch[:, 1:].to(device)
        
        # Mixed Precision
        with torch.cuda.amp.autocast(enabled=(device=='cuda')):
            logits, loss = model(inp, targets=tgt)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
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
    logger.info("Starting HDFS Training...")
    config.set_seeds()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    # GPT2 tokenizer doesn't have pad token by default
    tokenizer.pad_token = tokenizer.eos_token
    
    # Data
    train_ds, val_ds = prepare_data(tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Model
    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_layer=config.N_LAYER,
        n_head=config.N_HEAD,
        n_embd=config.N_EMBD,
        dropout=config.DROPOUT
    )
    # Override block size in config for attention mask
    model_config.block_size = config.BLOCK_SIZE
    
    model = LogGPT(model_config)
    model.to(config.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.DEVICE == 'cuda'))
    
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config.EPOCHS + 1):
        t_loss = train_epoch(model, train_loader, optimizer, config.DEVICE, epoch, scaler)
        v_loss = evaluate(model, val_loader, config.DEVICE)
        
        logger.info(f"Epoch {epoch} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            logger.info(f"Saving Best Model to {config.MODEL_DIR}...")
            torch.save(model.state_dict(), config.MODEL_DIR / "hdfs_loggpt.pt")
            torch.save(model_config, config.MODEL_DIR / "config.pt")
            
    logger.info("Training Complete.")

if __name__ == "__main__":
    main()
