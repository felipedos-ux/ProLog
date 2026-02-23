import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, AutoTokenizer
import polars as pl
from tqdm import tqdm
import os
import math
import random

from model import LogGPT
from utils.logger import setup_logger

# IMPORT COUNT CONFIG
import config_count as config
from config_count import (
    MODEL_NAME, BLOCK_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE,
    DATA_FILE, OUTPUT_DIR, MODEL_DIR, VOCAB_BUFFER, DROPOUT, 
    TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT, SEED,
    SESSION_ID_COL, LABEL_COL, TIMESTAMP_COL
)

logger = setup_logger(__name__)

class LogDataset(Dataset):
    def __init__(self, tokenized_data, block_size):
        self.examples = []
        for item in tokenized_data:
            input_ids = item['input_ids']
            # O dataset já vem em chunks pequenos (janelas de 20 logs)
            # Mas o tokenizer pode ter gerado sequencias maiores que block_size?
            # block_size = 64. Janela 20 logs * 3 tokens = 60. Deve caber.
            
            if len(input_ids) > block_size:
                input_ids = input_ids[:block_size]
            
            # Padding se for muito pequeno? 
            # LogGPT geralmente não usa padding se batch=1, mas com batch > 1 precisa.
            # Vamos usar Collate_fn dinâmico ou padding fixo?
            # Fixo é mais simples. Pad Token = EOS Token.
            
            # OBS: Next Token Prediction precisa de labels = input_ids deslocado.
            # HuggingFace DataCollatorForLanguageModeling faz isso?
            # Vamos fazer manual para controle total.
            
            self.examples.append(torch.tensor(input_ids, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

def collate_fn(batch):
    # Dynamic Padding to Max in Batch
    max_len = max(len(x) for x in batch)
    padded_batch = torch.zeros((len(batch), max_len), dtype=torch.long) # Pad with 0? Or EOS?
    # Tokenizer pad token id serÃ¡ obtido no main.
    # Mas aqui vamos assumir 0 se não for passado. 
    # Melhor passar tokenizer para collate ou padronizar.
    # Vamos padronizar com tokenizer.pad_token_id no main.
    
    # Simpler: just pad with last token (eos) if needed, or 50256 (gpt2 eos).
    # Vamos usar pad_token_id = 50256 (default GPT2).
    pad_id = 50256
    padded_batch.fill_(pad_id)
    
    for i, x in enumerate(batch):
        padded_batch[i, :len(x)] = x
        
    return padded_batch

def prepare_llm_dataset(tokenizer, block_size=128):
    logger.info(f"Loading data from {DATA_FILE}...")
    
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Run dataset_bgl_count.py first.")
        
    df = pl.read_csv(str(DATA_FILE), infer_schema_length=10000)
    
    # Filter NORMAL only for training (Unsupervised)
    # Mas dataset_bgl_count já filtrou?
    # Vamos conferir. dataset_bgl_count.py tem "if is_anom: continue".
    # Então ja é limpo.
    
    # Split Train/Val at SESSION level (Window ID here)
    session_ids = df[SESSION_ID_COL].unique().to_list()
    random.shuffle(session_ids)
    
    n_train = int(len(session_ids) * (1 - TEST_SIZE_NORMAL))
    train_ids = set(session_ids[:n_train])
    val_ids = set(session_ids[n_train:])
    
    logger.info(f"Split: {len(train_ids)} Train Windows, {len(val_ids)} Val Windows")

    # Group and Tokenize
    # Como já são janelas curtas, não precisamos agrupar por sessão e scannear.
    # Cada LINHA (ou grupo de linhas com mesmo SessionID) é uma sequência.
    
    # O dataset salva linha a linha.
    # Precisamos reagrupar para montar o texto da janela.
    
    df_train = df.filter(pl.col(SESSION_ID_COL).is_in(train_ids))
    df_val = df.filter(pl.col(SESSION_ID_COL).is_in(val_ids))
    
    def tokenize_df(target_df):
        grouped = target_df.group_by(SESSION_ID_COL, maintain_order=True).agg(pl.col("EventTemplate"))
        rows = grouped.to_dict(as_series=False)
        
        tokenized_data = []
        
        for templates in tqdm(rows["EventTemplate"], desc="Tokenizing"):
            # Join logs
            text = " \n ".join([str(t) for t in templates])
            ids = tokenizer.encode(text)
            tokenized_data.append({'input_ids': ids})
            
        return tokenized_data

    logger.info("Tokenizing Train...")
    train_data = tokenize_df(df_train)
    logger.info("Tokenizing Val...")
    val_data = tokenize_df(df_val)
    
    return LogDataset(train_data, block_size), LogDataset(val_data, block_size)

def train_epoch(model, dataloader, optimizer, device, epoch, scaler):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch in pbar:
        batch = batch.to(device)
        
        # Inputs e Targets
        # GPT2: input = T text, target = T text (shifted inside model usually??)
        # Huggingface GPT2LMHeadModel computes loss automatically if labels provided.
        # LogGPT custom implementation might differ.
        # Let's check LogGPT model.forward...
        # It usually returns logits, loss.
        # If labels provided.
        
        # We need to shift manually or pass labels?
        # Model signature: forward(self, x, targets=None)
        
        # Se LogGPT espera targets, é x shifted.
        
        # x: [B, T]
        # targets: [B, T]
        
        optimizer.zero_grad()
        
        # Shift in Dataset or here?
        # Custom LogGPT takes 'x' and 'targets'
        # x = inputs[:, :-1]
        # y = inputs[:, 1:]
        
        # BUT if batch has padding, we need to mask loss?
        # LogGPT loss function likely handles it if cross_entropy ignore_index used?
        # Let's assume standard causal mask logic inside model.
        
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        with torch.amp.autocast('cuda', enabled=(device=='cuda')):
             logits, loss = model(inputs, targets)
             
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return total_loss / len(dataloader)

def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            logits, loss = model(inputs, targets)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    logger.info("🚀 Initializing LogGPT Count-Based Training (N=20)")
    config.set_seeds()
    
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token # Ensure pad token exists
    
    # 2. Dataset
    train_dataset, val_dataset = prepare_llm_dataset(tokenizer, block_size=BLOCK_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 3. Model
    # Adaptação para new vocab size (se buffer usado)
    # Mas tokenizer é pré-treinado fixo distilgpt2.
    # Usamos vocab original.
    
    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_layer=6,
        n_head=8,
        n_embd=512,
        dropout=DROPOUT
    )
    # Inject block_size for LogGPT model compatibility
    model_config.block_size = BLOCK_SIZE
    
    model = LogGPT(model_config)
    model.to(DEVICE)
    
    # Opt & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))
    
    logger.info(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Training Loop
    best_loss = float('inf')
    patience = 3
    counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE, epoch, scaler)
        val_loss = evaluate_epoch(model, val_loader, DEVICE)
        
        ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
        logger.info(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | PPL: {ppl:.2f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            logger.info(f"✅ Saving best model to {MODEL_DIR}")
            torch.save(model.state_dict(), f"{MODEL_DIR}/loggpt_weights.pt")
            torch.save(model_config, f"{MODEL_DIR}/config.pt")
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping")
                break
                
    logger.info("🎉 Training Done.")

if __name__ == "__main__":
    main()
