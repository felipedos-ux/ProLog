# -*- coding: utf-8 -*-
"""
DEFINITIVE test: load trained model + real HDFS data, check actual loss
Uses only a small subset for speed.
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import polars as pl
from datasets import Dataset

from model import LogGPT, GPTConfig
from config import (
    MODEL_NAME, BLOCK_SIZE, DEVICE, VOCAB_BUFFER, DROPOUT,
    DATA_FILE, TEMPLATE_COL, LABEL_COL, SESSION_ID_COL, TIMESTAMP_COL
)

print("=" * 60)
print("  TESTE DEFINITIVO: Loss real com dados HDFS reais")
print("=" * 60)

# 1. Load tokenizer
print("\n[1/4] Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

# 2. Load a SMALL amount of real HDFS data (first 50K rows)
print("\n[2/4] Carregando dados HDFS reais (primeiras 50K linhas)...")
df = pl.read_csv(str(DATA_FILE), infer_schema_length=10000, n_rows=50000)
normal = df.filter(
    (pl.col(LABEL_COL) == 0) &
    (pl.col(TEMPLATE_COL).is_not_null())
)
print(f"  Normal logs: {len(normal)}")

# Unique templates
templates = normal[TEMPLATE_COL].unique()
print(f"  Unique templates: {len(templates)}")
for i, t in enumerate(templates.to_list()[:5]):
    print(f"    {i}: {str(t)[:80]}...")

# Group by session
sessions = (
    normal.sort(TIMESTAMP_COL)
    .group_by(SESSION_ID_COL)
    .agg(pl.col(TEMPLATE_COL))
    .select(TEMPLATE_COL)
)

# Convert to text
text_sessions = []
for row in sessions.rows():
    session_text = " \n ".join(str(t) for t in row[0])
    text_sessions.append(session_text)

print(f"  Sessions: {len(text_sessions)}")
print(f"  Sample session: {text_sessions[0][:150]}...")

# Create dataset (first 100 sessions for speed)
subset = text_sessions[:100]
dataset = Dataset.from_dict({"text": subset})

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Chunk
def group_texts(examples):
    concat = {k: sum(examples[k], []) for k in examples.keys()}
    total = len(concat[list(examples.keys())[0]])
    total = (total // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i:i+BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)]
        for k, t in concat.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

chunked = tokenized.map(group_texts, batched=True, batch_size=1000)
print(f"  Chunked samples: {len(chunked)}")

# 3. Load trained model
print("\n[3/4] Carregando modelo treinado...")
config = GPTConfig(
    vocab_size=vocab_size + VOCAB_BUFFER,
    block_size=BLOCK_SIZE,
    n_layer=4, n_head=4, n_embd=256,
    dropout=DROPOUT
)
model = LogGPT(config)
state = torch.load("saved_models/hdfs_loggpt.pt", map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()
print("  Modelo carregado!")

# 4. Compute loss on real data
print("\n[4/4] Calculando loss em dados reais...")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
loader = DataLoader(chunked, batch_size=4, collate_fn=data_collator)

total_loss = 0.0
total_loss_shifted = 0.0
steps = 0
losses_list = []

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        logits, loss = model(input_ids, targets=labels)
        
        # Also compute shifted loss (proper causal LM)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = input_ids[:, 1:].contiguous()
        loss_shifted = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1)
        )
        
        total_loss += loss.item()
        total_loss_shifted += loss_shifted.item()
        losses_list.append(loss.item())
        steps += 1
        
        if steps <= 5:
            print(f"  Batch {steps}: loss={loss.item():.10f} shifted={loss_shifted.item():.10f}")
        
        if steps >= 50:
            break

avg_loss = total_loss / steps
avg_shifted = total_loss_shifted / steps
import numpy as np
losses_arr = np.array(losses_list)

print(f"\n{'='*60}")
print(f"  RESULTADOS ({steps} batches)")
print(f"{'='*60}")
print(f"  Avg Loss (model.py):  {avg_loss:.10f}  (.4f = {avg_loss:.4f})")
print(f"  Avg Loss (shifted):   {avg_shifted:.10f}  (.4f = {avg_shifted:.4f})")
print(f"  Min loss:  {losses_arr.min():.10f}")
print(f"  Max loss:  {losses_arr.max():.10f}")
print(f"  Std loss:  {losses_arr.std():.10f}")
print()

if avg_loss < 0.0001:
    print("  CONCLUSAO: Loss REALMENTE eh ~0 em dados de treino!")
    print("  O modelo MEMORIZOU os padroes do HDFS.")
    print("  Isso eh aceitavel para anomaly detection.")
    print("  NAO precisa retreinar - modelo esta OK!")
elif avg_loss < 0.01:
    print("  CONCLUSAO: Loss muito baixo mas nao zero.")
    print("  Modelo convergiu bem. Treinamento valido.")
else:
    print(f"  CONCLUSAO: Loss real = {avg_loss:.4f}")
    print(f"  Modelo parcialmente convergido.")
    print(f"  O .4f no log truncava um loss que era significativo.")
    print(f"  PROBLEMA: O early stopping pode ter sido enganado!")
