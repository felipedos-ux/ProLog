# -*- coding: utf-8 -*-
"""
FAST diagnostic: why is training loss 0.0000?
Does NOT reload the full dataset - uses direct analysis.
"""
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from model import LogGPT, GPTConfig
from config import MODEL_NAME, BLOCK_SIZE, DEVICE, VOCAB_BUFFER, DROPOUT

print("=" * 60)
print("  DIAGNOSTICO RAPIDO: Loss = 0.0000")
print("=" * 60)

# 1. Load tokenizer
print("\n[1/5] Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size
print(f"  vocab_size={vocab_size}, pad_token_id={tokenizer.pad_token_id}")

# 2. Simulate a realistic HDFS log sample
print("\n[2/5] Criando amostra sintetica de logs HDFS...")
hdfs_text = "Receiving block blk_123 src: /10.0.0.1 dest: /10.0.0.2 \n PacketResponder 0 for block blk_123 terminating \n BLOCK* NameSystem.addStoredBlock: blockMap updated \n Receiving block blk_456 src: /10.0.0.3 dest: /10.0.0.4"

tokens = tokenizer(hdfs_text, return_tensors="pt")
input_ids = tokens["input_ids"]
print(f"  input_ids shape: {input_ids.shape}")
print(f"  input_ids: {input_ids[0][:30].tolist()}")

# Create a proper block of 128 tokens
if input_ids.shape[1] < BLOCK_SIZE:
    # Repeat to fill block
    repeats = (BLOCK_SIZE // input_ids.shape[1]) + 1
    input_ids = input_ids.repeat(1, repeats)[:, :BLOCK_SIZE]
else:
    input_ids = input_ids[:, :BLOCK_SIZE]

print(f"  Block input_ids shape: {input_ids.shape}")

# 3. Test DataCollator behavior
print("\n[3/5] Testando DataCollatorForLanguageModeling...")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Simulate what the collator does with a sample
sample = {"input_ids": input_ids[0].tolist(), "labels": input_ids[0].tolist()}
batch = data_collator([sample])

print(f"  Batch keys: {list(batch.keys())}")
print(f"  input_ids shape: {batch['input_ids'].shape}")
print(f"  labels shape: {batch['labels'].shape}")
print(f"  input_ids[0][:20]: {batch['input_ids'][0][:20].tolist()}")
print(f"  labels[0][:20]: {batch['labels'][0][:20].tolist()}")

total = batch['labels'].numel()
masked = (batch['labels'] == -100).sum().item()
print(f"  Labels totais: {total}")
print(f"  Labels mascarados (-100): {masked} ({masked/total*100:.1f}%)")
print(f"  Labels validos: {total-masked} ({(total-masked)/total*100:.1f}%)")

# 4. Test model forward with both scenarios
print("\n[4/5] Testando forward pass do modelo...")

config = GPTConfig(
    vocab_size=vocab_size + VOCAB_BUFFER,
    block_size=BLOCK_SIZE,
    n_layer=4, n_head=4, n_embd=256,
    dropout=DROPOUT
)
model = LogGPT(config)
model.to(DEVICE)
model.eval()

ids = batch['input_ids'].to(DEVICE)
labels = batch['labels'].to(DEVICE)

with torch.no_grad():
    logits, loss_model = model(ids, targets=labels)

print(f"  logits shape: {logits.shape}")
print(f"  Loss do model.py (targets=labels do collator): {loss_model.item():.10f}")

# 5. The KEY question: how does cross_entropy handle -100?
print("\n[5/5] ANALISE CRITICA: como cross_entropy lida com -100?")
print()

flat_logits = logits.view(-1, logits.size(-1))
flat_labels = labels.view(-1)

# A) SEM ignore_index (como esta no model.py)
loss_no_ignore = F.cross_entropy(flat_logits, flat_labels)
print(f"  A) cross_entropy SEM ignore_index: {loss_no_ignore.item():.10f}")

# B) COM ignore_index=-100
loss_with_ignore = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
print(f"  B) cross_entropy COM ignore_index=-100: {loss_with_ignore.item():.10f}")

# C) Com input_ids como targets (shifted)
shift_logits = logits[:, :-1, :].contiguous()
shift_targets = ids[:, 1:].contiguous()
loss_shifted = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_targets.view(-1)
)
print(f"  C) cross_entropy shifted (input_ids): {loss_shifted.item():.10f}")

# D) Com input_ids direto como targets (sem shift)
loss_direct = F.cross_entropy(
    flat_logits,
    ids.view(-1)
)
print(f"  D) cross_entropy direto (input_ids): {loss_direct.item():.10f}")

print()
print("=" * 60)
print("  RESULTADO")
print("=" * 60)
print()

# Check what -100 does as a target for cross_entropy
print("  EXPLICACAO TECNICA:")
print(f"  - PyTorch cross_entropy: ignore_index default = -100")
print(f"  - Se target=-100, o loss para esse token eh IGNORADO")
print()

if masked > 0:
    print(f"  O DataCollator mascara {masked}/{total} labels como -100")
    print()
    
    if masked == total:
        print("  >>> CAUSA RAIZ: TODOS os labels sao -100!")
        print("  >>> cross_entropy com ignore_index=-100 (default) retorna 0/NaN")
    else:
        print(f"  >>> {masked} labels sao -100, {total-masked} sao validos")
    
    print()
    print("  POREM: cross_entropy DEFAULT ignore_index = -100")
    print("  Entao mesmo SEM especificar ignore_index, PyTorch JA ignora -100!")
    print()
    print(f"  Loss A (sem ignore explicito): {loss_no_ignore.item():.10f}")
    print(f"  Loss B (com ignore=-100):      {loss_with_ignore.item():.10f}")
    print()
    
    if abs(loss_no_ignore.item() - loss_with_ignore.item()) < 1e-6:
        print("  >>> CONFIRMADO: Loss A == Loss B")
        print("  >>> PyTorch JA ignora -100 por default no cross_entropy!")
        print("  >>> O problema NAO e o ignore_index")
    else:
        print("  >>> Loss A != Loss B - comportamento inesperado")

print()
print("  COMPARACAO DE LOSSES:")
print(f"  Model.py (atual):     {loss_no_ignore.item():.10f}")
print(f"  Com ignore_index:     {loss_with_ignore.item():.10f}")
print(f"  Shifted (correto LM): {loss_shifted.item():.10f}")
print(f"  Direto (sem shift):   {loss_direct.item():.10f}")
print()

# Check if ALL losses are near zero
all_near_zero = all(x < 0.001 for x in [
    loss_no_ignore.item(), loss_with_ignore.item(),
    loss_shifted.item(), loss_direct.item()
])

if all_near_zero:
    print("  !!! TODOS os losses sao ~0 - modelo random nao deveria dar isso")
    print("  !!! Verificar se vocab esta correto e tokens sao validos")
else:
    print(f"  Loss shifted (correto): {loss_shifted.item():.4f}")
    print(f"  Loss model.py (atual):  {loss_no_ignore.item():.4f}")
    if loss_no_ignore.item() < 0.001 and loss_shifted.item() > 0.1:
        print()
        print("  >>> BUG CONFIRMADO!")
        print("  >>> model.py usa targets=labels do DataCollator")
        print("  >>> DataCollator mascara labels com -100")
        print("  >>> cross_entropy ignora -100 por default")
        print("  >>> Resultado: loss = 0 (nenhum token para calcular!)")
        print()
        print("  CORRECAO: Nao usar DataCollator labels,")
        print("  usar input_ids shifted como targets.")

print()
print("  FIM DO DIAGNOSTICO")
