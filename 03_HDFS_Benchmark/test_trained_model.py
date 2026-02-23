# -*- coding: utf-8 -*-
"""
Test the TRAINED model loss - is it really near zero?
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import LogGPT, GPTConfig
from config import MODEL_NAME, BLOCK_SIZE, DEVICE, VOCAB_BUFFER, DROPOUT

print("=" * 60)
print("  TESTE: Loss do modelo TREINADO")
print("=" * 60)

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

# 2. Create model and load trained weights
config = GPTConfig(
    vocab_size=vocab_size + VOCAB_BUFFER,
    block_size=BLOCK_SIZE,
    n_layer=4, n_head=4, n_embd=256,
    dropout=DROPOUT
)
model = LogGPT(config)

# Load saved weights
weights_path = "saved_models/hdfs_loggpt.pt"
state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()
print(f"\n  Modelo carregado: {weights_path}")

# 3. Test with HDFS-like text
hdfs_text = "Receiving block blk_123 src: /10.0.0.1 dest: /10.0.0.2 \n PacketResponder 0 for block blk_123 terminating \n BLOCK* NameSystem.addStoredBlock: blockMap updated \n Receiving block blk_456 src: /10.0.0.3 dest: /10.0.0.4"

tokens = tokenizer(hdfs_text, return_tensors="pt")
input_ids = tokens["input_ids"]

# Repeat to fill block
if input_ids.shape[1] < BLOCK_SIZE:
    repeats = (BLOCK_SIZE // input_ids.shape[1]) + 1
    input_ids = input_ids.repeat(1, repeats)[:, :BLOCK_SIZE]
else:
    input_ids = input_ids[:, :BLOCK_SIZE]

input_ids = input_ids.to(DEVICE)
labels = input_ids.clone()

with torch.no_grad():
    logits, loss = model(input_ids, targets=labels)

print(f"\n  Loss com 4 decimais (.4f):  {loss.item():.4f}")
print(f"  Loss com 10 decimais (.10f): {loss.item():.10f}")
print(f"  Loss com 20 decimais (.20f): {loss.item():.20f}")
print(f"  Loss raw: {loss.item()}")

# 4. Check probability distribution
probs = F.softmax(logits[0, 0], dim=-1)
top5 = torch.topk(probs, 5)
print(f"\n  Top 5 probabilidades para token 0:")
for i in range(5):
    tok_id = top5.indices[i].item()
    prob = top5.values[i].item()
    tok = tokenizer.decode([tok_id])
    print(f"    {tok_id:>6} ({tok!r:>10}): {prob:.6f}")

# 5. Check some losses per position
print(f"\n  Losses por posicao (primeiros 20 tokens):")
for pos in range(min(20, BLOCK_SIZE)):
    pos_logits = logits[0, pos]
    pos_target = labels[0, pos]
    pos_loss = F.cross_entropy(pos_logits.unsqueeze(0), pos_target.unsqueeze(0))
    pred_token = logits[0, pos].argmax().item()
    actual_token = labels[0, pos].item()
    correct = "OK" if pred_token == actual_token else "X "
    print(f"    pos {pos:>3}: loss={pos_loss.item():.6f} pred={pred_token:>6} actual={actual_token:>6} [{correct}]")

print()
print("=" * 60)
print("  CONCLUSAO")
print("=" * 60)
print()

if loss.item() < 0.0001:
    print("  Loss < 0.0001 CONFIRMADO!")
    print("  O modelo REALMENTE convergiu para loss proximo de zero.")
    print("  HDFS tem apenas ~30 templates unicos e sequencias muito repetitivas.")
    print("  Com 558K sessoes e modelo de 30M parametros, memorizar eh facil.")
    print()
    print("  SIGNIFICADO:")
    print("  - O treinamento NAO foi em vao!")
    print("  - O modelo APRENDEU os padroes normais perfeitamente")
    print("  - Para ANOMALY DETECTION, isso eh BOM:")
    print("    sequences anomalas terao loss MUITO MAIOR")
    print()
    print("  PROXIMO PASSO:")
    print("  - Pode prosseguir para calibracao e deteccao")
    print("  - O modelo salvo na epoca 7 eh valido")
elif loss.item() < 0.01:
    print("  Loss entre 0.0001 e 0.01 - modelo convergiu muito bem")
    print("  Normal para dataset repetitivo como HDFS")
else:
    print(f"  Loss = {loss.item():.6f} - modelo nao convergiu tanto")
    print("  Precisa investigar mais")
