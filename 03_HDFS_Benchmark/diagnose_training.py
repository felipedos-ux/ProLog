# -*- coding: utf-8 -*-
"""
Diagnostic script to identify why training loss is 0.0000
Checks: dataset content, model output, loss computation, label format
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

from dataset import prepare_llm_dataset
from model import LogGPT, GPTConfig
from config import (
    MODEL_NAME, MODEL_DIR, BLOCK_SIZE, BATCH_SIZE, DEVICE,
    VOCAB_BUFFER, DROPOUT, SEED, set_seeds
)

set_seeds()


def main():
    print("=" * 60)
    print("  DIAGNOSTICO: Por que Loss = 0.0000?")
    print("=" * 60)
    print()

    # 1. Load tokenizer
    print("[1/6] Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    print("  Vocab size:", vocab_size)
    print("  Pad token id:", tokenizer.pad_token_id)
    print("  EOS token id:", tokenizer.eos_token_id)
    print()

    # 2. Prepare dataset
    print("[2/6] Preparando dataset...")
    lm_datasets = prepare_llm_dataset(tokenizer, block_size=BLOCK_SIZE)

    if isinstance(lm_datasets, dict):
        train_dataset = lm_datasets["train"]
        val_dataset = lm_datasets.get("test") or lm_datasets.get("validation")
    else:
        split = lm_datasets.train_test_split(test_size=0.1, seed=SEED)
        train_dataset = split["train"]
        val_dataset = split["test"]

    print("  Train samples:", len(train_dataset))
    print("  Val samples:", len(val_dataset))
    print("  Columns:", train_dataset.column_names)
    print()

    # 3. Inspect raw dataset samples
    print("[3/6] Inspecionando amostras do dataset...")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        input_ids = sample["input_ids"]
        labels = sample.get("labels", None)

        print(f"\n  --- Amostra {i} ---")
        print(f"  input_ids length: {len(input_ids)}")
        print(f"  input_ids[:20]: {input_ids[:20]}")
        print(f"  input_ids unique tokens: {len(set(input_ids))}")

        if labels is not None:
            print(f"  labels length: {len(labels)}")
            print(f"  labels[:20]: {labels[:20]}")
            are_equal = (input_ids == labels)
            print(f"  labels == input_ids: {are_equal}")
            num_masked = sum(1 for l in labels if l == -100)
            print(f"  labels com -100: {num_masked}/{len(labels)}")
        else:
            print("  labels: NONE (nao existe no dataset)")

        # Decode text sample
        decoded = tokenizer.decode(input_ids[:50])
        print(f"  Decoded text[:100]: {repr(decoded[:100])}")
    print()

    # 4. Test DataCollator
    print("[4/6] Testando DataCollator...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Get 2 samples and collate
    samples = [train_dataset[i] for i in range(2)]
    batch = data_collator(samples)

    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  input_ids[0][:20]: {batch['input_ids'][0][:20].tolist()}")
    print(f"  labels[0][:20]: {batch['labels'][0][:20].tolist()}")

    # Check how many labels are -100
    total_labels = batch['labels'].numel()
    masked_labels = (batch['labels'] == -100).sum().item()
    valid_labels = total_labels - masked_labels
    print(f"  Total labels: {total_labels}")
    print(f"  Labels mascarados (-100): {masked_labels} ({masked_labels/total_labels*100:.1f}%)")
    print(f"  Labels validos: {valid_labels} ({valid_labels/total_labels*100:.1f}%)")

    if masked_labels == total_labels:
        print()
        print("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("  !!! PROBLEMA ENCONTRADO: TODOS LABELS = -100 !!!")
        print("  !!! cross_entropy ignora -100, loss = 0      !!!")
        print("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print()
    print()

    # 5. Test model forward pass
    print("[5/6] Testando forward pass do modelo...")

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
    model.eval()

    input_ids = batch['input_ids'].to(DEVICE)
    labels = batch['labels'].to(DEVICE)

    with torch.no_grad():
        logits, loss = model(input_ids, targets=labels)

    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss (model): {loss}")
    print(f"  Loss (model) .4f: {loss.item():.4f}")
    print(f"  Loss (model) .10f: {loss.item():.10f}")
    print()

    # 6. Manual loss computation
    print("[6/6] Calculando loss MANUALMENTE...")

    # Method A: Same as model.py (targets = labels from collator)
    flat_logits = logits.view(-1, logits.size(-1))
    flat_labels = labels.view(-1)
    loss_a = F.cross_entropy(flat_logits, flat_labels)
    print(f"  Loss A (labels do collator): {loss_a.item():.10f}")

    # Method B: ignore -100
    loss_b = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
    print(f"  Loss B (ignore -100):        {loss_b.item():.10f}")

    # Method C: Use input_ids shifted as targets (proper causal LM)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = input_ids[:, 1:].contiguous()
    loss_c = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1)
    )
    print(f"  Loss C (shifted input_ids):  {loss_c.item():.10f}")

    print()
    print("=" * 60)
    print("  DIAGNOSTICO COMPLETO")
    print("=" * 60)
    print()

    # Summary
    if masked_labels == total_labels:
        print("  CAUSA RAIZ: DataCollator mascara TODOS os labels como -100")
        print("  EFEITO: cross_entropy retorna 0 (nenhum token para computar loss)")
        print()
        print("  SOLUCAO: O model.py usa F.cross_entropy SEM ignore_index=-100,")
        print("  o que faz -100 ser tratado como token ID valido (fora do vocab).")
        print("  Isso causa loss = 0 ou comportamento indefinido.")
        print()
        print("  CORRECAO NECESSARIA:")
        print("  No model.py, mudar:")
        print("    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))")
        print("  Para:")
        print("    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)")
    elif loss_a.item() < 0.001:
        print("  CAUSA RAIZ: Loss genuinamente muito baixo")
        print("  Modelo pode estar memorizando dataset")
    else:
        print("  Loss A: {:.10f}".format(loss_a.item()))
        print("  Loss B: {:.10f}".format(loss_b.item()))
        print("  Loss C: {:.10f}".format(loss_c.item()))
        print("  Verificar qual valor eh o correto")

    print()
    print("  FIM DO DIAGNOSTICO")


if __name__ == "__main__":
    main()
