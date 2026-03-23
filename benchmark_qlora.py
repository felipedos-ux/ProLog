"""
LLM QLoRA Benchmark for Log Anomaly Detection (OpenStack).

Uses 4-bit quantization (bitsandbytes) and LoRA (peft) to fine-tune
pre-trained 7B-class models on normal OpenStack sessions.
Evaluates using Cross-Entropy threshold detection on the test set.

Models tested:
  1. Phi-3.5-mini    (Microsoft)
  2. Qwen2-7B        (Alibaba)
  3. Mistral-7B      (Mistral AI)
  (Llama and Gemma skipped as they are gated)

NOTE: Top-K detection was found to be invalid for OpenStack (TN=0, no
discrimination). Cross-Entropy with calibrated threshold is used instead.
"""

import sys, os, math, json, random, time, traceback
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import polars as pl

# ── Project paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "01_OpenStack_Validated"))
from dataset import load_openstack_data, prepare_session_strings
from config import (
    DATA_FILE, BLOCK_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    DEVICE, SEED, TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT, LOG_COLUMN,
    set_seeds
)

# Limit to 2 epochs for fine-tuning as LLMs adapt fast
QLORA_EPOCHS = 2
QLORA_BATCH_SIZE = 2  # Small batch size to avoid OOM on 12GB VRAM
GRAD_ACCUM_STEPS = 4  # Effective BS = 8
MAX_SEQ_LENGTH = 128  # OpenStack logs are short, original LogGPT used 64

# ═════════════════════════════════════════════════════════════════════════════
# CANDIDATES
# ═════════════════════════════════════════════════════════════════════════════
CANDIDATES = [
    {"name": "Phi-3.5-mini", "hf_id": "microsoft/Phi-3.5-mini-instruct"},
    {"name": "Qwen2-7B",     "hf_id": "Qwen/Qwen2-7B"},
    {"name": "Mistral-7B",   "hf_id": "mistralai/Mistral-7B-v0.3"},
]

# ═════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═════════════════════════════════════════════════════════════════════════════
def prepare_data():
    """Load and split OpenStack data. Returns train_seqs, val_sessions, test_sessions."""
    df = load_openstack_data()

    normal_sessions = prepare_session_strings(df, label_filter=0)
    session_texts = normal_sessions["EventTemplate"].to_list()
    normal_test_ids = normal_sessions["test_id"].to_list()

    random.seed(SEED)
    # Pair texts with ids before shuffling
    paired = list(zip(session_texts, normal_test_ids))
    random.shuffle(paired)
    session_texts, normal_test_ids = zip(*paired)
    session_texts, normal_test_ids = list(session_texts), list(normal_test_ids)

    n_train = int(len(session_texts) * 0.9)
    train_seqs = session_texts[:n_train]
    val_normal_seqs = session_texts[n_train:]

    all_sessions = prepare_session_strings(df)
    normal_ids = df.filter(pl.col("anom_label") == 0)["test_id"].unique().to_list()
    anom_ids = df.filter(pl.col("anom_label") == 1)["test_id"].unique().to_list()
    _, test_val_ids = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    _, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)
    test_ids_set = set(test_norm_ids + anom_ids)
    test_sessions = all_sessions.filter(pl.col("test_id").is_in(list(test_ids_set)))

    # Build calibration set: use val_normal_seqs (normal) + some anomaly sessions
    val_anom_sessions = all_sessions.filter(pl.col("label") == 1).head(50)
    val_norm_df = pl.DataFrame({
        "EventTemplate": val_normal_seqs,
        "label": [0] * len(val_normal_seqs),
        "test_id": [f"val_n_{i}" for i in range(len(val_normal_seqs))],
    })
    # Cast to consistent schema before concat
    schema = {"EventTemplate": pl.Utf8, "label": pl.Int64, "test_id": pl.Utf8}
    val_norm_sel = val_norm_df.select([
        pl.col("EventTemplate").cast(pl.Utf8),
        pl.col("label").cast(pl.Int64),
        pl.col("test_id").cast(pl.Utf8),
    ])
    val_anom_sel = val_anom_sessions.select([
        pl.col("EventTemplate").cast(pl.Utf8),
        pl.col("label").cast(pl.Int64),
        pl.col("test_id").cast(pl.Utf8),
    ])
    val_sessions = pl.concat([val_norm_sel, val_anom_sel])

    return train_seqs, val_sessions, test_sessions, len(test_norm_ids), len(anom_ids)


# ═════════════════════════════════════════════════════════════════════════════
# DATASET
# ═════════════════════════════════════════════════════════════════════════════
class LogSessionDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.examples = []
        for text in texts:
            ids = tokenizer.encode(text, truncation=True, max_length=block_size)
            if len(ids) >= 2:
                self.examples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class TestDataset(Dataset):
    def __init__(self, sessions_df, tokenizer, block_size):
        self.data = []
        for row in sessions_df.iter_rows(named=True):
            ids = tokenizer.encode(row["EventTemplate"], truncation=True,
                                   max_length=block_size)
            if len(ids) >= 2:
                self.data.append({
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "label": row["label"],
                    "test_id": row["test_id"],
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_train(batch, pad_id):
    max_len = max(len(x) for x in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
    return padded


def collate_test(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels, test_ids = [], []
    for i, x in enumerate(batch):
        l = len(x["input_ids"])
        padded[i, :l] = x["input_ids"]
        labels.append(x["label"])
        test_ids.append(x["test_id"])
    return {
        "input_ids": padded,
        "label": torch.tensor(labels, dtype=torch.long),
        "test_id": test_ids,
    }


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-ENTROPY DETECTION UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def compute_session_losses(model, dataset, pad_id, device, desc="Computing losses"):
    """Compute per-session average cross-entropy loss."""
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=lambda b: collate_test(b, pad_id),
    )
    losses, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)
            lbl = batch["label"].item()

            outputs = model(input_ids=input_ids)
            logits = outputs.logits

            # Compute per-token cross-entropy, ignore pad tokens
            inp = input_ids[:, :-1]
            tgt = input_ids[:, 1:]
            logits_shift = logits[:, :-1, :]

            # Calculate loss ignoring pad tokens
            loss = F.cross_entropy(
                logits_shift.reshape(-1, logits_shift.size(-1)),
                tgt.reshape(-1),
                ignore_index=pad_id,
                reduction="mean"
            )
            losses.append(loss.item())
            labels.append(lbl)

    return np.array(losses), np.array(labels)


def calibrate_threshold(losses, labels, n_steps=100):
    """Find the threshold that maximizes F1 on a validation set."""
    thresholds = np.linspace(
        np.percentile(losses, 5),
        np.percentile(losses, 95),
        n_steps
    )
    best_f1, best_thresh = 0, thresholds[0]

    for thresh in thresholds:
        preds = (losses > thresh).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh, best_f1


# ═════════════════════════════════════════════════════════════════════════════
# TRAIN + DETECT
# ═════════════════════════════════════════════════════════════════════════════
def run_qlora_benchmark(candidate, train_seqs, val_sessions, test_sessions, n_norm, n_anom):
    name = candidate["name"]
    hf_id = candidate["hf_id"]
    print(f"\n{'='*70}")
    print(f"  {name}  ({hf_id}) - QLoRA Fine-Tune")
    print(f"{'='*70}")

    free_memory()

    # 1. Load tokenizer
    print("  Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    except Exception as e:
        print(f"  ERROR loading tokenizer: {e}")
        return None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    pad_id = tokenizer.pad_token_id

    # 2. Loading Model with bnb 4-bit config
    print("  Loading 4-bit model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return None

    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False  # Critical for gradient checkpointing

    # 3. Apply PEFT/LoRA
    print("  Applying LoRA adapters...")
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)

    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(f"  Trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.2f}")

    # 4. Data loaders
    print(f"  Tokenizing train ({len(train_seqs)} sessions)...")
    train_ds = LogSessionDataset(train_seqs, tokenizer, MAX_SEQ_LENGTH)

    train_loader = DataLoader(
        train_ds, batch_size=QLORA_BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_train(b, pad_id)
    )

    # 5. Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    best_val_loss = float("inf")

    print(f"  Training {QLORA_EPOCHS} epochs...")
    t0 = time.time()

    for epoch in range(1, QLORA_EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            inp = batch[:, :-1].to(DEVICE)
            tgt = batch[:, 1:].to(DEVICE)

            outputs = model(input_ids=inp, labels=tgt)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            if (step + 1) % 20 == 0:
                print(f"      Step {step + 1}/{len(train_loader)} | Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f}")

        t_loss = total_loss / len(train_loader)
        t_ppl = math.exp(t_loss) if t_loss < 20 else float("inf")
        print(f"    Epoch {epoch:2d}/{QLORA_EPOCHS} | Train {t_loss:.4f} (PPL {t_ppl:.1f})")

    train_time = time.time() - t0
    print(f"  Training done in {train_time:.0f}s")

    # 6. Cross-Entropy Detection with Calibrated Threshold
    print("  Calibrating threshold on validation set...")

    # Build val dataset
    val_ds = TestDataset(val_sessions, tokenizer, MAX_SEQ_LENGTH)
    val_losses, val_labels = compute_session_losses(
        model, val_ds, pad_id, DEVICE, desc=f"  Calibrating ({name})"
    )

    threshold, val_f1 = calibrate_threshold(val_losses, val_labels)
    print(f"  Threshold: {threshold:.4f} (val F1={val_f1:.4f})")

    # 7. Apply threshold on test set
    print("  Running Cross-Entropy detection on test set...")
    test_ds = TestDataset(test_sessions, tokenizer, MAX_SEQ_LENGTH)
    test_losses, test_labels = compute_session_losses(
        model, test_ds, pad_id, DEVICE, desc=f"  Detecting ({name})"
    )

    y_pred = (test_losses > threshold).astype(int)
    y_true = test_labels.astype(int)

    # 8. Metrics
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"  Results: F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f}")
    print(f"           TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"           Threshold={threshold:.4f}")

    free_memory()

    return {
        "name": name,
        "hf_id": hf_id,
        "train_time_s": train_time,
        "best_val_loss": best_val_loss,
        "threshold": threshold,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    set_seeds()
    print(f"\n{'#'*70}")
    print(f"#  LLM QLORA BENCHMARK — OpenStack (Cross-Entropy)")
    print(f"#  Device: {DEVICE.upper()}")
    print(f"#  Training: {QLORA_EPOCHS} epochs, lr=2e-4, bs={QLORA_BATCH_SIZE}")
    print(f"{'#'*70}")

    print("\nLoading data...")
    train_seqs, val_sessions, test_sessions, n_norm, n_anom = prepare_data()
    print(f"  Train: {len(train_seqs)} | "
          f"Val: {len(val_sessions)} | "
          f"Test: {n_norm} norm + {n_anom} anom")

    results = []

    # Try recovering previous results to allow restarting
    out_path = ROOT / "benchmark_qlora_results.json"
    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                prev = json.load(f)
            # Only use results that have 'threshold' key (CE method)
            results = [r for r in prev if "threshold" in r]
            if results:
                print(f"Loaded {len(results)} previous CE results from {out_path}.")
        except Exception:
            pass

    completed_names = {r["name"] for r in results}

    for candidate in CANDIDATES:
        if candidate["name"] in completed_names:
            print(f"\nSkipping {candidate['name']} (already in results).")
            continue

        try:
            r = run_qlora_benchmark(candidate, train_seqs, val_sessions, test_sessions, n_norm, n_anom)
            if r:
                results.append(r)
                # Save incremental results
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
        except Exception as e:
            print(f"\n  FAILED: {candidate['name']}: {e}")
            traceback.print_exc()
            free_memory()

    # Summary
    print(f"\n{'='*90}")
    print(f"{'Model':<22} | {'F1':>6} | {'Prec':>6} | {'Rec':>6} | "
          f"{'TP':>4} | {'FP':>4} | {'FN':>4} | {'TN':>4} | {'Time':>6}")
    print(f"{'-'*90}")

    for r in results:
        print(f"{r['name']:<22} | "
              f"{r['f1']:>6.4f} | {r['precision']:>6.4f} | {r['recall']:>6.4f} | "
              f"{r['tp']:>4} | {r['fp']:>4} | {r['fn']:>4} | {r['tn']:>4} | "
              f"{r['train_time_s']:>5.0f}s")
    print(f"{'='*90}")

if __name__ == "__main__":
    main()
