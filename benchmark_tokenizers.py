"""
LLM Tokenizer Benchmark for Log Anomaly Detection (OpenStack).

Keeps the same LogGPT architecture (4 layers, 4 heads, 256 embd)
and swaps ONLY the tokenizer to measure the impact of tokenization
on anomaly detection performance.

Models tested:
  1. GPT-2           (baseline)
  2. Phi-3.5-mini    (Microsoft)
  3. Qwen2-7B        (Alibaba)
  4. Llama-3.1-8B    (Meta)
  5. Gemma-2-9B      (Google)
  6. Mistral-7B      (Mistral AI)
"""

import sys, os, math, json, random, time, traceback
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
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
    DEVICE, N_LAYER, N_HEAD, N_EMBD, DROPOUT, SEED,
    TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT, LOG_COLUMN,
    set_seeds
)

# ── Model (copy of LogGPT to avoid import issues) ───────────────────────────
class GPTConfig:
    def __init__(self, vocab_size, block_size=1024, n_layer=4, n_head=4,
                 n_embd=256, dropout=0.0):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer("bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                  .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class LogGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss


# ═════════════════════════════════════════════════════════════════════════════
# TOKENIZER CANDIDATES
# ═════════════════════════════════════════════════════════════════════════════
CANDIDATES = [
    {"name": "GPT-2 (baseline)",   "hf_id": "gpt2"},
    {"name": "Phi-3.5-mini",       "hf_id": "microsoft/Phi-3.5-mini-instruct"},
    {"name": "Qwen2-7B",           "hf_id": "Qwen/Qwen2-7B"},
    {"name": "Llama-3.1-8B",       "hf_id": "meta-llama/Llama-3.1-8B"},
    {"name": "Gemma-2-9B",         "hf_id": "google/gemma-2-9b"},
    {"name": "Mistral-7B",         "hf_id": "mistralai/Mistral-7B-v0.3"},
]

K = 5  # Top-K detection parameter

# Already completed results (skip re-training)
COMPLETED_RESULTS = {
    "GPT-2 (baseline)": {
        "name": "GPT-2 (baseline)", "hf_id": "gpt2",
        "vocab_size": 50257, "effective_vocab": 50257,
        "params_M": 29.2, "train_time_s": 57, "best_val_loss": 1.8871,
        "f1": 0.9235, "precision": 0.8579, "recall": 1.0000,
        "tp": 169, "fp": 28, "fn": 0, "tn": 0,
    },
    "Phi-3.5-mini": {
        "name": "Phi-3.5-mini", "hf_id": "microsoft/Phi-3.5-mini-instruct",
        "vocab_size": 32011, "effective_vocab": 32011,
        "params_M": 19.8, "train_time_s": 45, "best_val_loss": 2.4377,
        "f1": 0.9235, "precision": 0.8579, "recall": 1.0000,
        "tp": 169, "fp": 28, "fn": 0, "tn": 0,
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═════════════════════════════════════════════════════════════════════════════
def prepare_data():
    """Load and split OpenStack data (same as train_custom.py)."""
    df = load_openstack_data()

    # Normal sessions for training
    normal_sessions = prepare_session_strings(df, label_filter=0)
    session_texts = normal_sessions["EventTemplate"].to_list()

    random.seed(SEED)
    random.shuffle(session_texts)
    n_train = int(len(session_texts) * 0.9)
    train_seqs = session_texts[:n_train]
    val_seqs = session_texts[n_train:]

    # All sessions for test
    all_sessions = prepare_session_strings(df)
    normal_ids = df.filter(pl.col("anom_label") == 0)["test_id"].unique().to_list()
    anom_ids = df.filter(pl.col("anom_label") == 1)["test_id"].unique().to_list()
    _, test_val_ids = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    _, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)
    test_ids_set = set(test_norm_ids + anom_ids)
    test_sessions = all_sessions.filter(pl.col("test_id").is_in(list(test_ids_set)))

    return train_seqs, val_seqs, test_sessions, len(test_norm_ids), len(anom_ids)


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


# ═════════════════════════════════════════════════════════════════════════════
# TRAIN + DETECT
# ═════════════════════════════════════════════════════════════════════════════
def run_benchmark(candidate, train_seqs, val_seqs, test_sessions,
                  n_norm, n_anom, device):
    name = candidate["name"]
    hf_id = candidate["hf_id"]
    print(f"\n{'='*70}")
    print(f"  {name}  ({hf_id})")
    print(f"{'='*70}")

    # 1. Load tokenizer
    print(f"  Loading tokenizer...")
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
    vocab_size = len(tokenizer)

    # Cap vocab_size to avoid OOM on embedding layer
    # For very large vocabs (>150K), this would use too much VRAM
    MAX_VOCAB = 50257  # Same as GPT-2, avoids slow softmax on large vocabs
    if vocab_size > MAX_VOCAB:
        print(f"  WARNING: vocab_size={vocab_size} > {MAX_VOCAB}, capping to {MAX_VOCAB}")
        print(f"  (Tokens above this ID will be truncated)")
        effective_vocab = MAX_VOCAB
    else:
        effective_vocab = vocab_size

    # Clamp pad_id to effective vocab range
    if pad_id is not None and pad_id >= effective_vocab:
        pad_id = effective_vocab - 1  # Use last valid token as pad
        print(f"  WARNING: pad_id clamped to {pad_id}")

    print(f"  Vocab: {vocab_size} (effective: {effective_vocab}, pad_id: {pad_id})")

    # 2. Create model
    config = GPTConfig(
        vocab_size=effective_vocab,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
    )
    model = LogGPT(config)
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: {param_count:.1f}M params")

    # 3. Prepare data
    print(f"  Tokenizing train ({len(train_seqs)} sessions)...")
    train_ds = LogSessionDataset(train_seqs, tokenizer, BLOCK_SIZE)
    val_ds = LogSessionDataset(val_seqs, tokenizer, BLOCK_SIZE)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_train(b, pad_id)
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: collate_train(b, pad_id)
    )

    # 4. Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float("inf")
    best_state = None

    print(f"  Training {EPOCHS} epochs...")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            # Clamp token IDs to effective vocab
            batch = batch.clamp(max=effective_vocab - 1)
            inp = batch[:, :-1].to(device)
            tgt = batch[:, 1:].to(device)
            logits, loss = model(inp, targets=tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        t_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        v_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.clamp(max=effective_vocab - 1)
                inp = batch[:, :-1].to(device)
                tgt = batch[:, 1:].to(device)
                _, loss = model(inp, targets=tgt)
                v_total += loss.item()
        v_loss = v_total / len(val_loader)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        t_ppl = math.exp(t_loss) if t_loss < 20 else float("inf")
        v_ppl = math.exp(v_loss) if v_loss < 20 else float("inf")
        print(f"    Epoch {epoch:2d}/{EPOCHS} | "
              f"Train {t_loss:.4f} (PPL {t_ppl:.1f}) | "
              f"Val {v_loss:.4f} (PPL {v_ppl:.1f})"
              + (" *" if v_loss <= best_val_loss else ""))

    train_time = time.time() - t0
    print(f"  Training done in {train_time:.0f}s (best val loss: {best_val_loss:.4f})")

    # 5. Load best model and detect
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()

    print(f"  Running Top-{K} detection...")
    test_ds = TestDataset(test_sessions, tokenizer, BLOCK_SIZE)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: collate_test(b, pad_id)
    )

    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"  Detecting ({name})", leave=False):
            input_ids = batch["input_ids"].clamp(max=effective_vocab - 1).to(device)
            labels_batch = batch["label"].numpy()

            logits, _ = model(input_ids)
            targets = input_ids[:, 1:]
            preds = logits[:, :-1, :]

            probs = torch.softmax(preds, dim=-1)
            _, topk_inds = torch.topk(probs, K, dim=-1)
            matches = (topk_inds == targets.unsqueeze(-1)).any(dim=-1)
            target_mask = (targets != pad_id)
            valid_anomalies = (~matches) & target_mask
            is_anom = valid_anomalies.any(dim=1).cpu().numpy()

            for i in range(len(labels_batch)):
                y_true.append(int(labels_batch[i]))
                y_pred.append(1 if is_anom[i] else 0)

    # 6. Metrics
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"  Results: F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f}")
    print(f"           TP={tp} FP={fp} FN={fn} TN={tn}")

    # Cleanup GPU
    del model, optimizer, train_ds, val_ds, test_ds
    torch.cuda.empty_cache()

    return {
        "name": name,
        "hf_id": hf_id,
        "vocab_size": vocab_size,
        "effective_vocab": effective_vocab,
        "params_M": param_count,
        "train_time_s": train_time,
        "best_val_loss": best_val_loss,
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
    device = DEVICE
    print(f"\n{'#'*70}")
    print(f"#  LLM TOKENIZER BENCHMARK — OpenStack (Top-{K})")
    print(f"#  Device: {device.upper()}")
    print(f"#  Architecture: LogGPT ({N_LAYER}L, {N_HEAD}H, {N_EMBD}E)")
    print(f"#  Training: {EPOCHS} epochs, lr={LEARNING_RATE}, bs={BATCH_SIZE}")
    print(f"{'#'*70}")

    # Load data once
    print("\nLoading data...")
    train_seqs, val_seqs, test_sessions, n_norm, n_anom = prepare_data()
    print(f"  Train: {len(train_seqs)} | Val: {len(val_seqs)} | "
          f"Test: {n_norm} norm + {n_anom} anom")

    results = []
    for candidate in CANDIDATES:
        name = candidate["name"]
        # Use cached results if available
        if name in COMPLETED_RESULTS:
            print(f"\n  {name}: Using cached results (F1={COMPLETED_RESULTS[name]['f1']:.4f})")
            results.append(COMPLETED_RESULTS[name])
            continue
        try:
            r = run_benchmark(candidate, train_seqs, val_seqs, test_sessions,
                              n_norm, n_anom, device)
            if r:
                results.append(r)
        except Exception as e:
            print(f"\n  FAILED: {name}: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()

    # ── Summary Table ────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"{'Model':<22} | {'Vocab':>8} | {'Params':>7} | "
          f"{'F1':>6} | {'Prec':>6} | {'Rec':>6} | "
          f"{'TP':>4} | {'FP':>4} | {'FN':>4} | {'TN':>4} | {'Time':>6}")
    print(f"{'-'*90}")

    for r in results:
        print(f"{r['name']:<22} | {r['effective_vocab']:>8} | "
              f"{r['params_M']:>6.1f}M | "
              f"{r['f1']:>6.4f} | {r['precision']:>6.4f} | {r['recall']:>6.4f} | "
              f"{r['tp']:>4} | {r['fp']:>4} | {r['fn']:>4} | {r['tn']:>4} | "
              f"{r['train_time_s']:>5.0f}s")

    print(f"{'='*90}")

    # Best model
    if results:
        best = max(results, key=lambda x: x["f1"])
        print(f"\nBest: {best['name']} (F1={best['f1']:.4f})")

    # Save results
    out_path = ROOT / "benchmark_tokenizer_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
