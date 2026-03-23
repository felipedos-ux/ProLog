"""
QLoRA Benchmark + Lead Time for HDFS dataset.
Uses a small subset of the 1.7GB HDFS dataset for tractability.
Evaluates log-by-log CE detection + lead time (same as HDFS detect.py).
"""

import sys, os, math, json, random, time, traceback, gc
from pathlib import Path

import numpy as np
import pandas as pd
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

ROOT = Path(__file__).parent

# ── Constants ────────────────────────────────────────────────────────────────
QLORA_EPOCHS = 2
QLORA_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Dataset limits (keep small!)
MAX_TRAIN_SESSIONS = 500    # Normal sessions for training
MAX_TEST_NORM = 50          # Normal sessions for test
MAX_TEST_ANOM = 100         # Anomaly sessions for test
MAX_VAL_NORM = 30           # Normal sessions for validation
MAX_VAL_ANOM = 30           # Anomaly sessions for validation
SKIP_START_LOGS = 3         # Same as HDFS detect.py

DATA_FILE = ROOT / "data" / "HDFS" / "HDFS_data_processed.csv"

CANDIDATES = [
    {"name": "Phi-3.5-mini", "hf_id": "microsoft/Phi-3.5-mini-instruct"},
    {"name": "Qwen2-7B",     "hf_id": "Qwen/Qwen2-7B"},
    {"name": "Mistral-7B",   "hf_id": "mistralai/Mistral-7B-v0.3"},
]


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREP (with subsampling)
# ═════════════════════════════════════════════════════════════════════════════
def load_hdfs_subset():
    """Load HDFS data with aggressive subsampling for speed."""
    print(f"📦 Loading HDFS data from {DATA_FILE}...")
    print("  (This is a 1.7GB file, loading may take 1-2 minutes)")

    df = pl.read_csv(str(DATA_FILE), infer_schema_length=10000)
    print(f"  Total rows: {len(df):,}")

    # Get session IDs by label
    normal_ids = df.filter(pl.col("anom_label") == 0)["session_id"].unique().to_list()
    anom_ids = df.filter(pl.col("anom_label") == 1)["session_id"].unique().to_list()
    print(f"  Total sessions: {len(normal_ids):,} normal, {len(anom_ids):,} anomaly")

    # Subsample
    random.seed(SEED)
    random.shuffle(normal_ids)
    random.shuffle(anom_ids)

    train_ids = normal_ids[:MAX_TRAIN_SESSIONS]
    val_norm_ids = normal_ids[MAX_TRAIN_SESSIONS:MAX_TRAIN_SESSIONS + MAX_VAL_NORM]
    test_norm_ids = normal_ids[MAX_TRAIN_SESSIONS + MAX_VAL_NORM:
                               MAX_TRAIN_SESSIONS + MAX_VAL_NORM + MAX_TEST_NORM]
    val_anom_ids = anom_ids[:MAX_VAL_ANOM]
    test_anom_ids = anom_ids[MAX_VAL_ANOM:MAX_VAL_ANOM + MAX_TEST_ANOM]

    # Build session text sequences for training
    all_needed_ids = set(train_ids + val_norm_ids + test_norm_ids +
                         val_anom_ids + test_anom_ids)
    df_sub = df.filter(pl.col("session_id").is_in(list(all_needed_ids)))
    print(f"  Subset rows: {len(df_sub):,}")

    # Group by session → concat EventTemplates
    sessions = (
        df_sub.sort("timestamp")
        .group_by("session_id")
        .agg([
            pl.col("EventTemplate").str.concat(" \n "),
            pl.col("anom_label").max().alias("label"),
        ])
    )
    sessions_dict = {
        row["session_id"]: {
            "text": row["EventTemplate"],
            "label": row["label"],
        }
        for row in sessions.iter_rows(named=True)
    }

    # Build train_seqs (list of text strings, normal only)
    train_seqs = [sessions_dict[sid]["text"] for sid in train_ids if sid in sessions_dict]

    # Build val and test DataFrames
    val_records = []
    for sid in val_norm_ids + val_anom_ids:
        if sid in sessions_dict:
            val_records.append({
                "session_id": sid,
                "EventTemplate": sessions_dict[sid]["text"],
                "label": sessions_dict[sid]["label"],
            })
    val_df = pl.DataFrame(val_records)

    test_records = []
    for sid in test_norm_ids + test_anom_ids:
        if sid in sessions_dict:
            test_records.append({
                "session_id": sid,
                "EventTemplate": sessions_dict[sid]["text"],
                "label": sessions_dict[sid]["label"],
            })
    test_df = pl.DataFrame(test_records)

    print(f"  Train: {len(train_seqs)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"  Test: {sum(1 for r in test_records if r['label']==0)} norm + "
          f"{sum(1 for r in test_records if r['label']==1)} anom")

    return train_seqs, val_df, test_df, df_sub


# ═════════════════════════════════════════════════════════════════════════════
# DATASETS
# ═════════════════════════════════════════════════════════════════════════════
class LogSessionDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.examples = []
        for text in texts:
            ids = tokenizer.encode(text, truncation=True, max_length=block_size)
            if len(ids) >= 2:
                self.examples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]


def collate_train(batch, pad_id):
    max_len = max(len(x) for x in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
    return padded


# ═════════════════════════════════════════════════════════════════════════════
# SESSION-LEVEL CE DETECTION (for threshold calibration)
# ═════════════════════════════════════════════════════════════════════════════
def compute_session_ce(text, model, tokenizer, pad_id, device):
    """Compute average CE loss for a full session."""
    ids = tokenizer.encode(text, truncation=True, max_length=MAX_SEQ_LENGTH)
    if len(ids) < 2:
        return 0.0
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
    tgt = input_ids[:, 1:]
    logits_shift = logits[:, :-1, :]
    loss = F.cross_entropy(
        logits_shift.reshape(-1, logits_shift.size(-1)),
        tgt.reshape(-1), ignore_index=pad_id, reduction="mean"
    ).item()
    return loss


def calibrate_threshold(val_df, model, tokenizer, pad_id, device):
    """Find best CE threshold on validation set."""
    losses = []
    labels = []
    for row in val_df.iter_rows(named=True):
        loss = compute_session_ce(row["EventTemplate"], model, tokenizer, pad_id, device)
        losses.append(loss)
        labels.append(row["label"])

    best_f1 = 0
    best_thresh = 5.0
    for thresh in np.arange(min(losses), max(losses), 0.1):
        preds = [1 if l > thresh else 0 for l in losses]
        p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary",
                                                      zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh

    return best_thresh, best_f1


# ═════════════════════════════════════════════════════════════════════════════
# LOG-BY-LOG DETECTION WITH LEAD TIME (same logic as HDFS detect.py)
# ═════════════════════════════════════════════════════════════════════════════
def evaluate_session_logbylog(sid, label, session_df, model, tokenizer, threshold, device):
    """
    Evaluates session log-by-log (streaming) with CE threshold.
    Returns lead time = time from first alert to last log (failure marker).
    """
    session_df = session_df.sort("timestamp")
    templates = session_df["EventTemplate"].to_list()
    raw_ts = session_df["timestamp"].to_list()

    try:
        timestamps = [pd.to_datetime(ts) for ts in raw_ts]
    except Exception:
        return None

    failure_ts = timestamps[-1]

    is_detected = False
    first_alert_ts = None
    first_alert_loss = 0.0
    context_ids = []

    for i, current_log in enumerate(templates):
        if current_log is None:
            current_log = ""
        text = (" \n " if i > 0 else "") + str(current_log)
        new_ids = tokenizer.encode(text)

        if i < SKIP_START_LOGS:
            context_ids.extend(new_ids)
            if len(context_ids) > MAX_SEQ_LENGTH:
                context_ids = context_ids[-MAX_SEQ_LENGTH:]
            continue

        if i == 0:
            context_ids.extend(new_ids)
            continue

        full_seq = context_ids + new_ids
        if len(full_seq) > MAX_SEQ_LENGTH:
            input_seq = full_seq[-MAX_SEQ_LENGTH:]
            target_start_idx = len(input_seq) - len(new_ids)
        else:
            input_seq = full_seq
            target_start_idx = len(context_ids)

        x = torch.tensor(input_seq, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids=x)
            logits = outputs.logits

        target_indices = list(range(target_start_idx, len(input_seq)))
        logit_indices = [idx - 1 for idx in target_indices]

        if not logit_indices or logit_indices[0] < 0 or logit_indices[-1] >= logits.size(1):
            loss_val = 0.0
        else:
            relevant_logits = logits[0, logit_indices, :]
            relevant_targets = torch.tensor(
                input_seq[target_start_idx:], dtype=torch.long, device=device
            )
            if relevant_logits.shape[0] != relevant_targets.shape[0]:
                loss_val = 0.0
            else:
                loss_val = F.cross_entropy(relevant_logits, relevant_targets).item()

        if loss_val > threshold:
            is_detected = True
            first_alert_ts = timestamps[i]
            first_alert_loss = loss_val
            break

        context_ids.extend(new_ids)
        if len(context_ids) > MAX_SEQ_LENGTH:
            context_ids = context_ids[-MAX_SEQ_LENGTH:]

    result = {
        "session_id": sid,
        "is_detected": is_detected,
        "label": label,
        "lead_time_minutes": 0.0,
        "alert_loss": first_alert_loss,
    }

    if is_detected and first_alert_ts is not None:
        lead = (failure_ts - first_alert_ts).total_seconds() / 60
        result["lead_time_minutes"] = lead

    return result


# ═════════════════════════════════════════════════════════════════════════════
# MAIN: Train + Detect per model
# ═════════════════════════════════════════════════════════════════════════════
def run_model(candidate, train_seqs, val_df, test_df, df_sub):
    name = candidate["name"]
    hf_id = candidate["hf_id"]

    print(f"\n{'='*70}")
    print(f"  {name} — HDFS QLoRA + Lead Time")
    print(f"{'='*70}")

    free_memory()

    # 1. Tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    pad_id = tokenizer.pad_token_id

    # 2. Model
    print("  Loading 4-bit model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    # 3. LoRA
    print("  Applying LoRA adapters...")
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)
    tp_count, all_p = model.get_nb_trainable_parameters()
    print(f"  Trainable: {tp_count:,d} / {all_p:,d}")

    # 4. Train
    train_ds = LogSessionDataset(train_seqs, tokenizer, MAX_SEQ_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=QLORA_BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: collate_train(b, pad_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    print(f"  Training {QLORA_EPOCHS} epochs...")
    t0 = time.time()
    for epoch in range(1, QLORA_EPOCHS + 1):
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
                print(f"      Step {step+1}/{len(train_loader)} | Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f}")
        avg = total_loss / len(train_loader)
        ppl = math.exp(avg) if avg < 20 else float("inf")
        print(f"    Epoch {epoch}/{QLORA_EPOCHS} | Train {avg:.4f} (PPL {ppl:.1f})")
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.0f}s")

    # 5. Calibrate threshold on validation set
    print("  Calibrating threshold on validation set...")
    model.eval()
    threshold, val_f1 = calibrate_threshold(val_df, model, tokenizer, pad_id, DEVICE)
    print(f"  Threshold: {threshold:.4f} (val F1={val_f1:.4f})")

    # 6. Log-by-log detection with lead time on test set
    print(f"  Running log-by-log detection + lead time (threshold={threshold:.4f})...")

    test_ids = test_df["session_id"].to_list()
    test_labels = test_df["label"].to_list()

    results = []
    for i, (sid, label) in enumerate(tqdm(zip(test_ids, test_labels),
                                           total=len(test_ids),
                                           desc=f"  Detecting ({name})")):
        session_data = df_sub.filter(pl.col("session_id") == sid)
        if len(session_data) == 0:
            continue
        res = evaluate_session_logbylog(sid, label, session_data, model,
                                         tokenizer, threshold, DEVICE)
        if res:
            results.append(res)

    # Metrics
    tp = sum(1 for r in results if r["label"] == 1 and r["is_detected"])
    fn = sum(1 for r in results if r["label"] == 1 and not r["is_detected"])
    fp = sum(1 for r in results if r["label"] == 0 and r["is_detected"])
    tn = sum(1 for r in results if r["label"] == 0 and not r["is_detected"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Lead time analysis
    tp_results = [r for r in results if r["label"] == 1 and r["is_detected"]]
    lt_positive = [r for r in tp_results if r["lead_time_minutes"] > 0]
    lt_zero = [r for r in tp_results if r["lead_time_minutes"] == 0]
    lt_negative = [r for r in tp_results if r["lead_time_minutes"] < 0]

    avg_lt = np.mean([r["lead_time_minutes"] for r in lt_positive]) if lt_positive else 0
    max_lt = np.max([r["lead_time_minutes"] for r in lt_positive]) if lt_positive else 0
    med_lt = np.median([r["lead_time_minutes"] for r in lt_positive]) if lt_positive else 0

    print(f"\n  Results: F1={f1:.4f} | Prec={precision:.4f} | Rec={recall:.4f}")
    print(f"           TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"\n  📊 Lead Time ({name}):")
    print(f"     Total TPs: {len(tp_results)}")
    print(f"     ✅ Anticipated (>0 min): {len(lt_positive)} ({100*len(lt_positive)/max(1,len(tp_results)):.1f}%)")
    print(f"     ⏱  Simultaneous (=0):    {len(lt_zero)}")
    print(f"     ❌ Reactive (<0):         {len(lt_negative)}")
    if lt_positive:
        print(f"     Avg lead (>0): {avg_lt:.2f} min")
        print(f"     Max lead:      {max_lt:.2f} min")
        print(f"     Median lead:   {med_lt:.2f} min")

    # ── FREE VRAM COMPLETELY before next model ──
    del model, tokenizer, optimizer, train_ds, train_loader
    free_memory()
    vram = torch.cuda.memory_allocated() / 1024**2
    print(f"  🧹 VRAM after cleanup: {vram:.0f} MiB")

    return {
        "name": name,
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "train_time_s": round(train_time, 1),
        "threshold": round(threshold, 4),
        "total_tp": len(tp_results),
        "lt_anticipated": len(lt_positive),
        "lt_simultaneous": len(lt_zero),
        "lt_reactive": len(lt_negative),
        "lt_anticipated_pct": round(100.0 * len(lt_positive) / max(1, len(tp_results)), 1),
        "avg_leadtime_min": round(avg_lt, 2),
        "max_leadtime_min": round(max_lt, 2),
        "median_leadtime_min": round(med_lt, 2),
        "details": results,
    }


def main():
    set_seeds()
    print(f"\n{'#'*70}")
    print(f"#  LLM QLORA BENCHMARK + LEAD TIME — HDFS (Subset)")
    print(f"#  Device: {DEVICE.upper()}")
    print(f"#  Training: {QLORA_EPOCHS} epochs, lr=2e-4, bs={QLORA_BATCH_SIZE}")
    print(f"{'#'*70}")

    # Load data
    train_seqs, val_df, test_df, df_sub = load_hdfs_subset()

    # Run each model
    all_results = []
    out_path = ROOT / "benchmark_qlora_hdfs_results.json"

    # Pre-load results from previous run if available
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
        completed = {r["name"] for r in all_results}
        print(f"  Skipping already completed: {completed}")
    else:
        completed = set()

    for candidate in CANDIDATES:
        if candidate["name"] in completed:
            print(f"\n  ⏭ Skipping {candidate['name']} (already completed)")
            continue
        try:
            r = run_model(candidate, train_seqs, val_df, test_df, df_sub)
            summary = {k: v for k, v in r.items() if k != "details"}
            all_results.append(summary)

            # Save details per model
            detail_path = ROOT / f"hdfs_leadtime_{candidate['name'].replace(' ', '_').lower()}.json"
            with open(detail_path, "w") as f:
                json.dump(r["details"], f, indent=2, default=str)

            # Save incremental summary
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)

        except Exception as e:
            print(f"\n  FAILED: {candidate['name']}: {e}")
            traceback.print_exc()
            free_memory()

    # Summary table
    print(f"\n{'='*110}")
    print(f"{'Model':<18} | {'F1':>6} | {'TP':>4} | {'Anticipated':>12} | {'Reactive':>10} | "
          f"{'Avg LT':>8} | {'Max LT':>8} | {'Time':>6}")
    print(f"{'-'*110}")
    for r in all_results:
        print(f"{r['name']:<18} | {r['f1']:>6.4f} | {r['tp']:>4} | "
              f"{r['lt_anticipated']:>4} ({r['lt_anticipated_pct']:>5.1f}%) | "
              f"{r['lt_reactive']:>4}          | "
              f"{r['avg_leadtime_min']:>6.1f}m | {r['max_leadtime_min']:>6.1f}m | "
              f"{r['train_time_s']:>5.0f}s")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()
