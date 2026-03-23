"""
Lead Time calculation for QLoRA fine-tuned models.

For each model:
1. Load the pre-trained LLM with 4-bit quantization
2. Apply QLoRA adapters (re-train if needed)
3. For True Positives: find the first position in the session where
   the token is NOT in Top-K predictions
4. Map that position to a real timestamp
5. Compute lead_time = error_timestamp - alert_timestamp
6. Report: how many had lead_time > 0 (anticipated) vs <= 0 (reactive)

Uses the SAME training + detection config as benchmark_qlora.py.
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
sys.path.insert(0, str(ROOT / "01_OpenStack_Validated"))
from dataset import load_openstack_data, prepare_session_strings
from config import (
    DATA_FILE, BLOCK_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    DEVICE, SEED, TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT, LOG_COLUMN,
    set_seeds
)

# ── Same constants as benchmark_qlora.py ─────────────────────────────────────
QLORA_EPOCHS = 2
QLORA_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LENGTH = 128
K = 5  # Top-K for position-level anomaly detection (lead time calculation)

CANDIDATES = [
    {"name": "Phi-3.5-mini", "hf_id": "microsoft/Phi-3.5-mini-instruct"},
    {"name": "Qwen2-7B",     "hf_id": "Qwen/Qwen2-7B"},
    {"name": "Mistral-7B",   "hf_id": "mistralai/Mistral-7B-v0.3"},
]

# Load previous CE thresholds from benchmark results
THRESHOLDS = {}
results_path = ROOT / "benchmark_qlora_results.json"
if results_path.exists():
    with open(results_path) as f:
        for r in json.load(f):
            THRESHOLDS[r["name"]] = r.get("threshold", 8.0)


# ═════════════════════════════════════════════════════════════════════════════
# DATA
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


class TestDatasetWithMeta(Dataset):
    """Test dataset that also stores test_id and event_template for lead time."""
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
                    "event_template": row["EventTemplate"],
                    "seq_len": len(ids),
                })
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


def collate_train(batch, pad_id):
    max_len = max(len(x) for x in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
    return padded


def collate_test_meta(batch, pad_id):
    max_len = max(x["seq_len"] for x in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels, test_ids, event_templates = [], [], []
    for i, x in enumerate(batch):
        l = x["seq_len"]
        padded[i, :l] = x["input_ids"]
        labels.append(x["label"])
        test_ids.append(x["test_id"])
        event_templates.append(x["event_template"])
    return {
        "input_ids": padded,
        "label": torch.tensor(labels, dtype=torch.long),
        "test_id": test_ids,
        "event_template": event_templates,
    }


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ═════════════════════════════════════════════════════════════════════════════
def run_leadtime(candidate, train_seqs, test_sessions, error_info_dict,
                 session_timestamps_dict, ce_threshold):
    name = candidate["name"]
    hf_id = candidate["hf_id"]
    print(f"\n{'='*70}")
    print(f"  {name} — Lead Time Analysis")
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
    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(f"  Trainable: {trainable_params:,d} / {all_param:,d}")

    # 4. Training (same as benchmark)
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
            if (step + 1) % 40 == 0:
                print(f"      Step {step+1}/{len(train_loader)} | Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f}")
        t_loss = total_loss / len(train_loader)
        t_ppl = math.exp(t_loss) if t_loss < 20 else float("inf")
        print(f"    Epoch {epoch}/{QLORA_EPOCHS} | Train {t_loss:.4f} (PPL {t_ppl:.1f})")
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.0f}s")

    # 5. Detection with Lead Time
    print(f"  Running detection + lead time (CE threshold={ce_threshold:.4f})...")
    model.eval()

    test_ds = TestDatasetWithMeta(test_sessions, tokenizer, MAX_SEQ_LENGTH)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=lambda b: collate_test_meta(b, pad_id))

    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"  Detecting ({name})", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            true_label = batch["label"].item()
            tid = batch["test_id"][0]
            event_template = batch["event_template"][0]

            outputs = model(input_ids=input_ids)
            logits = outputs.logits

            # ── Session-level CE loss (for anomaly classification) ──
            tgt = input_ids[:, 1:]
            logits_shift = logits[:, :-1, :]
            ce_loss = F.cross_entropy(
                logits_shift.reshape(-1, logits_shift.size(-1)),
                tgt.reshape(-1), ignore_index=pad_id, reduction="mean"
            ).item()

            pred_label = 1 if ce_loss > ce_threshold else 0

            # ── Token-level Top-K (for lead time position) ──
            first_anomaly_step = -1
            if pred_label == 1:
                probs = torch.softmax(logits_shift, dim=-1)
                _, topk_inds = torch.topk(probs, K, dim=-1)
                targets = input_ids[:, 1:]
                matches = (topk_inds == targets.unsqueeze(-1)).any(dim=-1)
                target_mask = (targets != pad_id)
                valid_anomalies = (~matches) & target_mask

                if valid_anomalies.any():
                    first_anomaly_step = int(valid_anomalies.int().argmax(dim=1).item())

            # ── Lead Time ──
            err_info = error_info_dict.get(tid, {})
            first_error_timestamp = err_info.get("first_error_timestamp", None)
            first_error_index = err_info.get("first_error_index", None)
            lead_time_seconds = None
            lead_time_minutes = None
            alert_step_before_error = None

            if pred_label == 1 and first_anomaly_step >= 0 and first_error_timestamp is not None:
                session_ts = session_timestamps_dict.get(tid, [])
                if first_anomaly_step < len(session_ts):
                    alert_ts = pd.to_datetime(session_ts[first_anomaly_step])
                    error_ts = pd.to_datetime(first_error_timestamp)
                    delta = (error_ts - alert_ts).total_seconds()
                    lead_time_seconds = delta
                    lead_time_minutes = delta / 60.0

                if first_error_index is not None:
                    alert_step_before_error = int(first_error_index) - first_anomaly_step

            results.append({
                "test_id": tid,
                "label": int(true_label),
                "predicted": pred_label,
                "ce_loss": ce_loss,
                "first_anomaly_step": first_anomaly_step,
                "lead_time_seconds": lead_time_seconds,
                "lead_time_minutes": lead_time_minutes,
                "alert_step_before_error": alert_step_before_error,
            })

    # ── Metrics ──
    y_true = [r["label"] for r in results]
    y_pred = [r["predicted"] for r in results]
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # ── Lead Time Analysis ──
    tp_results = [r for r in results if r["label"] == 1 and r["predicted"] == 1]
    lt_available = [r for r in tp_results if r["lead_time_seconds"] is not None]
    lt_positive = [r for r in lt_available if r["lead_time_seconds"] > 0]
    lt_zero = [r for r in lt_available if r["lead_time_seconds"] == 0]
    lt_negative = [r for r in lt_available if r["lead_time_seconds"] < 0]

    avg_lt = np.mean([r["lead_time_seconds"] for r in lt_available]) if lt_available else 0
    avg_lt_pos = np.mean([r["lead_time_seconds"] for r in lt_positive]) if lt_positive else 0

    print(f"\n  Results: F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f}")
    print(f"           TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"\n  📊 Lead Time Analysis ({name}):")
    print(f"     Total TPs: {len(tp_results)}")
    print(f"     With lead time data: {len(lt_available)}")
    print(f"     ✅ Anticipated (>0): {len(lt_positive)} ({100*len(lt_positive)/max(1,len(lt_available)):.1f}%)")
    print(f"     ⏱  Simultaneous (=0): {len(lt_zero)}")
    print(f"     ❌ Reactive (<0):     {len(lt_negative)} ({100*len(lt_negative)/max(1,len(lt_available)):.1f}%)")
    print(f"     Avg lead time (all): {avg_lt:.1f}s ({avg_lt/60:.1f}min)")
    print(f"     Avg lead time (>0):  {avg_lt_pos:.1f}s ({avg_lt_pos/60:.1f}min)")

    free_memory()

    return {
        "name": name,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "train_time_s": train_time,
        "threshold": ce_threshold,
        "total_tp": len(tp_results),
        "lt_available": len(lt_available),
        "lt_anticipated": len(lt_positive),
        "lt_simultaneous": len(lt_zero),
        "lt_reactive": len(lt_negative),
        "lt_anticipated_pct": round(100 * len(lt_positive) / max(1, len(lt_available)), 1),
        "avg_leadtime_s": round(avg_lt, 1),
        "avg_leadtime_anticipated_s": round(avg_lt_pos, 1),
        "details": results,
    }


def main():
    set_seeds()
    print(f"\n{'#'*70}")
    print(f"#  LLM QLORA LEAD TIME ANALYSIS — OpenStack")
    print(f"#  Device: {DEVICE.upper()}")
    print(f"{'#'*70}")

    # ── Load data ──
    print("\nLoading data...")
    df = load_openstack_data()

    normal_sessions = prepare_session_strings(df, label_filter=0)
    session_texts = normal_sessions["EventTemplate"].to_list()
    random.seed(SEED)
    random.shuffle(session_texts)
    n_train = int(len(session_texts) * 0.9)
    train_seqs = session_texts[:n_train]

    all_sessions = prepare_session_strings(df)
    normal_ids = df.filter(pl.col("anom_label") == 0)["test_id"].unique().to_list()
    anom_ids = df.filter(pl.col("anom_label") == 1)["test_id"].unique().to_list()
    _, test_val_ids = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    _, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)
    test_ids_set = set(test_norm_ids + anom_ids)
    test_sessions = all_sessions.filter(pl.col("test_id").is_in(list(test_ids_set)))

    # ── Error info for lead time ──
    error_info = (
        df.sort("timestamp")
        .group_by("test_id")
        .agg([
            pl.col("anom_label").max().alias("is_anomaly"),
            pl.col("EventTemplate").filter(pl.col("anom_label") == 1).first().alias("first_error_template"),
            pl.col("timestamp").filter(pl.col("anom_label") == 1).first().alias("first_error_timestamp"),
            pl.col("EventId").filter(pl.col("anom_label") == 1).first().alias("first_error_eventid"),
            pl.col("anom_label").cum_sum().eq(1).arg_true().first().alias("first_error_index"),
        ])
    )
    error_info_dict = {row["test_id"]: row for row in error_info.iter_rows(named=True)}

    session_timestamps_dict = {}
    for tid, grp in df.sort("timestamp").group_by("test_id", maintain_order=True):
        session_timestamps_dict[tid] = grp["timestamp"].to_list()

    print(f"  Train: {len(train_seqs)} | Test: {len(test_norm_ids)} norm + {len(anom_ids)} anom")

    # ── Run each model ──
    all_results = []
    out_path = ROOT / "benchmark_qlora_leadtime.json"

    for candidate in CANDIDATES:
        ce_thresh = THRESHOLDS.get(candidate["name"], 8.0)
        try:
            r = run_leadtime(candidate, train_seqs, test_sessions,
                            error_info_dict, session_timestamps_dict, ce_thresh)
            # Save without per-session details to keep file small
            summary = {k: v for k, v in r.items() if k != "details"}
            all_results.append(summary)

            # Save details per model
            detail_path = ROOT / f"leadtime_details_{candidate['name'].replace(' ', '_').lower()}.json"
            with open(detail_path, "w") as f:
                json.dump(r["details"], f, indent=2, default=str)
            print(f"  Saved details to {detail_path}")

            # Save incremental summary
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)

        except Exception as e:
            print(f"\n  FAILED: {candidate['name']}: {e}")
            traceback.print_exc()
            free_memory()

    # ── Summary ──
    print(f"\n{'='*100}")
    print(f"{'Model':<18} | {'F1':>6} | {'TP':>4} | {'Anticipated':>12} | {'Reactive':>10} | {'Avg LT (>0)':>12} | {'Time':>6}")
    print(f"{'-'*100}")
    for r in all_results:
        print(f"{r['name']:<18} | {r['f1']:>6.4f} | {r['tp']:>4} | "
              f"{r['lt_anticipated']:>4} ({r['lt_anticipated_pct']:>5.1f}%) | "
              f"{r['lt_reactive']:>4} ({100-r['lt_anticipated_pct']:>5.1f}%) | "
              f"{r['avg_leadtime_anticipated_s']:>8.0f}s     | {r['train_time_s']:>5.0f}s")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
