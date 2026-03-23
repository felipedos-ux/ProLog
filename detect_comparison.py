"""
detect_comparison.py — Top-K vs Cross-Entropy em OpenStack e HDFS
==================================================================
Roda AMBOS os métodos de detecção nos DOIS datasets usando os modelos
já treinados, e gera uma tabela comparativa de F1, Precision, Recall
e Lead Time.

Inferência: batch (sessão completa de uma vez) para ambos os métodos,
eliminando a variável do modo de inferência.
"""

import os, sys, json, math, hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
import polars as pl
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

ROOT = Path(__file__).parent

# ── Import OpenStack modules ──────────────────────────────────────────────────
sys.path.insert(0, str(ROOT / "01_OpenStack_Validated"))
from model import LogGPT, GPTConfig
import config as os_cfg
from dataset import load_openstack_data, prepare_session_strings
from torch.utils.data import Dataset, DataLoader

# ── Import HDFS config via importlib to avoid module cache collision ──────────
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("hdfs_cfg", str(ROOT / "03_HDFS_Benchmark" / "config.py"))
hdfs_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(hdfs_cfg)

# ─── Utilities ────────────────────────────────────────────────────────────────
PAD_TOKEN_ID = 50256  # GPT2 EOS

def metrics(labels, preds):
    """Returns precision, recall, f1."""
    tp = sum(l == 1 and p == 1 for l, p in zip(labels, preds))
    fp = sum(l == 0 and p == 1 for l, p in zip(labels, preds))
    fn = sum(l == 1 and p == 0 for l, p in zip(labels, preds))
    tn = sum(l == 0 and p == 0 for l, p in zip(labels, preds))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, tp, fp, fn, tn


def calibrate_threshold_on_val(model, val_ids, val_labels, session_texts_dict,
                                 tokenizer, block_size, device, n_steps=50):
    """
    Finds the Cross-Entropy threshold that maximises F1 on the validation set.
    val_ids: list of session IDs, val_labels: list of 0/1
    session_texts_dict: {id: token_list}
    """
    # Compute loss for each session
    losses = []
    with torch.no_grad():
        for sid in tqdm(val_ids, desc="Calibrating threshold (val)", leave=False):
            toks = session_texts_dict.get(sid, [])
            if len(toks) < 2:
                losses.append(0.0)
                continue
            seq = toks[:block_size]
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits, _ = model(x)
            inp = x[:, :-1]; tgt = x[:, 1:]
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                                   tgt.reshape(-1), ignore_index=PAD_TOKEN_ID)
            losses.append(loss.item())
    
    # Sweep thresholds
    threshs = np.linspace(min(losses) * 0.8, max(losses) * 1.2, n_steps)
    best_f1, best_thresh = 0, threshs[0]
    for t in threshs:
        preds = [1 if l > t else 0 for l in losses]
        _, _, f1, *_ = metrics(val_labels, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return float(best_thresh), float(best_f1)


# ─── Collate for batch inference ──────────────────────────────────────────────
class SessionBatchDataset(Dataset):
    def __init__(self, sessions, tokenizer, block_size):
        self.items = []
        for sid, label, text in sessions:
            toks = tokenizer.encode(text, truncation=True, max_length=block_size)
            self.items.append((sid, label, toks))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate_session(batch):
    sids, labels, toks_list = zip(*batch)
    max_l = max(len(t) for t in toks_list)
    padded = torch.full((len(toks_list), max_l), PAD_TOKEN_ID, dtype=torch.long)
    for i, t in enumerate(toks_list):
        padded[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    return list(sids), torch.tensor(labels, dtype=torch.long), padded


def run_detection_batch(model, loader, device, K=5, threshold=None,
                        session_timestamps=None, session_error_ts=None):
    """
    Runs both Top-K and Cross-Entropy detection on the loader.
    Returns list of dicts: {sid, label, topk_pred, ce_pred, topk_lt, ce_lt}
    """
    results = []
    model.eval()
    with torch.no_grad():
        for sids, labels, input_ids in tqdm(loader, desc="Detecting", leave=False):
            input_ids = input_ids.to(device)
            logits, _ = model(input_ids)
            
            targets  = input_ids[:, 1:]          # (B, T-1)
            preds_lg = logits[:, :-1, :]          # (B, T-1, V)

            # ── Top-K ────────────────────────────────────────────────────────
            probs = torch.softmax(preds_lg, dim=-1)
            _, topk_inds = torch.topk(probs, K, dim=-1)
            matches = (topk_inds == targets.unsqueeze(-1)).any(dim=-1)
            target_mask = (targets != PAD_TOKEN_ID)
            valid_anom = (~matches) & target_mask
            is_topk_anom = valid_anom.any(dim=1).cpu().numpy()
            first_topk_step = valid_anom.int().argmax(dim=1).cpu().numpy()

            # ── Cross-Entropy ─────────────────────────────────────────────────
            ce_losses = []
            for b in range(input_ids.size(0)):
                mask = target_mask[b]
                if mask.sum() == 0:
                    ce_losses.append(0.0)
                    continue
                lg = preds_lg[b][mask]
                tg = targets[b][mask]
                ce_losses.append(F.cross_entropy(lg, tg).item())

            for i, sid in enumerate(sids):
                lbl = int(labels[i].item())
                topk_pred = int(is_topk_anom[i])
                ce_pred   = int(ce_losses[i] > threshold) if threshold else 0
                fstep = int(first_topk_step[i]) if topk_pred else -1

                topk_lt = None
                ce_lt   = None
                if session_timestamps and session_error_ts:
                    ts_list  = session_timestamps.get(sid, [])
                    err_ts   = session_error_ts.get(sid)
                    if err_ts:
                        # Top-K lead time
                        if topk_pred and fstep >= 0 and fstep < len(ts_list):
                            try:
                                at = pd.to_datetime(ts_list[fstep])
                                et = pd.to_datetime(err_ts)
                                topk_lt = (et - at).total_seconds() / 60.0
                            except: pass
                        # CE lead time: alert on whole session so use first token ts
                        if ce_pred and ts_list:
                            try:
                                at = pd.to_datetime(ts_list[0])
                                et = pd.to_datetime(err_ts)
                                ce_lt = (et - at).total_seconds() / 60.0
                            except: pass

                results.append({
                    "sid": sid, "label": lbl,
                    "topk_pred": topk_pred, "ce_pred": ce_pred,
                    "topk_lt": topk_lt, "ce_lt": ce_lt,
                    "ce_loss": ce_losses[i],
                })
    return results


def summarise(results, method_key, label_key="label"):
    preds  = [r[method_key] for r in results]
    labels = [r[label_key]  for r in results]
    prec, rec, f1, tp, fp, fn, tn = metrics(labels, preds)
    lt_key = "topk_lt" if "topk" in method_key else "ce_lt"
    lts = [r[lt_key] for r in results if r[method_key] == 1 and r[lt_key] is not None and r[lt_key] > 0]
    lt_mean = sum(lts) / len(lts) if lts else None
    return prec, rec, f1, tp, fp, fn, tn, lt_mean


def print_table(rows):
    header = f"{'Dataset':10} | {'Method':18} | {'F1':6} | {'Prec':6} | {'Rec':6} | {'TP':5} | {'FP':5} | {'FN':5} | {'TN':5} | {'Lead Avg':10}"
    print("\n" + "="*len(header))
    print(header)
    print("-"*len(header))
    for r in rows:
        lt = f"{r['lt_mean']:.2f}min" if r['lt_mean'] else "  —  "
        print(f"{r['dataset']:10} | {r['method']:18} | "
              f"{r['f1']:.4f} | {r['prec']:.4f} | {r['rec']:.4f} | "
              f"{r['tp']:5} | {r['fp']:5} | {r['fn']:5} | {r['tn']:5} | {lt:10}")
    print("="*len(header))


# ═══════════════════════════════════════════════════════════════════════════════
# OPENSTACK
# ═══════════════════════════════════════════════════════════════════════════════
def run_openstack(K=5, batch_size=16, device="cuda"):
    print("\n" + "█"*60)
    print("█  OPENSTACK — Loading model and data...")
    print("█"*60)

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model_dir = ROOT / "01_OpenStack_Validated" / "models" / "loggpt_custom"
    cfg   = torch.load(model_dir / "config.pt", weights_only=False)
    if not hasattr(cfg, "dropout"): cfg.dropout = 0.0
    model = LogGPT(cfg)
    model.load_state_dict(torch.load(model_dir / "loggpt_weights.pt", weights_only=False))
    model.to(device).eval()
    print(f"   ✅ Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Load data via existing OpenStack helpers (same as detect_custom.py)
    sys.path.insert(0, str(ROOT / "01_OpenStack_Validated"))
    from dataset import load_openstack_data, prepare_session_strings
    sys.path.pop(0)

    df = load_openstack_data()
    all_sessions = prepare_session_strings(df)

    # Same split as detect_custom.py
    normal_ids = df.filter(pl.col("anom_label") == 0)["test_id"].unique().to_list()
    anom_ids   = df.filter(pl.col("anom_label") == 1)["test_id"].unique().to_list()
    _, test_val_ids  = train_test_split(normal_ids, test_size=os_cfg.TEST_SIZE_NORMAL, random_state=42)
    val_ids, test_norm_ids = train_test_split(test_val_ids, test_size=os_cfg.TEST_SIZE_VAL_SPLIT, random_state=42)

    test_norm_ids = test_norm_ids[:1000]
    anom_ids = anom_ids[:1000]
    test_ids_set = set(test_norm_ids + anom_ids)
    test_sessions_df = all_sessions.filter(pl.col("test_id").is_in(list(test_ids_set)))
    print(f"   Test set: {len(test_norm_ids)} normal + {len(anom_ids)} anomaly sessions")

    # Build session text dict for tokenisation
    sess_text = {row["test_id"]: row["EventTemplate"]
                 for row in test_sessions_df.iter_rows(named=True)}
    sess_label = {}
    for row in test_sessions_df.iter_rows(named=True):
        tid = row["test_id"]
        # anomaly if in anom_ids
        sess_label[tid] = 1 if tid in set(anom_ids) else 0

    # Timestamps for lead time
    session_timestamps = {}
    session_error_ts   = {}
    for (tid,), grp in df.sort("timestamp").group_by("test_id", maintain_order=True):
        ts_list = grp["timestamp"].to_list()
        session_timestamps[tid] = ts_list
        err_rows = grp.filter(pl.col("anom_label") == 1)
        if len(err_rows) > 0:
            session_error_ts[tid] = err_rows["timestamp"][0]

    # Build sessions list: (sid, label, text)
    sessions_list = [(sid, sess_label[sid], text)
                     for sid, text in sess_text.items()]

    # ── Calibrate Cross-Entropy threshold on validation set ──────────────────
    print("   Calibrating CE threshold on validation set...")
    val_sessions_df = all_sessions.filter(pl.col("test_id").is_in(val_ids))
    val_text  = {r["test_id"]: r["EventTemplate"]   for r in val_sessions_df.iter_rows(named=True)}
    val_label = {r["test_id"]: (1 if r["test_id"] in set(anom_ids) else 0)
                 for r in val_sessions_df.iter_rows(named=True)}
    # Encode val sessions
    val_tok = {}
    for sid, text in val_text.items():
        val_tok[sid] = tokenizer.encode(text, truncation=True, max_length=cfg.block_size)

    val_ids_list  = list(val_tok.keys())
    val_lbl_list  = [val_label[sid] for sid in val_ids_list]
    ce_thresh_os, cal_f1 = calibrate_threshold_on_val(
        model, val_ids_list, val_lbl_list, val_tok, tokenizer, cfg.block_size, device)
    print(f"   ✅ CE threshold = {ce_thresh_os:.4f}  (val F1 = {cal_f1:.4f})")

    # ── Run detection ────────────────────────────────────────────────────────
    ds     = SessionBatchDataset(sessions_list, tokenizer, cfg.block_size)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_session, shuffle=False)
    results = run_detection_batch(model, loader, device, K=K, threshold=ce_thresh_os,
                                  session_timestamps=session_timestamps,
                                  session_error_ts=session_error_ts)
    return results, "OpenStack"


# ═══════════════════════════════════════════════════════════════════════════════
# HDFS
# ═══════════════════════════════════════════════════════════════════════════════
def run_hdfs(K=5, batch_size=16, device="cuda"):
    print("\n" + "█"*60)
    print("█  HDFS — Loading model and data...")
    print("█"*60)

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model_dir = ROOT / "03_HDFS_Benchmark" / "saved_models"
    cfg   = torch.load(model_dir / "config.pt", weights_only=False)
    if not hasattr(cfg, "dropout"): cfg.dropout = 0.0
    model = LogGPT(cfg)
    model.load_state_dict(torch.load(model_dir / "hdfs_loggpt.pt", weights_only=False))
    model.to(device).eval()
    print(f"   ✅ Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Load HDFS config (already loaded at module level as hdfs_cfg)
    # Load data with pandas (handles 11M rows fine)
    data_path = str(hdfs_cfg.DATA_FILE)
    print(f"   Loading {data_path} ...")
    df_pd = pd.read_csv(data_path, usecols=["session_id", "timestamp", "EventTemplate", "anom_label"])
    print(f"   Loaded {len(df_pd):,} rows")

    # Same split as detect.py
    all_normal_ids = df_pd[df_pd["anom_label"] == 0]["session_id"].unique().tolist()
    anom_ids       = df_pd[df_pd["anom_label"] == 1]["session_id"].unique().tolist()
    _, test_val_ids  = train_test_split(all_normal_ids, test_size=hdfs_cfg.TEST_SIZE_NORMAL, random_state=42)
    val_ids, test_norm_ids = train_test_split(test_val_ids, test_size=hdfs_cfg.TEST_SIZE_VAL_SPLIT, random_state=42)
    print(f"   Test set: {len(test_norm_ids)} normal + {len(anom_ids)} anomaly sessions")

    test_norm_ids = test_norm_ids[:1000]
    anom_ids = anom_ids[:1000]
    test_ids_set = set(test_norm_ids) | set(anom_ids)
    val_ids_set  = set(val_ids)

    # Sort once
    df_pd = df_pd.sort_values("timestamp")

    # Build session data (only for test + val)
    relevant_ids = test_ids_set | val_ids_set
    df_rel = df_pd[df_pd["session_id"].isin(relevant_ids)]

    print("   Building session text dict...")
    session_texts      = {}
    session_timestamps = {}
    session_error_ts   = {}
    val_tok            = {}

    for sid, grp in tqdm(df_rel.groupby("session_id", sort=False), desc="Building sessions", leave=False):
        templates = grp["EventTemplate"].fillna("").tolist()
        ts_list   = grp["timestamp"].tolist()

        if sid in test_ids_set:
            session_texts[sid]      = " \n ".join(str(t) for t in templates)
            session_timestamps[sid] = ts_list
            err_rows = grp[grp["anom_label"] == 1]
            if len(err_rows) > 0:
                session_error_ts[sid] = err_rows["timestamp"].iloc[0]

        if sid in val_ids_set:
            text = " \n ".join(str(t) for t in templates)
            val_tok[sid] = tokenizer.encode(text, truncation=True, max_length=cfg.block_size)

    sess_label = {sid: (1 if sid in set(anom_ids) else 0) for sid in test_ids_set}

    # ── Load existing CE threshold ────────────────────────────────────────────
    thresh_path = ROOT / "03_HDFS_Benchmark" / "threshold_config.json"
    ce_thresh_hd = hdfs_cfg.THRESHOLD
    if thresh_path.exists():
        with open(thresh_path) as f:
            ce_thresh_hd = json.load(f).get("threshold", ce_thresh_hd)
    print(f"   ✅ CE threshold = {ce_thresh_hd:.4f} (from threshold_config.json)")

    print(f"   → Using threshold from threshold_config.json: {ce_thresh_hd:.4f}")

    sessions_list = [(sid, sess_label[sid], text)
                     for sid, text in session_texts.items()]

    ds     = SessionBatchDataset(sessions_list, tokenizer, cfg.block_size)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_session, shuffle=False)
    results = run_detection_batch(model, loader, device, K=K, threshold=ce_thresh_hd,
                                  session_timestamps=session_timestamps,
                                  session_error_ts=session_error_ts)
    return results, "HDFS"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    import traceback
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔬 detect_comparison.py — Top-K vs Cross-Entropy")
    print(f"   Device: {device.upper()}")

    table_rows = []
    all_results = {}

    for run_fn, ds_name in [(run_openstack, "OpenStack"), (run_hdfs, "HDFS")]:
        try:
            results, ds_name = run_fn(K=5, device=device)
        except Exception as e:
            print(f"\n❌ ERROR in {ds_name}: {e}")
            traceback.print_exc()
            continue
        all_results[ds_name] = results

        for method in ["topk", "ce"]:
            pred_key = f"{method}_pred"
            prec, rec, f1, tp, fp, fn, tn, lt_mean = summarise(results, pred_key)
            table_rows.append({
                "dataset": ds_name,
                "method": "Top-K (K=5)" if method == "topk" else "Cross-Entropy",
                "f1": f1, "prec": prec, "rec": rec,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "lt_mean": lt_mean,
            })

    if not table_rows:
        print("\n❌ No results to show.")
        return

    print_table(table_rows)

    # ── Recommendation ─────────────────────────────────────────────────────
    print("\n🏆 RECOMMENDATION:")
    os_rows = [r for r in table_rows if r["dataset"] == "OpenStack"]
    hd_rows = [r for r in table_rows if r["dataset"] == "HDFS"]

    if os_rows and hd_rows:
        os_topk = next((r for r in os_rows if "Top-K" in r["method"]), None)
        os_ce   = next((r for r in os_rows if "Cross" in r["method"]), None)
        hd_topk = next((r for r in hd_rows if "Top-K" in r["method"]), None)
        hd_ce   = next((r for r in hd_rows if "Cross" in r["method"]), None)

        if os_topk and hd_topk and os_ce and hd_ce:
            topk_avg_f1 = (os_topk["f1"] + hd_topk["f1"]) / 2
            ce_avg_f1   = (os_ce["f1"]   + hd_ce["f1"])   / 2
            winner = "Top-K (K=5)" if topk_avg_f1 >= ce_avg_f1 else "Cross-Entropy"
            print(f"   Top-K avg F1:         {topk_avg_f1:.4f}")
            print(f"   Cross-Entropy avg F1: {ce_avg_f1:.4f}")
            print(f"   → Winner: {winner}")
            print(f"\n   Use this method for the LLM benchmark.")

    # Save JSON
    out = {"table": table_rows}
    with open(ROOT / "detection_comparison_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n✅ Results saved to {ROOT / 'detection_comparison_results.json'}")


if __name__ == "__main__":
    main()
