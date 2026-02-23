# -*- coding: utf-8 -*-
"""
MEGA ANALYSIS: Detailed HDFS Detection Report
Categorizes errors by template, analyzes lead times, and generates
comprehensive statistics per error category.
"""
import pickle
import polars as pl
import pandas as pd
import numpy as np
import json
from collections import Counter, defaultdict
from pathlib import Path

from config import (
    DATA_FILE, SESSION_ID_COL, TIMESTAMP_COL,
    TEMPLATE_COL, LABEL_COL, INFER_SCHEMA_LENGTH
)

print("=" * 70)
print("  MEGA ANALYSIS: HDFS Anomaly Detection - Detailed Report")
print("=" * 70)

# =========================================================================
# 1. LOAD DETECTION RESULTS
# =========================================================================
print("\n[1/6] Loading detection results...")
with open("detection_results_partial.pkl", "rb") as f:
    results = pickle.load(f)

print(f"  Total sessions evaluated: {len(results)}")

# Basic counts
tp_results = [r for r in results if r['label'] == 1 and r['is_detected']]
fn_results = [r for r in results if r['label'] == 1 and not r['is_detected']]
fp_results = [r for r in results if r['label'] == 0 and r['is_detected']]
tn_results = [r for r in results if r['label'] == 0 and not r['is_detected']]

print(f"  TP={len(tp_results)} FN={len(fn_results)} FP={len(fp_results)} TN={len(tn_results)}")

# All anomalous sessions
all_anomalous = [r for r in results if r['label'] == 1]
all_normal = [r for r in results if r['label'] == 0]
print(f"  Anomalous sessions: {len(all_anomalous)}")
print(f"  Normal sessions: {len(all_normal)}")

# =========================================================================
# 2. LOAD DATASET FOR TEMPLATE INFORMATION
# =========================================================================
print("\n[2/6] Loading HDFS dataset for template analysis...")
df = pl.read_csv(str(DATA_FILE), infer_schema_length=INFER_SCHEMA_LENGTH)
print(f"  Total log lines: {len(df)}")

# Get all unique templates
all_templates = df[TEMPLATE_COL].unique().to_list()
print(f"  Unique templates: {len(all_templates)}")

# =========================================================================
# 3. CATEGORIZE ERRORS BY TEMPLATE TYPE
# =========================================================================
print("\n[3/6] Categorizing errors by template type...")

# Create template categories based on keywords
def categorize_template(template):
    """Categorize HDFS log template into error category."""
    t = str(template).lower()
    if 'exception' in t and 'interruptedio' in t:
        return "InterruptedIOException"
    elif 'exception' in t and 'sockettime' in t:
        return "SocketTimeoutException"
    elif 'exception' in t and 'closedbyinterrupt' in t:
        return "ClosedByInterruptException"
    elif 'exception' in t and 'eofexception' in t:
        return "EOFException"
    elif 'exception' in t and 'sockettimeout' in t:
        return "SocketTimeoutException"
    elif 'exception' in t:
        return "Other Exception"
    elif 'writeblock' in t and 'exception' in t.lower():
        return "WriteBlock Exception"
    elif 'writeblock' in t:
        return "WriteBlock Error"
    elif 'receiveblock' in t:
        return "ReceiveBlock Error"
    elif 'connection reset' in t:
        return "Connection Reset"
    elif 'block' in t and 'valid' in t:
        return "Block Already Valid"
    elif 'unexpected error' in t and 'delete' in t:
        return "Block Delete Error"
    elif 'packetresponder' in t:
        return "PacketResponder Error"
    elif 'namesystem' in t or 'blockmap' in t:
        return "NameSystem/BlockMap"
    elif 'receiving' in t or 'received' in t:
        return "Block Receiving"
    elif 'replicate' in t or 'replicat' in t:
        return "Replication"
    elif 'verification' in t:
        return "Block Verification"
    elif 'served' in t or 'serving' in t:
        return "Block Serving"
    elif 'changing' in t and 'offset' in t:
        return "Block Offset Change"
    elif 'delete' in t:
        return "Block Deletion"
    elif 'allocat' in t:
        return "Block Allocation"
    else:
        return "Other"

# Map templates to categories
template_categories = {}
for t in all_templates:
    template_categories[str(t)] = categorize_template(t)

# Print unique categories
unique_cats = set(template_categories.values())
print(f"  Unique categories: {len(unique_cats)}")
for cat in sorted(unique_cats):
    count = sum(1 for v in template_categories.values() if v == cat)
    print(f"    {cat}: {count} templates")

# =========================================================================
# 4. ANALYZE SESSIONS BY ERROR CATEGORY
# =========================================================================
print("\n[4/6] Analyzing sessions by error category...")

# For each anomalous session, find dominant template categories
result_ids = {r['session_id'] for r in results}
anomalous_ids = {r['session_id'] for r in all_anomalous}
normal_ids = {r['session_id'] for r in all_normal}

# Group logs by session for anomalous sessions
print("  Loading anomalous session details (this may take a moment)...")
anomalous_df = df.filter(
    pl.col(SESSION_ID_COL).is_in(list(anomalous_ids))
)

# For each session, get templates and categorize
session_categories = {}
session_templates_detail = {}

for session_id in anomalous_ids:
    sess_logs = anomalous_df.filter(pl.col(SESSION_ID_COL) == session_id)
    templates = sess_logs[TEMPLATE_COL].to_list()

    cats = [categorize_template(t) for t in templates]
    cat_counts = Counter(cats)

    # Dominant category (most frequent)
    dominant = cat_counts.most_common(1)[0][0] if cat_counts else "Unknown"
    session_categories[session_id] = {
        "dominant": dominant,
        "categories": dict(cat_counts),
        "num_logs": len(templates),
        "unique_cats": len(cat_counts)
    }
    session_templates_detail[session_id] = templates

# Build results by category
result_by_session = {r['session_id']: r for r in results}

category_stats = defaultdict(lambda: {
    "total": 0, "detected": 0, "missed": 0,
    "lead_times": [], "alert_losses": [],
    "session_sizes": [], "zero_lead": 0
})

for sid, info in session_categories.items():
    r = result_by_session.get(sid)
    if r is None:
        continue

    cat = info["dominant"]
    stats = category_stats[cat]
    stats["total"] += 1
    stats["session_sizes"].append(info["num_logs"])

    if r['is_detected']:
        stats["detected"] += 1
        if r["lead_time"] > 0:
            stats["lead_times"].append(r["lead_time"])
        else:
            stats["zero_lead"] += 1
        stats["alert_losses"].append(r["alert_loss"])
    else:
        stats["missed"] += 1

# =========================================================================
# 5. COMPUTE DETAILED METRICS PER CATEGORY
# =========================================================================
print("\n[5/6] Computing detailed metrics per category...")

category_report = []
for cat, stats in sorted(category_stats.items(), key=lambda x: x[1]["total"], reverse=True):
    total = stats["total"]
    detected = stats["detected"]
    missed = stats["missed"]
    recall = detected / total if total > 0 else 0

    leads = stats["lead_times"]
    avg_lead = np.mean(leads) if leads else 0
    median_lead = np.median(leads) if leads else 0
    min_lead = np.min(leads) if leads else 0
    max_lead = np.max(leads) if leads else 0
    std_lead = np.std(leads) if leads else 0

    avg_loss = np.mean(stats["alert_losses"]) if stats["alert_losses"] else 0
    avg_size = np.mean(stats["session_sizes"]) if stats["session_sizes"] else 0

    category_report.append({
        "category": cat,
        "total_sessions": total,
        "detected": detected,
        "missed": missed,
        "recall": recall,
        "avg_lead_time_min": avg_lead,
        "median_lead_time_min": median_lead,
        "min_lead_time_min": min_lead,
        "max_lead_time_min": max_lead,
        "std_lead_time_min": std_lead,
        "zero_lead_count": stats["zero_lead"],
        "anticipated_count": len(leads),
        "avg_alert_loss": avg_loss,
        "avg_session_size": avg_size,
    })

# =========================================================================
# 6. GENERATE OUTPUT
# =========================================================================
print("\n[6/6] Generating mega report...")

# Sort by total sessions
category_report.sort(key=lambda x: x["total_sessions"], reverse=True)

# --- CATEGORY TABLE ---
print("\n" + "=" * 120)
print(f"{'CATEGORY':<30} {'TOTAL':>6} {'DET':>5} {'MISS':>5} {'RECALL':>7} {'AVG LEAD':>9} {'MED LEAD':>9} {'MIN LEAD':>9} {'MAX LEAD':>9}")
print("=" * 120)
for c in category_report:
    print(f"{c['category']:<30} {c['total_sessions']:>6} {c['detected']:>5} {c['missed']:>5} {c['recall']:>7.1%} {c['avg_lead_time_min']:>8.1f}m {c['median_lead_time_min']:>8.1f}m {c['min_lead_time_min']:>8.1f}m {c['max_lead_time_min']:>8.1f}m")
print("=" * 120)

# --- FASTEST DETECTION (SMALLEST LEAD TIME > 0) ---
print("\n\n" + "=" * 70)
print("  TOP 20: Fastest Anomaly Detection (Shortest Lead Time > 0)")
print("=" * 70)

tp_with_lead = [r for r in tp_results if r['lead_time'] > 0]
tp_sorted_asc = sorted(tp_with_lead, key=lambda x: x['lead_time'])

for i, r in enumerate(tp_sorted_asc[:20]):
    cat = session_categories.get(r['session_id'], {}).get("dominant", "Unknown")
    size = session_categories.get(r['session_id'], {}).get("num_logs", 0)
    print(f"  {i+1:>2}. Lead: {r['lead_time']:>8.2f} min | Loss: {r['alert_loss']:.4f} | Cat: {cat:<28} | Logs: {size}")

# --- SLOWEST DETECTION (LONGEST LEAD TIME) ---
print("\n\n" + "=" * 70)
print("  TOP 20: Most Anticipated Anomalies (Longest Lead Time)")
print("=" * 70)

tp_sorted_desc = sorted(tp_with_lead, key=lambda x: x['lead_time'], reverse=True)
for i, r in enumerate(tp_sorted_desc[:20]):
    cat = session_categories.get(r['session_id'], {}).get("dominant", "Unknown")
    size = session_categories.get(r['session_id'], {}).get("num_logs", 0)
    print(f"  {i+1:>2}. Lead: {r['lead_time']:>8.2f} min | Loss: {r['alert_loss']:.4f} | Cat: {cat:<28} | Logs: {size}")

# --- HIGHEST LOSS ALERTS ---
print("\n\n" + "=" * 70)
print("  TOP 20: Highest Alert Loss (Most Anomalous Patterns)")
print("=" * 70)

tp_sorted_loss = sorted(tp_results, key=lambda x: x['alert_loss'], reverse=True)
for i, r in enumerate(tp_sorted_loss[:20]):
    cat = session_categories.get(r['session_id'], {}).get("dominant", "Unknown")
    lead = r['lead_time']
    print(f"  {i+1:>2}. Loss: {r['alert_loss']:>8.4f} | Lead: {lead:>8.2f} min | Cat: {cat}")

# --- MISSED ANOMALIES (FN) ---
print("\n\n" + "=" * 70)
print("  MISSED ANOMALIES (False Negatives) by Category")
print("=" * 70)

fn_by_cat = defaultdict(list)
for r in fn_results:
    cat = session_categories.get(r['session_id'], {}).get("dominant", "Unknown")
    fn_by_cat[cat].append(r)

for cat, items in sorted(fn_by_cat.items(), key=lambda x: len(x[1]), reverse=True):
    sizes = [session_categories.get(r['session_id'], {}).get("num_logs", 0) for r in items]
    print(f"  {cat:<30} {len(items):>5} missed | Avg session size: {np.mean(sizes):.0f} logs")

# --- FALSE POSITIVES ---
print("\n\n" + "=" * 70)
print("  FALSE POSITIVES: Normal Sessions Flagged as Anomalous")
print("=" * 70)

# Get FP session details from dataset
fp_ids = [r['session_id'] for r in fp_results]
print(f"  Total FP: {len(fp_results)}")

fp_losses = [r['alert_loss'] for r in fp_results]
if fp_losses:
    print(f"  FP Loss range: {min(fp_losses):.4f} - {max(fp_losses):.4f}")
    print(f"  FP Avg Loss: {np.mean(fp_losses):.4f}")
    print(f"  FP Median Loss: {np.median(fp_losses):.4f}")

# --- GLOBAL SUMMARY ---
print("\n\n" + "=" * 70)
print("  GLOBAL SUMMARY")
print("=" * 70)

all_leads = [r['lead_time'] for r in tp_results if r['lead_time'] > 0]
all_losses_tp = [r['alert_loss'] for r in tp_results]
all_losses_fp = [r['alert_loss'] for r in fp_results]

print(f"\n  Precision:   {len(tp_results)/(len(tp_results)+len(fp_results)):.4f}")
print(f"  Recall:      {len(tp_results)/(len(tp_results)+len(fn_results)):.4f}")
f1 = 2*len(tp_results)/(2*len(tp_results)+len(fp_results)+len(fn_results))
print(f"  F1 Score:    {f1:.4f}")
print(f"  Accuracy:    {(len(tp_results)+len(tn_results))/len(results):.4f}")
print(f"  Specificity: {len(tn_results)/(len(tn_results)+len(fp_results)):.4f}")

print(f"\n  Lead Time (positive only, N={len(all_leads)}):")
if all_leads:
    print(f"    Mean:   {np.mean(all_leads):>8.2f} min ({np.mean(all_leads)/60:.1f}h)")
    print(f"    Median: {np.median(all_leads):>8.2f} min ({np.median(all_leads)/60:.1f}h)")
    print(f"    Std:    {np.std(all_leads):>8.2f} min")
    print(f"    Min:    {np.min(all_leads):>8.2f} min")
    print(f"    Max:    {np.max(all_leads):>8.2f} min")

    # Lead time distribution
    print(f"\n  Lead Time Distribution:")
    buckets = [(0, 1), (1, 5), (5, 15), (15, 30), (30, 60), (60, 120), (120, 300), (300, 600), (600, float('inf'))]
    for lo, hi in buckets:
        count = sum(1 for l in all_leads if lo < l <= hi)
        pct = count / len(all_leads) * 100
        label = f"{lo}-{hi}min" if hi != float('inf') else f">{lo}min"
        bar = "#" * int(pct / 2)
        print(f"    {label:>12}: {count:>5} ({pct:>5.1f}%) {bar}")

print(f"\n  Alert Loss Distribution (TP):")
if all_losses_tp:
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(all_losses_tp, p)
        print(f"    P{p:>2}: {val:.4f}")

# --- SAVE JSON FOR REPORT ---
output = {
    "summary": {
        "total_sessions": len(results),
        "anomalous": len(all_anomalous),
        "normal": len(all_normal),
        "tp": len(tp_results),
        "fn": len(fn_results),
        "fp": len(fp_results),
        "tn": len(tn_results),
        "precision": len(tp_results)/(len(tp_results)+len(fp_results)),
        "recall": len(tp_results)/(len(tp_results)+len(fn_results)),
        "f1": f1,
    },
    "categories": category_report,
    "lead_time_stats": {
        "mean": float(np.mean(all_leads)) if all_leads else 0,
        "median": float(np.median(all_leads)) if all_leads else 0,
        "std": float(np.std(all_leads)) if all_leads else 0,
        "min": float(np.min(all_leads)) if all_leads else 0,
        "max": float(np.max(all_leads)) if all_leads else 0,
    },
    "fastest_detections": [
        {"session_id": r['session_id'], "lead_time": r['lead_time'],
         "category": session_categories.get(r['session_id'], {}).get("dominant", "Unknown")}
        for r in tp_sorted_asc[:10]
    ] if tp_sorted_asc else [],
    "missed_by_category": {
        cat: len(items) for cat, items in fn_by_cat.items()
    }
}

with open("mega_analysis_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("\n\n  Results saved to: mega_analysis_results.json")
print("=" * 70)
print("  MEGA ANALYSIS COMPLETE!")
print("=" * 70)
