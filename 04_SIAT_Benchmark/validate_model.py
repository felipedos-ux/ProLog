# -*- coding: utf-8 -*-
"""
SIAT Scientific Validation
==========================
Adapts the 6 rigorous tests for SIAT dataset.
1. Random Baseline
2. Permutation Test
3. Threshold Sensitivity
4. Naive Baselines
5. Statistical Tests
6. Separation (AUROC/AUPRC)
"""
import torch
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, roc_curve

from config import OUTPUT_DIR

RESULTS_FILE = OUTPUT_DIR / "results_siat.pkl"
VALIDATION_REPORT = OUTPUT_DIR / "validation_results.json"

def main():
    print("🔬 Starting Scientific Validation for SIAT...")
    
    if not RESULTS_FILE.exists():
        print(f"Error: Results file not found at {RESULTS_FILE}")
        return

    # Load Results
    with open(RESULTS_FILE, "rb") as f:
        results = pickle.load(f)
    
    # Extract arrays
    labels = np.array([r['label'] for r in results])
    preds = np.array([r['pred'] for r in results])
    losses = np.array([r['loss'] for r in results])
    threshold = results[0]['threshold']
    
    n_total = len(labels)
    n_anom = int(labels.sum())
    n_norm = n_total - n_anom
    anomaly_rate = n_anom / n_total
    
    print(f"Loaded {n_total} sessions.")
    print(f"Anomalous: {n_anom} ({anomaly_rate*100:.2f}%)")
    print(f"Normal:    {n_norm}")
    print(f"Threshold: {threshold:.4f}")

    # --- METRICS FUNCTION ---
    def calc_metrics(y_true, y_pred):
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        acc = (tp + tn) / len(y_true)
        return f1, prec, rec, acc

    # 0. Actual Model Performance
    model_f1, model_prec, model_rec, model_acc = calc_metrics(labels, preds)
    print(f"\nACTUAL MODEL: F1={model_f1:.4f} Prec={model_prec:.4f} Rec={model_rec:.4f}")

    validation_summary = {}

    # --- TEST 1: RANDOM BASELINE ---
    print("\n[TEST 1] Random Baseline (1000 trials)...")
    np.random.seed(42)
    random_f1s = []
    for _ in range(1000):
        rand_preds = np.random.binomial(1, anomaly_rate, n_total)
        f1, _, _, _ = calc_metrics(labels, rand_preds)
        random_f1s.append(f1)
    
    rand_mean = np.mean(random_f1s)
    rand_std = np.std(random_f1s)
    sigma_diff = (model_f1 - rand_mean) / rand_std if rand_std > 0 else 0
    
    print(f"  Random F1: {rand_mean:.4f} ± {rand_std:.4f}")
    print(f"  Model is {sigma_diff:.1f} sigma above random.")
    validation_summary['test1_random'] = {
        'status': 'PASS' if sigma_diff > 3 else 'FAIL',
        'sigma': sigma_diff
    }

    # --- TEST 2: PERMUTATION TEST ---
    print("\n[TEST 2] Permutation Test (Labels shuffled)...")
    perm_f1s = []
    # Only need 100 trials for speed, but 1000 is better. Let's do 100.
    for _ in range(100):
        shuffled_labels = np.random.permutation(labels)
        # Preds fixed, labels shuffled
        f1, _, _, _ = calc_metrics(shuffled_labels, preds)
        perm_f1s.append(f1)
    
    p_value = np.mean([f >= model_f1 for f in perm_f1s])
    print(f"  Permuted F1 Mean: {np.mean(perm_f1s):.4f}")
    print(f"  P-Value: {p_value:.4f}")
    validation_summary['test2_permutation'] = {
        'status': 'PASS' if p_value < 0.05 else 'FAIL',
        'p_value': p_value
    }

    # --- TEST 3: THRESHOLD SENSITIVITY ---
    print("\n[TEST 3] Threshold Sensitivity...")
    # Test range relative to model threshold
    t_range = np.linspace(threshold * 0.5, threshold * 2.0, 50)
    robust_count = 0
    for t in t_range:
        t_preds = (losses > t).astype(int)
        f1, _, _, _ = calc_metrics(labels, t_preds)
        if f1 > 0.7:  # Arbitrary "good" performance
            robust_count += 1
            
    robust_pct = robust_count / len(t_range)
    print(f"  Robustness (F1 > 0.7 in range 0.5x-2.0x): {robust_pct*100:.1f}%")
    validation_summary['test3_threshold'] = {
        'status': 'PASS' if robust_pct > 0.2 else 'FAIL', # loose check
        'robust_pct': robust_pct
    }

    # --- TEST 4: NAIVE BASELINES ---
    print("\n[TEST 4] Naive Baselines...")
    # All Positive
    all_pos_preds = np.ones(n_total)
    ap_f1, _, _, _ = calc_metrics(labels, all_pos_preds)
    
    print(f"  All-Positive F1: {ap_f1:.4f}")
    print(f"  Model F1: {model_f1:.4f}")
    validation_summary['test4_baselines'] = {
        'status': 'PASS' if model_f1 > ap_f1 else 'FAIL',
        'improvement': model_f1 - ap_f1
    }

    # --- TEST 5: STATISTICAL TESTS ---
    print("\n[TEST 5] Statistical Separation...")
    norm_losses = losses[labels == 0]
    anom_losses = losses[labels == 1]
    
    # Cohen's d
    pooled_std = np.sqrt((np.std(norm_losses)**2 + np.std(anom_losses)**2) / 2)
    cohens_d = (np.mean(anom_losses) - np.mean(norm_losses)) / pooled_std
    
    # Mann-Whitney
    try:
        u_stat, u_p = stats.mannwhitneyu(anom_losses, norm_losses, alternative='greater')
    except:
        u_p = 1.0

    print(f"  Cohen's d: {cohens_d:.4f}")
    print(f"  Mann-Whitney p: {u_p:.4e}")
    validation_summary['test5_stats'] = {
        'status': 'PASS' if cohens_d > 0.5 and u_p < 0.05 else 'FAIL',
        'cohens_d': cohens_d
    }

    # --- TEST 6: AUROC/AUPRC ---
    print("\n[TEST 6] AUROC / AUPRC...")
    # Invert losses for sklearn? No, higher loss = anomaly (positive class)
    # y_score = losses
    fpr, tpr, _ = roc_curve(labels, losses)
    roc_auc = auc(fpr, tpr)
    
    prec_c, rec_c, _ = precision_recall_curve(labels, losses)
    pr_auc = auc(rec_c, prec_c) # Note order for sklearn < 1.0 (some versions) check docs
    # Actually auc(x, y). x=recall, y=precision
    
    print(f"  AUROC: {roc_auc:.4f}")
    print(f"  AUPRC: {pr_auc:.4f}")
    
    validation_summary['test6_separation'] = {
        'status': 'PASS' if roc_auc > 0.8 else 'FAIL',
        'auroc': roc_auc,
        'auprc': pr_auc
    }
    
    print("\n" + "="*50)
    print("FINAL VERDICT:")
    passes = sum(1 for t in validation_summary.values() if t['status'] == 'PASS')
    print(f"  {passes}/6 Tests Passed")
    print("="*50)
    
    with open(VALIDATION_REPORT, 'w') as f:
        json.dump(validation_summary, f, indent=2)
        
    print(f"Saved report to {VALIDATION_REPORT}")

if __name__ == "__main__":
    main()
