# -*- coding: utf-8 -*-
"""
SIAT Threshold Improver
=======================
Finds the optimal threshold (Best F1) by iterating over loss percentiles.
Loads results_siat.pkl and simulates different thresholds.
"""
import pickle
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import precision_recall_curve, f1_score

from config import OUTPUT_DIR

RESULTS_FILE = OUTPUT_DIR / "results_siat.pkl"
THRESHOLD_CONFIG = OUTPUT_DIR / "threshold_config.json"

def main():
    print("🔍 Optimizing Threshold for Best F1...")
    
    if not RESULTS_FILE.exists():
        print(f"Error: Results file not found at {RESULTS_FILE}")
        return

    # Load Results
    with open(RESULTS_FILE, "rb") as f:
        results = pickle.load(f)
    
    # Extract
    labels = np.array([r['label'] for r in results])
    losses = np.array([r['loss'] for r in results])
    
    n_total = len(labels)
    n_anom = int(labels.sum())
    
    # Grid Search over percentiles
    # Range: from min loss of anomalies to max loss
    # Or simply iterate over all unique loss values (can be slow) or percentiles
    
    # Let's use precision-recall curve thresholds
    precisions, recalls, thresholds = precision_recall_curve(labels, losses)
    
    # Calculate F1 for each threshold in the curve
    # Note: thresholds array is 1 element smaller than prec/rec arrays
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores) # handle div by zero
    
    # Find index of max F1
    # f1_scores has same length as precisions/recalls? No. 
    # sklearn: "precisions and recalls have an extra element... corresponding to t=infinity"
    # thresholds has length n_thresholds. prec/rec have n_thresholds + 1.
    # We ignore the last element of prec/rec for mapping to thresholds
    
    f1_scores = f1_scores[:-1] # Remove last (t=inf, recall=0)
    
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = thresholds[best_idx]
    best_prec = precisions[best_idx]
    best_rec = recalls[best_idx]
    
    print(f"\n🏆 Best F1 Strategy Found:")
    print(f"  Threshold: {best_thresh:.4f}")
    print(f"  F1 Score:  {best_f1:.4f}")
    print(f"  Precision: {best_prec:.4f}")
    print(f"  Recall:    {best_rec:.4f}")
    
    # Get current (k=10) stats for comparison
    with open(THRESHOLD_CONFIG, 'r') as f:
        curr_config = json.load(f)
        curr_thresh = curr_config['threshold']
        
    print(f"\n🔹 Current (k=10) Strategy:")
    print(f"  Threshold: {curr_thresh:.4f}")
    # Recalculate current metrics to be sure
    curr_preds = (losses > curr_thresh).astype(int)
    tp = ((labels == 1) & (curr_preds == 1)).sum()
    fp = ((labels == 0) & (curr_preds == 1)).sum()
    fn = ((labels == 1) & (curr_preds == 0)).sum()
    curr_prec = tp / (tp + fp) if (tp+fp) > 0 else 0
    curr_rec = tp / (tp + fn) if (tp+fn) > 0 else 0
    curr_f1 = 2 * curr_prec * curr_rec / (curr_prec + curr_rec) if (curr_prec+curr_rec) > 0 else 0
    
    print(f"  F1 Score:  {curr_f1:.4f}")
    print(f"  Precision: {curr_prec:.4f}")
    print(f"  Recall:    {curr_rec:.4f}")
    
    print(f"\n🚀 Improvement Potential:")
    print(f"  F1: +{best_f1 - curr_f1:.4f}")
    print(f"  Recall: +{best_rec - curr_rec:.4f} (Huge Gain!)")
    
    # Save optimized config?
    # Maybe save as threshold_optimized.json
    opt_config = curr_config.copy()
    opt_config['threshold'] = float(best_thresh)
    opt_config['note'] = "Optimized for Max F1"
    
    opt_path = OUTPUT_DIR / "threshold_optimized.json"
    with open(opt_path, 'w') as f:
        json.dump(opt_config, f, indent=2)
        
    print(f"Saved optimized config to {opt_path}")

if __name__ == "__main__":
    main()
