"""
Adaptive threshold calibration for LogGPT anomaly detection.
Uses Custom EventTokenizer (1 event = 1 token).
"""

import torch
import torch.nn.functional as F
import polars as pl
import numpy as np
from tqdm import tqdm
import json
import os

from model import LogGPT, GPTConfig
from event_tokenizer import EventTokenizer
from sklearn.model_selection import train_test_split

from config import (
    MODEL_NAME, MODEL_DIR, DEVICE, DATA_FILE, 
    SKIP_START_LOGS, INFER_SCHEMA_LENGTH, 
    TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT,
    LOG_COLUMN,
    set_seeds
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


def collect_session_losses(model, tokenizer, df, session_ids, device):
    """Collect per-event losses for each session."""
    results = {}
    block_size = model.config.block_size
    
    for tid in tqdm(session_ids, desc="Collecting losses"):
        session_df = df.filter(pl.col("test_id") == tid)
        ts_col = "time_hour" if "time_hour" in session_df.columns else "timestamp"
        session_df = session_df.sort(ts_col)
        events = session_df[LOG_COLUMN].to_list()
        
        # Encode all events
        event_token_ids = [tokenizer.encode_single(str(e)) for e in events]
        
        session_losses = []
        
        for i in range(SKIP_START_LOGS, len(events)):
            context_start = max(0, i - block_size + 1)
            context = event_token_ids[context_start:i]
            target = event_token_ids[i]
            
            if len(context) == 0:
                continue
            
            x = torch.tensor([context], dtype=torch.long, device=device)
            
            with torch.no_grad():
                logits, _ = model(x)
            
            last_logits = logits[0, -1, :]
            target_tensor = torch.tensor([target], dtype=torch.long, device=device)
            loss_val = F.cross_entropy(last_logits.unsqueeze(0), target_tensor).item()
            session_losses.append(loss_val)
        
        if session_losses:
            results[tid] = {
                "max_loss": np.max(session_losses),
                "avg_loss": np.mean(session_losses),
                "all_losses": session_losses,
            }
    
    return results


def main():
    logger.info("🔧 Calibrating Adaptive Threshold (Custom EventTokenizer)")
    set_seeds()
    
    # Load model and tokenizer
    tok_path = os.path.join(str(MODEL_DIR), "event_tokenizer.json")
    tokenizer = EventTokenizer.load(tok_path)
    
    config = torch.load(f"{MODEL_DIR}/config.pt", weights_only=False)
    if not hasattr(config, 'dropout'):
        config.dropout = 0.0
    
    model = LogGPT(config)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/loggpt_weights.pt", weights_only=False))
    model.to(DEVICE)
    model.eval()
    
    # Load data
    df = pl.read_csv(str(DATA_FILE), infer_schema_length=INFER_SCHEMA_LENGTH)
    
    normal_ids = df.filter(pl.col("anom_label") == 0)["test_id"].unique().to_list()
    anom_ids = df.filter(pl.col("anom_label") == 1)["test_id"].unique().to_list()
    
    # Split normal IDs (same split as training)
    train_ids, test_val_ids = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    val_ids, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)
    
    # Use val + test normal for calibration (all normal NOT in training)
    calib_normal_ids = val_ids + test_norm_ids
    
    logger.info(f"Calibration: {len(calib_normal_ids)} Normal, {len(anom_ids)} Anomaly sessions")
    
    # Collect losses
    logger.info("📊 Collecting Normal session losses...")
    normal_losses = collect_session_losses(model, tokenizer, df, calib_normal_ids, DEVICE)
    
    logger.info("📊 Collecting Anomaly session losses...")
    anom_losses = collect_session_losses(model, tokenizer, df, anom_ids, DEVICE)
    
    # Distribution analysis
    normal_max = [v["max_loss"] for v in normal_losses.values()]
    normal_avg = [v["avg_loss"] for v in normal_losses.values()]
    anom_max = [v["max_loss"] for v in anom_losses.values()]
    anom_avg = [v["avg_loss"] for v in anom_losses.values()]
    
    logger.info(f"\n📈 Loss Distribution:")
    logger.info(f"   Normal  Max Loss: mean={np.mean(normal_max):.4f}, std={np.std(normal_max):.4f}")
    logger.info(f"   Anomaly Max Loss: mean={np.mean(anom_max):.4f}, std={np.std(anom_max):.4f}")
    logger.info(f"   Normal  Avg Loss: mean={np.mean(normal_avg):.4f}, std={np.std(normal_avg):.4f}")
    logger.info(f"   Anomaly Avg Loss: mean={np.mean(anom_avg):.4f}, std={np.std(anom_avg):.4f}")
    
    separation_max = np.mean(anom_max) - np.mean(normal_max)
    separation_avg = np.mean(anom_avg) - np.mean(normal_avg)
    logger.info(f"   🔍 Separation (max-loss): {separation_max:.4f}")
    logger.info(f"   🔍 Separation (avg-loss): {separation_avg:.4f}")
    
    if separation_max < 0.5:
        logger.warning("⚠️ LOW MAX-LOSS SEPARATION — using avg-loss for calibration")
        # Use avg loss if max loss doesn't separate
        use_avg = True
    else:
        use_avg = False
    
    # Grid search for best threshold
    normal_metric = normal_avg if use_avg else normal_max
    anom_metric = anom_avg if use_avg else anom_max
    metric_name = "avg_loss" if use_avg else "max_loss"
    
    # Also try sigma-based thresholds
    mean_loss = np.mean(normal_metric)
    std_loss = np.std(normal_metric)
    
    best_score = -1
    best_config = {}
    
    # Grid: sigma-based
    for k in np.arange(0.5, 5.1, 0.1):
        threshold = mean_loss + k * std_loss
        tp = sum(1 for v in anom_metric if v > threshold)
        fp = sum(1 for v in normal_metric if v > threshold)
        tn = sum(1 for v in normal_metric if v <= threshold)
        fn = sum(1 for v in anom_metric if v <= threshold)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        score = 0.6 * f1 + 0.4 * specificity
        
        if score > best_score:
            best_score = score
            best_config = {
                "mean_loss": float(mean_loss),
                "std_loss": float(std_loss),
                "k_sigma": float(k),
                "threshold": float(threshold),
                "method": f"balanced_{metric_name}",
                "metrics": {"f1": f1, "precision": precision, "recall": recall, "specificity": specificity},
                "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            }
    
    # Grid: absolute thresholds
    all_losses = sorted(normal_metric + anom_metric)
    for threshold in np.arange(min(all_losses), max(all_losses), 0.1):
        tp = sum(1 for v in anom_metric if v > threshold)
        fp = sum(1 for v in normal_metric if v > threshold)
        tn = sum(1 for v in normal_metric if v <= threshold)
        fn = sum(1 for v in anom_metric if v <= threshold)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        score = 0.6 * f1 + 0.4 * specificity
        
        if score > best_score:
            best_score = score
            best_config = {
                "mean_loss": float(mean_loss),
                "std_loss": float(std_loss),
                "k_sigma": None,
                "threshold": float(threshold),
                "method": f"balanced_{metric_name}_absolute",
                "metrics": {"f1": f1, "precision": precision, "recall": recall, "specificity": specificity},
                "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            }
    
    # Add distribution info
    best_config["distribution"] = {
        "normal_max_loss_mean": float(np.mean(normal_max)),
        "normal_max_loss_std": float(np.std(normal_max)),
        "anomaly_max_loss_mean": float(np.mean(anom_max)),
        "anomaly_max_loss_std": float(np.std(anom_max)),
        "normal_avg_loss_mean": float(np.mean(normal_avg)),
        "normal_avg_loss_std": float(np.std(normal_avg)),
        "anomaly_avg_loss_mean": float(np.mean(anom_avg)),
        "anomaly_avg_loss_std": float(np.std(anom_avg)),
        "separation_max": float(separation_max),
        "separation_avg": float(separation_avg),
        "metric_used": metric_name,
    }
    
    # Save
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threshold_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=4)
    
    logger.info(f"\n🎯 Best Config (Score={best_score:.4f}):")
    logger.info(f"   Threshold: {best_config['threshold']:.4f}")
    logger.info(f"   Method: {best_config['method']}")
    logger.info(f"   Metrics: {best_config['metrics']}")
    logger.info(f"   Confusion: {best_config['confusion_matrix']}")
    logger.info(f"   💾 Saved to {config_path}")


if __name__ == "__main__":
    main()
