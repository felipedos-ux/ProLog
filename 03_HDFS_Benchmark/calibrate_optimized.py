"""
Optimized Adaptive Calibration for HDFS with CPU throttling.
Designed to avoid overwhelming the system while finding optimal threshold.
"""

import torch
import torch.nn.functional as F
import polars as pl
import numpy as np
import json
import time
import pickle
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from model import LogGPT
from config import (
    MODEL_NAME, MODEL_DIR, DEVICE, DATA_FILE,
    SKIP_START_LOGS, INFER_SCHEMA_LENGTH,
    TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT,
    SESSION_ID_COL, TIMESTAMP_COL, TEMPLATE_COL, LABEL_COL,
    set_seeds
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Optimization Parameters (Optimized for RTX 3080 Ti + Ryzen 3600)
SAMPLE_RATIO = 1.0  # Use 100% of sessions for accurate calibration
BATCH_SIZE = 500  # Process 500 sessions per batch (5x increase)
CPU_DELAY = 0.0  # No delay - use full CPU/GPU power
CHECKPOINT_FILE = "calibration_checkpoint.pkl"


def collect_losses_batch(model, tokenizer, sessions_df, desc="Collecting"):
    """Collects losses for a batch of sessions with CPU throttling."""
    all_losses = []
    MAX_CONTEXT_LEN = model.config.block_size
    
    unique_ids = sessions_df[SESSION_ID_COL].unique().to_list()
    
    for tid in tqdm(unique_ids, desc=desc):
        session_df = sessions_df.filter(pl.col(SESSION_ID_COL) == tid)
        session_df = session_df.sort(TIMESTAMP_COL)
        templates = session_df[TEMPLATE_COL].to_list()
        
        context_ids = []
        
        for i, current_log in enumerate(templates):
            if current_log is None:
                current_log = ""
            text = (" \n " if i > 0 else "") + str(current_log)
            new_ids = tokenizer.encode(text)
            
            if i < SKIP_START_LOGS:
                context_ids.extend(new_ids)
                if len(context_ids) > MAX_CONTEXT_LEN:
                    context_ids = context_ids[-MAX_CONTEXT_LEN:]
                continue
            
            if i == 0:
                context_ids.extend(new_ids)
                continue
                
            full_seq = context_ids + new_ids
            if len(full_seq) > MAX_CONTEXT_LEN:
                input_seq = full_seq[-MAX_CONTEXT_LEN:]
                target_start_idx = len(input_seq) - len(new_ids)
            else:
                input_seq = full_seq
                target_start_idx = len(context_ids)
                
            x = torch.tensor(input_seq, dtype=torch.long, device=DEVICE).unsqueeze(0)
            
            with torch.no_grad():
                logits, _ = model(x)
            
            target_indices = range(target_start_idx, len(input_seq))
            logit_indices = [idx - 1 for idx in target_indices]
            
            valid = True
            if not logit_indices:
                valid = False
            elif logit_indices[0] < 0 or logit_indices[-1] >= logits.size(1):
                valid = False
            
            if valid:
                relevant_logits = logits[0, logit_indices, :]
                relevant_targets = torch.tensor(input_seq[target_start_idx:], dtype=torch.long, device=DEVICE)
                
                if relevant_logits.shape[0] == relevant_targets.shape[0]:
                    loss_val = F.cross_entropy(relevant_logits, relevant_targets).item()
                    all_losses.append(loss_val)
            
            context_ids.extend(new_ids)
            if len(context_ids) > MAX_CONTEXT_LEN:
                context_ids = context_ids[-MAX_CONTEXT_LEN:]
                
    return np.array(all_losses)


def get_session_max_losses_optimized(model, tokenizer, ids, data_df, batch_size=BATCH_SIZE):
    """Get max loss per session with batching and CPU throttling."""
    max_losses = []
    
    # Process in batches
    for batch_start in range(0, len(ids), batch_size):
        batch_end = min(batch_start + batch_size, len(ids))
        batch_ids = ids[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(ids)-1)//batch_size + 1} ({len(batch_ids)} sessions)")
        
        for tid in tqdm(batch_ids, desc=f"Batch {batch_start//batch_size + 1}"):
            s_df = data_df.filter(pl.col(SESSION_ID_COL) == tid)
            losses = collect_losses_batch(model, tokenizer, s_df, desc=None)
            if len(losses) > 0:
                max_losses.append(np.max(losses))
            else:
                max_losses.append(0.0)
        
        # CPU throttling: pause between batches
        if batch_end < len(ids):
            logger.info(f"⏸️  Pausing {CPU_DELAY}s to reduce CPU load...")
            time.sleep(CPU_DELAY)
            
    return np.array(max_losses)


def save_checkpoint(data):
    """Save calibration checkpoint."""
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"✅ Checkpoint saved")


def load_checkpoint():
    """Load checkpoint if exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return None


def main():
    logger.info("🚀 Optimized Adaptive Threshold Calibration (HDFS)")
    logger.info(f"   Sample Ratio: {SAMPLE_RATIO*100:.0f}%")
    logger.info(f"   Batch Size: {BATCH_SIZE}")
    logger.info(f"   CPU Delay: {CPU_DELAY}s between batches")
    
    set_seeds()
    
    # Check for checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        logger.info("📂 Resuming from checkpoint...")
        mean_loss = checkpoint['mean_loss']
        std_loss = checkpoint['std_loss']
        val_normal_max = checkpoint.get('val_normal_max')
        anomaly_max = checkpoint.get('anomaly_max')
        skip_collection = True
    else:
        skip_collection = False
    
    # 1. Load Resources
    logger.info("Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = torch.load(f"{MODEL_DIR}/config.pt", weights_only=False)
    if not hasattr(config, 'dropout'):
        config.dropout = 0.0
        
    model = LogGPT(config)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/hdfs_loggpt.pt", weights_only=False))
    model.to(DEVICE)
    model.eval()
    
    # 2. Data Split
    logger.info("Loading Data...")
    df = pl.read_csv(str(DATA_FILE), infer_schema_length=INFER_SCHEMA_LENGTH)
    
    normal_ids = df.filter(pl.col(LABEL_COL) == 0)[SESSION_ID_COL].unique().to_list()
    anom_ids = df.filter(pl.col(LABEL_COL) == 1)[SESSION_ID_COL].unique().to_list()
    
    # Split Normal: Train (80%), Test+Val (20%)
    train_ids, test_val_ids = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    val_ids, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)
    
    # Sample for faster calibration
    import random
    random.seed(42)
    val_ids_sample = random.sample(val_ids, int(len(val_ids) * SAMPLE_RATIO))
    anom_ids_sample = random.sample(anom_ids, int(len(anom_ids) * SAMPLE_RATIO))
    
    logger.info(f"Calibration Set (Sampled): {len(val_ids_sample)} Normal, {len(anom_ids_sample)} Anomaly Sessions")
    
    if not skip_collection:
        # 3. Collect Normal Losses
        val_normal_df = df.filter(pl.col(SESSION_ID_COL).is_in(val_ids_sample))
        logger.info("Step 1/3: Collecting normal validation losses...")
        normal_losses = collect_losses_batch(model, tokenizer, val_normal_df, desc="Normal Validation")
        
        mean_loss = np.mean(normal_losses)
        std_loss = np.std(normal_losses)
        
        logger.info(f"📈 Normal Loss Stats: Mean={mean_loss:.4f}, Std={std_loss:.4f}")
        logger.info(f"   Min={np.min(normal_losses):.4f}, Max={np.max(normal_losses):.4f}")
        
        # Save intermediate checkpoint
        save_checkpoint({
            'mean_loss': mean_loss,
            'std_loss': std_loss
        })
        
        # 4. Get Session-Level Max Losses
        logger.info("Step 2/3: Computing session-level max losses (Normal)...")
        val_normal_max = get_session_max_losses_optimized(model, tokenizer, val_ids_sample, val_normal_df)
        
        logger.info("Step 3/3: Computing session-level max losses (Anomaly)...")
        anom_df = df.filter(pl.col(SESSION_ID_COL).is_in(anom_ids_sample))
        anomaly_max = get_session_max_losses_optimized(model, tokenizer, anom_ids_sample, anom_df)
        
        # Save full checkpoint
        save_checkpoint({
            'mean_loss': mean_loss,
            'std_loss': std_loss,
            'val_normal_max': val_normal_max,
            'anomaly_max': anomaly_max
        })
    
    # 5. Grid Search for Optimal K
    best_f1 = 0
    best_k = 0
    best_th = 0
    best_precision = 0
    best_recall = 0
    
    logger.info("🔍 Grid Search for Optimal K...")
    
    for k in np.arange(0.0, 20.0, 0.2):  # Extended range for HDFS
        th = mean_loss + k * std_loss
        
        tp = np.sum(anomaly_max > th)
        fn = len(anomaly_max) - tp
        fp = np.sum(val_normal_max > th)
        tn = len(val_normal_max) - fp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 >= best_f1:
            best_f1 = f1
            best_k = k
            best_th = th
            best_precision = precision
            best_recall = recall
            
    logger.info(f"🏆 Best Result: K={best_k:.1f} | Threshold={best_th:.4f}")
    logger.info(f"   F1={best_f1:.4f} | Precision={best_precision:.4f} | Recall={best_recall:.4f}")
    
    # 6. Save Config
    config_data = {
        "mean_loss": float(mean_loss),
        "std_loss": float(std_loss),
        "k_sigma": float(best_k),
        "threshold": float(best_th),
        "f1_score": float(best_f1),
        "precision": float(best_precision),
        "recall": float(best_recall),
        "method": "adaptive_sigma_optimized",
        "sample_ratio": SAMPLE_RATIO
    }
    
    with open("threshold_config.json", "w") as f:
        json.dump(config_data, f, indent=4)
        
    logger.info("✅ Configuration saved to threshold_config.json")
    
    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logger.info("🗑️  Checkpoint cleaned up")


if __name__ == "__main__":
    main()
