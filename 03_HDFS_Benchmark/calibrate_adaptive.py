"""
Adaptive Threshold Calibration.
Identical logic to OpenStack: collects cross-entropy losses on normal validation sessions,
then grid-searches K (sigma multiplier) to optimize F1 against anomaly sessions.
"""

import torch
import torch.nn.functional as F
import polars as pl
import numpy as np
import json
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


def collect_losses(model, tokenizer, sessions_df, desc="Collecting Losses"):
    """Collects cross-entropy losses for every log in the provided sessions."""
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


def get_session_max_losses(model, tokenizer, ids, data_df):
    """Get max loss per session for threshold optimization."""
    max_losses = []
    for tid in tqdm(ids, desc="Session Max Loss"):
        s_df = data_df.filter(pl.col(SESSION_ID_COL) == tid)
        losses = collect_losses(model, tokenizer, s_df, desc=None)
        if len(losses) > 0:
            max_losses.append(np.max(losses))
        else:
            max_losses.append(0.0)
    return np.array(max_losses)


def main():
    logger.info("🚀 Starting Adaptive Threshold Calibration")
    set_seeds()
    
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
    
    logger.info(f"Calibration Set: {len(val_ids)} Normal, {len(anom_ids)} Anomaly Sessions")
    
    # 3. Collect Normal Losses
    val_normal_df = df.filter(pl.col(SESSION_ID_COL).is_in(val_ids))
    normal_losses = collect_losses(model, tokenizer, val_normal_df, desc="Scanning Normal Validation")
    
    mean_loss = np.mean(normal_losses)
    std_loss = np.std(normal_losses)
    
    logger.info(f"📈 Normal Loss Stats: Mean={mean_loss:.4f}, Std={std_loss:.4f}")
    logger.info(f"   Min={np.min(normal_losses):.4f}, Max={np.max(normal_losses):.4f}")
    
    # 4. Get Session-Level Max Losses
    logger.info("Computing Session-Level Max Losses...")
    val_normal_max = get_session_max_losses(model, tokenizer, val_ids, val_normal_df)
    
    anom_df = df.filter(pl.col(SESSION_ID_COL).is_in(anom_ids))
    anomaly_max = get_session_max_losses(model, tokenizer, anom_ids, anom_df)
    
    # 5. Grid Search for Optimal K
    best_f1 = 0
    best_k = 0
    best_th = 0
    
    logger.info("🔍 Grid Search for Optimal K...")
    
    for k in np.arange(0.0, 10.0, 0.1):
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
            
    logger.info(f"🏆 Best Result: K={best_k:.1f} | Threshold={best_th:.4f} | F1={best_f1:.4f}")
    
    # 6. Save Config
    config_data = {
        "mean_loss": float(mean_loss),
        "std_loss": float(std_loss),
        "k_sigma": float(best_k),
        "threshold": float(best_th),
        "method": "adaptive_sigma"
    }
    
    with open("threshold_config.json", "w") as f:
        json.dump(config_data, f, indent=4)
        
    logger.info("✅ Configuration saved to threshold_config.json")


if __name__ == "__main__":
    main()
