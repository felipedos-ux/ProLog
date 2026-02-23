
import torch
import torch.nn.functional as F
import polars as pl
import numpy as np
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from model import LogGPT
from config import (
    MODEL_NAME, MODEL_DIR, DEVICE, DATA_FILE, 
    SKIP_START_LOGS, INFER_SCHEMA_LENGTH, 
    TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT,
    SESSION_ID_COL, LABEL_COL, TIMESTAMP_COL,
    set_seeds
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# OTIMIZAÇÃO MÁXIMA: Batch size grande para saturar GPU
INFERENCE_BATCH_SIZE = 128  # Processa 128 logs simultaneamente

class LogInferenceDataset(Dataset):
    """Dataset para inferência batched de logs."""
    def __init__(self, log_sequences, max_len=128):
        self.sequences = log_sequences
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Pad ou truncate
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
        return torch.tensor(seq, dtype=torch.long)

def collate_fn(batch):
    """Collate com padding dinâmico."""
    max_len = max(len(x) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded

def get_session_max_loss_ultra_fast(
    model, 
    tokenizer, 
    session_ids: list,
    data_df: pl.DataFrame,
    batch_size: int = INFERENCE_BATCH_SIZE
) -> np.ndarray:
    """
    Versão ULTRA otimizada: Processa múltiplos logs de múltiplas sessões simultaneamente.
    """
    max_losses = []
    MAX_CONTEXT_LEN = model.config.block_size
    
    for tid in tqdm(session_ids, desc="Ultra-Fast Session Processing"):
        s_df = data_df.filter(pl.col(SESSION_ID_COL) == tid)
        s_df = s_df.sort(TIMESTAMP_COL)
        templates = s_df["EventTemplate"].to_list()
        
        # Preparar todas as sequências da sessão
        sequences = []
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
            
            # Criar sequência de input
            full_seq = context_ids + new_ids
            if len(full_seq) > MAX_CONTEXT_LEN:
                input_seq = full_seq[-MAX_CONTEXT_LEN:]
            else:
                input_seq = full_seq
            
            sequences.append((input_seq, len(context_ids), new_ids))
            
            # Atualizar contexto
            context_ids.extend(new_ids)
            if len(context_ids) > MAX_CONTEXT_LEN:
                context_ids = context_ids[-MAX_CONTEXT_LEN:]
        
        # Processar em batch
        if len(sequences) == 0:
            max_losses.append(0.0)
            continue
        
        session_losses = []
        
        # Criar dataset e dataloader
        input_seqs = [s[0] for s in sequences]
        dataset = LogInferenceDataset(input_seqs, MAX_CONTEXT_LEN)
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
        
        batch_idx = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                logits, _ = model(batch)
                
                # Processar cada item do batch
                for i in range(batch.size(0)):
                    global_idx = batch_idx + i
                    if global_idx >= len(sequences):
                        break
                    
                    input_seq, target_start_idx, new_ids = sequences[global_idx]
                    
                    # Calcular loss
                    target_indices = range(target_start_idx, len(input_seq))
                    logit_indices = [idx - 1 for idx in target_indices]
                    
                    if logit_indices and logit_indices[0] >= 0 and logit_indices[-1] < logits.size(1):
                        relevant_logits = logits[i, logit_indices, :]
                        relevant_targets = torch.tensor(input_seq[target_start_idx:], dtype=torch.long, device=DEVICE)
                        
                        if relevant_logits.shape[0] == relevant_targets.shape[0]:
                            loss_val = F.cross_entropy(relevant_logits, relevant_targets).item()
                            session_losses.append(loss_val)
                
                batch_idx += batch.size(0)
        
        if len(session_losses) > 0:
            max_losses.append(np.max(session_losses))
        else:
            max_losses.append(0.0)
    
    return np.array(max_losses)

def main():
    logger.info("🚀 ULTRA-FAST Calibration (GPU 100% Saturation Mode)")
    set_seeds()
    
    logger.info("Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = torch.load(f"{MODEL_DIR}/config.pt", weights_only=False)
    if not hasattr(config, 'dropout'): config.dropout = 0.0
    
    model = LogGPT(config)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/loggpt_weights.pt", weights_only=False))
    model.to(DEVICE)
    model.eval()
    
    logger.info("Loading Data...")
    df = pl.read_csv(str(DATA_FILE), infer_schema_length=INFER_SCHEMA_LENGTH)
    
    normal_ids = df.filter(pl.col(LABEL_COL) == 0)[SESSION_ID_COL].unique().to_list()
    anom_ids = df.filter(pl.col(LABEL_COL) == 1)[SESSION_ID_COL].unique().to_list()
    
    train_ids, test_val_ids = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    val_ids, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)
    
    logger.info(f"Calibration: {len(val_ids)} Normal, {len(anom_ids)} Anomalies")
    
    val_normal_df = df.filter(pl.col(SESSION_ID_COL).is_in(val_ids))
    anom_df = df.filter(pl.col(SESSION_ID_COL).is_in(anom_ids))
    
    logger.info(f"📊 Collecting Max Losses (Batch={INFERENCE_BATCH_SIZE})...")
    val_normal_max = get_session_max_loss_ultra_fast(model, tokenizer, val_ids, val_normal_df, INFERENCE_BATCH_SIZE)
    anomaly_max = get_session_max_loss_ultra_fast(model, tokenizer, anom_ids, anom_df, INFERENCE_BATCH_SIZE)
    
    mean_loss = np.mean(val_normal_max)
    std_loss = np.std(val_normal_max)
    
    logger.info(f"📈 Stats: Mean={mean_loss:.4f}, Std={std_loss:.4f}")
    
    best_f1 = 0
    best_k = 0
    best_th = 0
    
    logger.info("🔍 Grid Search...")
    for k in np.arange(0.0, 10.0, 0.1):
        th = mean_loss + k * std_loss
        tp = np.sum(anomaly_max > th)
        fn = len(anomaly_max) - tp
        fp = np.sum(val_normal_max > th)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 >= best_f1:
            best_f1 = f1
            best_k = k
            best_th = th
    
    logger.info(f"🏆 Best: K={best_k:.1f}, Threshold={best_th:.4f}, F1={best_f1:.4f}")
    
    config_data = {
        "mean_loss": float(mean_loss),
        "std_loss": float(std_loss),
        "k_sigma": float(best_k),
        "threshold": float(best_th),
        "method": "adaptive_sigma_ultra_fast"
    }
    
    with open("threshold_config.json", "w") as f:
        json.dump(config_data, f, indent=4)
    
    logger.info("✅ Saved to threshold_config.json")

if __name__ == "__main__":
    main()
