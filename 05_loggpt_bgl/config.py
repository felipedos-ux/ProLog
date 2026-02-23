from pathlib import Path
import os
import torch

# Paths
# Assumes structure: ProLog/
#   05_loggpt_bgl/config.py
#   data/BGL_processed.csv
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "BGL_processed.csv"  # ← MUDANÇA: BGL ao invés de OpenStack
MODEL_DIR = Path(__file__).parent / "model_weights_bgl"  # ← MUDANÇA: pasta específica BGL

# Ensure dirs exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Training Config (MANTIDO)
MODEL_NAME = "distilgpt2" # For tokenizer
BLOCK_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE =5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Config (MANTIDO)
VOCAB_BUFFER = 100 # Safety buffer for tokenizer vocab
N_LAYER = 4
N_HEAD = 4
N_EMBD = 256
DROPOUT = 0.1 # Regularization

# Detection Config (MANTIDO)
THRESHOLD = 5.0 # Default fallback
SKIP_START_LOGS = 3 # Skip first N logs to avoid cold start issues
LOG_DESC_MAX_LEN = 50 # For reporting
INFER_SCHEMA_LENGTH = 10000 # For polars read_csv
TEST_SIZE_NORMAL = 0.2
TEST_SIZE_VAL_SPLIT = 0.5

# BGL-Specific Config (NOVO)
SESSION_MAX_LOGS = 500  # Limite de logs por sessão (node_id pode ser muito longo)
SESSION_ID_COL = "node_id"  # BGL usa node_id como proxy de sessão
LABEL_COL = "label"  # BGL usa 'label' não 'anom_label'
TIMESTAMP_COL = "timestamp"  # BGL usa 'timestamp' (Unix epoch)

# Seeds (MANTIDO)
SEED = 42

def set_seeds():
    import random
    import numpy as np
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
