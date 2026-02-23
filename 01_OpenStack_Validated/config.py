from pathlib import Path
import os
import torch

# Paths
# Assumes structure: ProLog/
#   02_loggpt_small/config.py
#   data/OpenStack_data_original.csv
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "OpenStack_data_original.csv"
MODEL_DIR = Path(__file__).parent / "models" / "loggpt_custom"

# Ensure dirs exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Training Config
MODEL_NAME = "gpt2"  # Same as HDFS
BLOCK_SIZE = 1024  # Max context for GPT2 (OpenStack sessions avg 494 logs)
BATCH_SIZE = 8   # Reduced to prevent OOM with 1024 block_size
EPOCHS = 10
LEARNING_RATE = 1e-4  # Same as HDFS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Config
VOCAB_BUFFER = 100 # Safety buffer for tokenizer vocab
N_LAYER = 4
N_HEAD = 4
N_EMBD = 256
DROPOUT = 0.1

# Detection Config
THRESHOLD = 5.0 # Default fallback
SKIP_START_LOGS = 1  # Reduced from 3: anomaly sessions avg 7 logs, can't skip 3
LOG_COLUMN = "EventId"  # Use EventId (1-2 tokens) instead of EventTemplate (113 tokens avg)
LOG_DESC_MAX_LEN = 50 # For reporting
INFER_SCHEMA_LENGTH = 10000 # For polars read_csv
TEST_SIZE_NORMAL = 0.2
TEST_SIZE_VAL_SPLIT = 0.5

# Seeds
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
