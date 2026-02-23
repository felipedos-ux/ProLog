from pathlib import Path
import os
import torch
import random
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# DATASET COUNT-BASED
DATA_FILE = DATA_DIR / "BGL_count_train.csv"
MODEL_DIR = PROJECT_ROOT / "05_loggpt_bgl" / "model_weights_bgl_count"
OUTPUT_DIR = MODEL_DIR

# Ensure dirs exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Training Config
MODEL_NAME = "distilgpt2" 
BLOCK_SIZE = 64 # Janelas de 20 logs * media tokens (~3) = 60. 64 é seguro.
BATCH_SIZE = 128 # Windows curtíssimas, podemos aumentar Batch.
EPOCHS = 3
LEARNING_RATE = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
VOCAB_BUFFER = 1000  
DROPOUT = 0.1
SEED = 42

# Data Processing
SKIP_START_LOGS = 0 
INFER_SCHEMA_LENGTH = 10000 
LOG_DESC_MAX_LEN = 200

# Column Names
SESSION_ID_COL = "node_id" 
LABEL_COL = "label"
TIMESTAMP_COL = "timestamp"

# Val/Test split
TEST_SIZE_NORMAL = 0.2
TEST_SIZE_VAL_SPLIT = 0.5

def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
