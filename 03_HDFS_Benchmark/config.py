from pathlib import Path
import os
import torch

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HDFS_DIR = DATA_DIR / "HDFS"
DATA_FILE = HDFS_DIR / "HDFS_data_processed.csv"
MODEL_DIR = Path(__file__).parent / "saved_models"

# Source files for preprocessing
EVENT_TRACES_FILE = HDFS_DIR / "preprocessed" / "Event_traces.csv"
TEMPLATES_FILE = HDFS_DIR / "HDFS_full.log_templates.csv"
LABELS_FILE = HDFS_DIR / "anomaly_label.csv"

# Ensure dirs exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Training Config
MODEL_NAME = "distilgpt2"
BLOCK_SIZE = 128
BATCH_SIZE = 64  # Increased for RTX 3080 Ti (12GB VRAM)
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Config (same as OpenStack)
VOCAB_BUFFER = 100
N_LAYER = 4
N_HEAD = 4
N_EMBD = 256
DROPOUT = 0.1

# Detection Config
THRESHOLD = 5.0  # Default fallback
SKIP_START_LOGS = 3
LOG_DESC_MAX_LEN = 50
INFER_SCHEMA_LENGTH = 10000
TEST_SIZE_NORMAL = 0.2
TEST_SIZE_VAL_SPLIT = 0.5

# Session / Label column names (generic interface)
SESSION_ID_COL = "session_id"   # BlockId mapped to this
TIMESTAMP_COL = "timestamp"
TEMPLATE_COL = "EventTemplate"
LABEL_COL = "anom_label"

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
