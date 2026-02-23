from pathlib import Path
import torch

# PORTS - Paths
DATA_DIR = Path("D:/ProLog/data")
HDFS_DIR = DATA_DIR / "HDFS"
MODEL_DIR = Path("D:/ProLog/06_loggpt_hdfs/saved_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# PARSING
# We use the provided RAW logs
LOG_FILE = HDFS_DIR / "HDFS_full.log"
TEMPLATE_FILE = HDFS_DIR / "HDFS_templates.csv"
LABEL_FILE = HDFS_DIR / "anomaly_label.csv"  # Required for ground truth
# STRUCTURED_FILE is removed as we parse raw logs

 # Still checking if this exists
STRUCTURED_FILE = HDFS_DIR / "HDFS_full.log_structured.csv" 
TEMPLATE_FILE = HDFS_DIR / "HDFS_full.log_templates.csv" # From user

# DATASET
BLOCK_SIZE = 64 # HDFS sequences are shorter than BGL generally
TRAIN_SIZE = 0.8
SEED = 42

# MODEL
MODEL_NAME = "gpt2" # Base config
N_LAYER = 4 # Smaller model for HDFS (vocab is small)
N_HEAD = 4
N_EMBD = 256
DROPOUT = 0.1

# TRAINING
BATCH_SIZE = 32
EPOCHS = 10 # HDFS converges fast
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seeds():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
