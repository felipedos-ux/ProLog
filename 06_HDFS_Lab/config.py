"""
Centralized configuration for HDFS Lab experiments.
Each experiment can override defaults via experiment-specific configs.
"""
from pathlib import Path
import os
import torch

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
LAB_ROOT = Path(__file__).parent

DATA_DIR = PROJECT_ROOT / "data"
HDFS_DIR = DATA_DIR / "HDFS"
SOURCE_DATA_FILE = HDFS_DIR / "HDFS_data_processed.csv"

# Lab-specific paths
LAB_DATA_DIR = LAB_ROOT / "data"
RESULTS_DIR = LAB_ROOT / "results"
MODELS_DIR = LAB_ROOT / "saved_models"

# Ensure dirs exist
for d in [LAB_DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Standard Path exports
VOCAB_PATH = LAB_DATA_DIR / "vocab.json"
TRAIN_DATA_PATH = LAB_DATA_DIR / "hdfs_train_5k.csv"
TEST_DATA_PATH = LAB_DATA_DIR / "hdfs_test_subset.csv"

# Phase 1/2 Regex specific paths
VOCAB_PATH = LAB_DATA_DIR / "vocab_regex.json"
TRAIN_DATA_REGEX_PATH = LAB_DATA_DIR / "hdfs_train_5k_regex.csv"
TEST_DATA_REGEX_PATH = LAB_DATA_DIR / "hdfs_test_subset_regex.csv"

# ============================================================
# COLUMN NAMES
# ============================================================
SESSION_ID_COL = "session_id"
TIMESTAMP_COL = "timestamp"
TEMPLATE_COL = "EventTemplate"
LABEL_COL = "anom_label"

# ============================================================
# DATA SAMPLING CONFIG
# ============================================================
SAMPLE_TRAIN_SESSIONS = 5000       # Normal sessions for training
SAMPLE_TEST_NORMAL = 1000          # Normal sessions for testing
SAMPLE_SEED = 42
INFER_SCHEMA_LENGTH = 10000

# ============================================================
# TRAINING DEFAULTS (Fase 1: Language Modeling)
# ============================================================
MODEL_NAME = "distilgpt2"          # Tokenizer
BLOCK_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
PATIENCE = 3                       # Early stopping (fast experiments)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ============================================================
# ARCHITECTURE CONFIGS (switchable per experiment)
# ============================================================
VOCAB_BUFFER = 100
DROPOUT = 0.1

# Default: Our current architecture
ARCH_DEFAULT = {
    'name': 'default_256d',
    'n_layer': 4,
    'n_head': 4,
    'n_embd': 256,
    'batch_size': 64,
}

# Extended Context (1024 tokens)
ARCH_LARGE_CONTEXT = {
    'name': 'large_context_256d',
    'n_layer': 4,
    'n_head': 4,
    'n_embd': 256,
    'block_size': 1024,
    'batch_size': 16, # reduced to avoid OOM
}

# LogGPT-style: Smaller architecture
ARCH_LOGGPT = {
    'name': 'loggpt_60d',
    'n_layer': 6,
    'n_head': 6,
    'n_embd': 60,
    'batch_size': 64,
}

# ============================================================
# DETECTION CONFIGS
# ============================================================
SKIP_START_LOGS = 3
LOG_DESC_MAX_LEN = 50

# Top-K detection
TOP_K_RATIO = 0.5                  # K = 50% of unique keys (LogGPT paper)

# Threshold detection (baseline)
DEFAULT_THRESHOLD = 5.0

# ============================================================
# RL CONFIG (Fase 2: PPO Finetuning)
# ============================================================
RL_LEARNING_RATE = 1e-6            # 100x smaller than pretraining
RL_EPISODES = 20
RL_CLIP_EPSILON = 0.2
RL_EARLY_STOP_THRESHOLD = 0.95

# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================
EXPERIMENTS = {
    'A': {
        'name': 'Baseline 5k',
        'description': 'Pipeline atual com subset 5k sessões',
        'arch': ARCH_DEFAULT,
        'detection': 'threshold',
        'deduplicate': False,
        'use_rl': False,
        'is_regex': False,
    },
    'B': {
        'name': 'Top-K Detection',
        'description': 'Detecção por Top-K (K=50% unique keys) estilo LogGPT',
        'arch': ARCH_DEFAULT,
        'detection': 'topk',
        'deduplicate': False,
        'use_rl': False,
        'is_regex': False,
    },
    'C': {
        'name': 'Deduplication',
        'description': 'Deduplicação de sessões estilo SiaLog + Top-K',
        'arch': ARCH_DEFAULT,
        'detection': 'topk',
        'deduplicate': True,
        'use_rl': False,
        'is_regex': False,
    },
    'D': {
        'name': 'Small Architecture',
        'description': 'Arquitetura LogGPT (60d/6L/6H) + Top-K',
        'arch': ARCH_LOGGPT,
        'detection': 'topk',
        'deduplicate': False,
        'use_rl': False,
        'is_regex': False,
    },
    'E': {
        'name': 'RL PPO',
        'description': 'RL Finetuning com PPO + Top-K',
        'arch': ARCH_DEFAULT,
        'detection': 'topk',
        'deduplicate': False,
        'use_rl': True,
        'is_regex': False,
    },
    'F': {
        'name': 'Dedupe + Threshold (Final)',
        'description': 'Combinação: Deduplicação + Threshold GPT-2. Metodologia Validada.',
        'arch': ARCH_DEFAULT,
        'detection': 'threshold',
        'deduplicate': True,
        'use_rl': False,
        'is_regex': False,
    },
    'G': {
        'name': 'Regex Mínimo SOTA',
        'description': 'Phase 1 Otimizações: Regex Minimo + Treino Normal + Threshold',
        'arch': ARCH_DEFAULT,
        'detection': 'threshold',
        'deduplicate': False,
        'use_rl': False,
        'is_regex': True,
    },
    'H': {
        'name': 'Extended Context SOTA',
        'description': 'Phase 2: Regex Minimo + Block Size 1024 + Adaptive Threshold (k=2.5)',
        'arch': ARCH_LARGE_CONTEXT,
        'detection': 'threshold',
        'deduplicate': False,
        'use_rl': False,
        'is_regex': True,
        'k_sigma': 2.5,
    },
    'I': {
        'name': 'Self-Supervised Contrastive',
        'description': 'Phase 3: Regex + 1024 BS + InfoNCE Data Augmentation (HDFS pure)',
        'arch': ARCH_LARGE_CONTEXT,
        'detection': 'threshold',
        'deduplicate': False,
        'use_rl': False,
        'is_regex': True,
        'k_sigma': 2.5,
    },
}


def set_seeds(seed=SEED):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
