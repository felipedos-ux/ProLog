# -*- coding: utf-8 -*-
"""
SIAT Configuration
==================
Hyperparameters and paths for SIAT benchmark.
"""
from pathlib import Path
import torch

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "model_output"
SESSION_DATA = DATA_DIR / "siat_sessions.pkl"
CHECKPOINT_PATH = OUTPUT_DIR / "siat_loggpt.pt"


# Create dirs
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Data Config
BLOCK_SIZE = 128            # Max session length (SIAT avg is 72, so 128 is plenty)
VOCAB_SIZE_ESTIMATE = 500   # Pre-processing showed 371 + buffer
VOCAB_BUFFER = 50           # Safety margin

# Model Config (LogGPT-Small)
N_LAYER = 4
N_HEAD = 4
N_EMBD = 256
DROPOUT = 0.1

# Training Config
BATCH_SIZE = 64             # SIAT is small, can probably fit more but 64 is safe
EPOCHS = 100                # Small dataset converges fast, but needs more epochs
LEARNING_RATE = 5e-4        # Higher LR for smaller dataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
EARLY_STOPPING_PATIENCE = 10
