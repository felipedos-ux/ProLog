
import sys
import os
import torch
import shutil
from pathlib import Path

# Add current dir to path
sys.path.append(os.getcwd())

from config import DATA_FILE, MODEL_DIR, DEVICE, BATCH_SIZE
from dataset import load_bgl_data, prepare_llm_dataset
from model import LogGPT, GPTConfig
from train_custom import main as train_main
from transformers import AutoTokenizer

def test_pipeline():
    print("🚀 TEsting BGL Pipeline Integrity...")
    
    # 1. Test Data Loading
    print("\n[1/4] Testing Data Loading...")
    try:
        df = load_bgl_data()
        print(f"✅ Data Loaded: {len(df)} logs")
        print(f"   Columns: {df.columns}")
        if "node_id" not in df.columns or "label" not in df.columns:
            raise ValueError("❌ Missing required BGL columns (node_id, label)")
    except Exception as e:
        print(f"❌ Data Loading Failed: {e}")
        return

    # 2. Test Tokenization & Dataset Prep
    print("\n[2/4] Testing Dataset Prep (Small Subset)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Mock small dataset
        ds = prepare_llm_dataset(tokenizer, block_size=32)
        print(f"✅ Dataset Created: {ds}")
    except Exception as e:
        print(f"❌ Dataset Prep Failed: {e}")
        # Print full traceback in real scenario
        import traceback
        traceback.print_exc()
        return

    # 3. Test Model Init
    print("\n[3/4] Testing Model Params...")
    try:
        config = GPTConfig(vocab_size=50300, block_size=32, n_layer=2, n_head=2, n_embd=128)
        model = LogGPT(config)
        model.to(DEVICE)
        print("✅ Model Initialized on", DEVICE)
    except Exception as e:
        print(f"❌ Model Init Failed: {e}")
        return

    print("\n✅ PIPELINE INTEGRITY CHECK PASSED!")
    print("Ready to run full training.")

if __name__ == "__main__":
    test_pipeline()
