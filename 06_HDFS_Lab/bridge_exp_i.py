import os
import shutil
import json
import torch
from pathlib import Path

from config import MODELS_DIR, EXPERIMENTS, ARCH_LARGE_CONTEXT
from model import GPTConfig

def bridge_contrastive_to_eval(exp_name='I'):
    """
    Copies the loggpt_exp_i_contrastive.pt model into the standardized format
    saved_models/exp_i/model.pt along with config.pt and train_meta.json
    so that run_experiments.py can seamlessly evaluate it.
    """
    contrastive_model_path = MODELS_DIR / f"loggpt_exp_{exp_name.lower()}_contrastive.pt"
    
    if not contrastive_model_path.exists():
        print(f"Error: {contrastive_model_path} doesn't exist yet! Training must finish first.")
        return False
        
    exp_dir = MODELS_DIR / f"exp_{exp_name.lower()}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # 1. Copy model.pt
    dest_model_path = exp_dir / "model.pt"
    shutil.copy(contrastive_model_path, dest_model_path)
    print(f"Copied {contrastive_model_path.name} -> exp_{exp_name.lower()}/model.pt")
    
    # 2. Re-create and save config.pt
    # In train_contrastive.py we used:
    # config = GPTConfig(vocab_size=vocab_size + 1, block_size=block_size, n_layer=arch['n_layer'], ...)
    # vocab_size is distilgpt2's vocab size which is 50257. +1 = 50258
    config = GPTConfig(
        vocab_size=50258,
        block_size=1024,
        n_layer=4,
        n_head=4,
        n_embd=256
    )
    torch.save(config, str(exp_dir / "config.pt"))
    print(f"Saved synthetic config.pt")
    
    # 3. Create dummy train_meta.json
    train_meta = {
        'experiment_id': exp_name,
        'arch': ARCH_LARGE_CONTEXT,
        'param_count_M': 29.15,
        'epochs_run': 30,
        'best_val_loss': 0.0,
        'train_time_seconds': 0.0,
        'data_meta': {'type': 'contrastive'},
    }
    with open(exp_dir / "train_meta.json", 'w') as f:
        json.dump(train_meta, f, indent=2)
    print("Saved placeholder train_meta.json")
    
    print("Bridge complete! You can now run `python run_experiments.py --experiment I`.")
    return True

if __name__ == '__main__':
    bridge_contrastive_to_eval('I')
