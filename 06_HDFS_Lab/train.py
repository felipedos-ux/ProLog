"""
Training Script - Phase 1: Language Modeling (Causal LM).
Supports configurable architecture for A/B testing different model sizes.
"""
import os
import time
import math
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from dataset import prepare_llm_dataset
from model import LogGPT, GPTConfig
from config import (
    MODEL_NAME, MODELS_DIR, BLOCK_SIZE, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, DEVICE, VOCAB_BUFFER, DROPOUT, SEED, PATIENCE,
    ARCH_DEFAULT, set_seeds
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


def train_epoch(model, loader, optimizer, device, epoch_idx):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        _, loss = model(input_ids, targets=labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / steps if steps > 0 else 0.0


def evaluate_epoch(model, loader, device):
    """Evaluates the model on validation set."""
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in loader:
            idx = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)
            logits, loss, _ = model(idx, targets)
            total_loss += loss.item()
            steps += 1

    return total_loss / steps if steps > 0 else 0.0


def train_model(
    experiment_id: str,
    arch_config: dict = None,
    deduplicate: bool = False,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    patience: int = PATIENCE,
    is_regex: bool = False,
):
    """
    Full Phase 1 training pipeline.
    
    Args:
        experiment_id: e.g. 'A', 'B', etc. Used for model saving.
        arch_config: dict with n_layer, n_head, n_embd. Defaults to ARCH_DEFAULT.
        deduplicate: whether to deduplicate train sessions.
        epochs: max training epochs.
        lr: learning rate.
        patience: early stopping patience.
    
    Returns:
        model: trained LogGPT model
        config: GPTConfig used
        train_meta: dict with training metadata
    """
    set_seeds()
    arch = arch_config or ARCH_DEFAULT
    
    logger.info("=" * 60)
    logger.info(f"PHASE 1: PRETRAINING | Experiment {experiment_id}")
    logger.info(f"Architecture: {arch['name']} ({arch['n_embd']}d / {arch['n_layer']}L / {arch['n_head']}H)")
    logger.info("=" * 60)
    
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    
    # Extract dynamic architecture params
    arch_block_size = arch.get('block_size', BLOCK_SIZE)
    arch_batch_size = arch.get('batch_size', BATCH_SIZE)
    
    # 2. Dataset
    lm_datasets, data_meta = prepare_llm_dataset(
        tokenizer,
        block_size=arch_block_size,
        deduplicate=deduplicate,
        is_regex=is_regex,
    )
    
    # Split train/val
    split = lm_datasets.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = split["train"]
    val_dataset = split["test"]
    
    logger.info(f"Train chunks: {len(train_dataset)}, Val chunks: {len(val_dataset)}")
    
    # DataLoaders
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(train_dataset, batch_size=arch_batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=arch_batch_size, shuffle=False, collate_fn=data_collator)
    
    # 3. Model
    config = GPTConfig(
        vocab_size=vocab_size + VOCAB_BUFFER,
        block_size=arch_block_size,
        n_layer=arch['n_layer'],
        n_head=arch['n_head'],
        n_embd=arch['n_embd'],
        dropout=DROPOUT,
    )
    model = LogGPT(config)
    model.to(DEVICE)
    
    param_count = model.count_params()
    logger.info(f"Model params: {param_count / 1e6:.2f}M")
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # 5. Training Loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_history = []
    
    exp_model_dir = MODELS_DIR / f"exp_{experiment_id.lower()}"
    os.makedirs(exp_model_dir, exist_ok=True)
    
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        avg_train_loss = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
        avg_val_loss = evaluate_epoch(model, val_loader, DEVICE)
        ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float('inf')
        
        train_history.append({
            'epoch': epoch,
            'train_loss': round(avg_train_loss, 4),
            'val_loss': round(avg_val_loss, 4),
            'ppl': round(ppl, 2) if ppl != float('inf') else 'inf',
        })
        
        logger.info(f"Epoch {epoch}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | PPL: {ppl:.2f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), str(exp_model_dir / "model.pt"))
            torch.save(config, str(exp_model_dir / "config.pt"))
            logger.info(f"  ✅ Best model saved!")
        else:
            patience_counter += 1
            logger.info(f"  ⚠️ No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            logger.info(f"  🛑 Early stopping at epoch {epoch}")
            break
    
    train_time = time.time() - t0
    
    # Load best model
    model.load_state_dict(torch.load(str(exp_model_dir / "model.pt"), weights_only=False))
    
    train_meta = {
        'experiment_id': experiment_id,
        'arch': arch,
        'param_count': param_count,
        'param_count_M': round(param_count / 1e6, 2),
        'epochs_run': len(train_history),
        'best_val_loss': round(best_val_loss, 4),
        'train_time_seconds': round(train_time, 1),
        'data_meta': data_meta,
        'history': train_history,
    }
    
    # Save training metadata
    with open(str(exp_model_dir / "train_meta.json"), 'w') as f:
        json.dump(train_meta, f, indent=2, default=str)
    
    logger.info(f"✅ Phase 1 complete in {train_time:.0f}s | Best val loss: {best_val_loss:.4f}")
    
    return model, config, train_meta


if __name__ == '__main__':
    model, config, meta = train_model('test', arch_config=ARCH_DEFAULT)
    print(json.dumps(meta, indent=2, default=str))
