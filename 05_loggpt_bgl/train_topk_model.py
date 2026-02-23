"""
Treina LogGPT com vocabulário de templates (Top-K approach)

Diferenças da abordagem anterior:
- Vocab = 322 templates (vs 50k tokens BPE)
- Cada template é um token único
- Modelo aprende transições entre templates
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path

# Importar componentes
import sys
sys.path.append(str(Path(__file__).parent / "universal_detector"))
from universal_detector.model import LogGPT, GPTConfig
from topk_dataset import BGLTemplateDataset
import json

# Configuração
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path(r"D:\ProLog\data\BGL_sliding_windows")
VOCAB_PATH = "bgl_template_vocab.json"
OUTPUT_DIR = Path(r"D:\ProLog\08_loggpt_topk")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Hiperparâmetros  
LEARNING_RATE = 1e-4  # LR inicial para treino from scratch
BATCH_SIZE = 32
EPOCHS = 50  # Menos epochs já que vocab é menor
BLOCK_SIZE = 19  # 20 templates - 1 (input vs target)

# Arquitetura (mesma do LogGPT-Small)
N_LAYER = 4
N_HEAD = 4
N_EMBD = 256

def train():
    print("=" * 80)
    print("🚀 Training LogGPT with Template Vocabulary (Top-K approach)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print()
    
    # 1. Carregar vocabulário
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    VOCAB_SIZE = vocab_data['vocab_size']
    print(f"📚 Vocabulary:")
    print(f"   Total templates: {vocab_data['num_real_templates']}")
    print(f"   Vocab size (with special tokens): {VOCAB_SIZE}")
    print(f"   K for Top-K: {vocab_data['num_real_templates'] // 2}")
    print()
    
    # 2. Criar datasets
    print("📂 Loading datasets...")
    train_dataset = BGLTemplateDataset(
        DATA_DIR / "train.parquet",
        VOCAB_PATH,
        block_size=BLOCK_SIZE,
        mode="train"
    )
    
    val_dataset = BGLTemplateDataset(
        DATA_DIR / "val.parquet",
        VOCAB_PATH,
        block_size=BLOCK_SIZE,
        mode="val"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n📦 Data loaded:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print()
    
    # 3. Criar modelo
    print("🔧 Initializing model...")
    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD
    )
    model = LogGPT(config)
    model.to(DEVICE)
    model.train()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    print()
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training Loop
    print(f"🎯 Starting training for {EPOCHS} epochs...")
    print()
    
    best_val_loss = float('inf')
    model_save_path = OUTPUT_DIR / "loggpt_topk_best.pt"
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            logits, loss = model(input_ids, targets=targets)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                targets = batch['targets'].to(DEVICE)
                
                logits, loss = model(input_ids, targets=targets)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Salvar melhor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"   ✅ Best model saved (val loss: {best_val_loss:.4f})")
        
        # Checkpoint a cada 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = OUTPUT_DIR / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"   💾 Checkpoint saved")
    
    print("\n" + "=" * 80)
    print("🎉 Training Complete!")
    print("=" * 80)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {model_save_path}")
    print()
    
    return model_save_path

if __name__ == "__main__":
    model_path = train()
    print(f"✅ Model ready for Top-K detection at: {model_path}")
