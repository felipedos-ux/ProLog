"""
Fine-tuning LogGPT-Small no BGL Dataset (Sliding Window)

Baseado no paper LogGPT (arxiv):
- Learning Rate: 1e-6 (fine-tuning)
- Batch Size: 16
- Epochs: 100
- Training Data: 5,000 sequências normais

Estratégia:
1. Carregar LogGPT-Small pré-treinado no OpenStack
2. Fine-tune em janelas normais do BGL (sliding window, 20 eventos)
3. Salvar modelo fine-tuned para avaliação
"""

import torch
import polars as pl
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from pathlib import Path

# Importar componentes do LogGPT
import sys
sys.path.append(str(Path(__file__).parent / "universal_detector"))
from universal_detector.model import LogGPT, GPTConfig
from universal_detector.dataset import get_tokenizer

# Configuração
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path(r"D:\ProLog\data\BGL_sliding_windows")
PRETRAINED_MODEL = r"D:\ProLog\02_loggpt_small\model_weights\loggpt_weights.pt"
OUTPUT_DIR = Path(r"D:\ProLog\07_loggpt_small_bgl_finetuned")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Hiperparâmetros do paper LogGPT
LEARNING_RATE = 1e-6  # Fine-tuning LR
BATCH_SIZE = 16
EPOCHS = 100
MAX_TRAIN_SAMPLES = 5000  # 5k sequências normais

# Arquitetura LogGPT-Small
N_LAYER = 4
N_HEAD = 4
N_EMBD = 256
BLOCK_SIZE = 128
VOCAB_SIZE = 50357

class BGLSlidingDataset(Dataset):
    """Dataset para janelas sliding do BGL"""
    
    def __init__(self, df, tokenizer, max_samples=None):
        """
        Args:
            df: DataFrame Polars com colunas [window_id, sequence, label]
            tokenizer: Tokenizer do GPT2
            max_samples: Limite de amostras (None = todas)
        """
        self.tokenizer = tokenizer
        
        # Filtrar apenas janelas normais (label=0)
        normal_df = df.filter(pl.col("label") == 0)
        
        if max_samples:
            normal_df = normal_df.head(max_samples)
        
        self.sequences = []
        for row in normal_df.iter_rows(named=True):
            # Sequência de templates
            seq = row['sequence']
            self.sequences.append(seq)
        
        print(f"   Loaded {len(self.sequences)} normal sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Concatenar templates com espaço
        text = " ".join(seq)
        
        # Tokenizar
        ids = self.tokenizer.encode(text)
        
        # Truncar ou pad para BLOCK_SIZE
        if len(ids) > BLOCK_SIZE:
            ids = ids[:BLOCK_SIZE]
        else:
            # Pad com token de padding (assumindo que vocab tem padding token)
            ids = ids + [self.tokenizer.eos_token_id] * (BLOCK_SIZE - len(ids))
        
        input_ids = torch.tensor(ids, dtype=torch.long)
        
        # Para Causal LM, labels = input_ids shifted
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }

def load_pretrained_model(device):
    """Carrega LogGPT-Small pré-treinado no OpenStack"""
    print(f"🔄 Loading pre-trained LogGPT-Small from {PRETRAINED_MODEL}...")
    
    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD
    )
    model = LogGPT(config)
    
    # Carregar pesos pré-treinados
    state_dict = torch.load(PRETRAINED_MODEL, map_location=device, weights_only=True)
    
    # Remover prefixo "module." se existir
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    print("✅ Pre-trained model loaded successfully.")
    return model

def train():
    print("=" * 80)
    print("🚀 Fine-tuning LogGPT-Small on BGL Dataset")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Max Training Samples: {MAX_TRAIN_SAMPLES}")
    print()
    
    # 1. Carregar dados
    print("📂 Loading BGL sliding window data...")
    train_df = pl.read_parquet(DATA_DIR / "train.parquet")
    print(f"   Total training windows: {len(train_df)}")
    print(f"   Normal windows: {len(train_df.filter(pl.col('label') == 0))}")
    print(f"   Anomalous windows: {len(train_df.filter(pl.col('label') == 1))}")
    
    # 2. Criar dataset
    tokenizer = get_tokenizer()
    dataset = BGLSlidingDataset(train_df, tokenizer, max_samples=MAX_TRAIN_SAMPLES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"\n📦 Dataset created:")
    print(f"   Training samples: {len(dataset)}")
    print(f"   Batches per epoch: {len(loader)}")
    print(f"   Total training steps: {len(loader) * EPOCHS}")
    
    # 3. Carregar modelo pré-treinado
    model = load_pretrained_model(DEVICE)
    model.train()
    
    # 4. Optimizer (AdamW com LR do paper)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training Loop
    print(f"\n🎯 Starting fine-tuning for {EPOCHS} epochs...")
    print()
    
    best_loss = float('inf')
    model_save_path = OUTPUT_DIR / "loggpt_small_bgl_finetuned.pt"
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, loss = model(input_ids, targets=labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")
        
        # Salvar melhor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"   ✅ Best model saved (loss: {best_loss:.4f})")
        
        # Salvar checkpoint a cada 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = OUTPUT_DIR / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path}")
    
    print("\n" + "=" * 80)
    print("🎉 Fine-tuning Complete!")
    print("=" * 80)
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Model saved to: {model_save_path}")
    print()
    
    return model_save_path

if __name__ == "__main__":
    model_path = train()
    print(f"✅ Fine-tuned model ready at: {model_path}")
