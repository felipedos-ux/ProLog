import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import math
from tqdm import tqdm
from utils.logger import setup_logger
from config import EXPERIMENTS, ARCH_DEFAULT, VOCAB_PATH, TRAIN_DATA_REGEX_PATH, MODELS_DIR
from model import GPTConfig, LogGPT
from dataset_contrastive import get_dataloaders

# Contrastive InfoNCE Loss
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        """
        z_i, z_j: Respresentations of view 1 and view 2. Shape: (batch_size, hidden_dim)
        """
        batch_size = z_i.shape[0]
        
        # Normalize representations
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        
        # Concatenate all representations in the batch (2N)
        z = torch.cat([z_i, z_j], dim=0) # Shape: (2 * batch_size, hidden_dim)
        
        # Compute similarity matrix
        sim_matrix = self.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        # Mask out self-similarity (the diagonal)
        sim_matrix.fill_diagonal_(-1e9)
        
        # Labels for contrastive task:
        # z_i is matched with z_j (which is at index i + batch_size)
        # z_j is matched with z_i (which is at index i)
        labels = torch.cat([torch.arange(batch_size, 2*batch_size), 
                          torch.arange(batch_size)], dim=0).to(z_i.device)
                          
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

def train_contrastive_model(exp_name='I'):
    """
    Trains the LogGPT model using a combined LM Loss + InfoNCE Loss
    """
    logger = setup_logger('train_contrastive')
    exp_config = EXPERIMENTS[exp_name]
    arch = exp_config['arch']
    
    block_size = arch.get('block_size', 256)
    batch_size = arch.get('batch_size', 64)
    lambda_contrastive = 0.5 # Weight of the contrastive loss
    
    logger.info("="*60)
    logger.info(f"PHASE 3: CONTRASTIVE PRETRAINING | Experiment {exp_name}")
    logger.info(f"Architecture: {exp_config.get('arch_name', 'default_large_context')} | lambda: {lambda_contrastive}")
    logger.info("="*60)

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader, vocab_size = get_dataloaders(
        data_path=TRAIN_DATA_REGEX_PATH, 
        batch_size=batch_size,
        block_size=block_size,
        max_sessions=5000
    )
    
    # Needs +1 for potential <MASK> token fallback logic in vocab counting
    config = GPTConfig(
        vocab_size=vocab_size + 1, 
        block_size=block_size,
        n_layer=arch['n_layer'],
        n_head=arch['n_head'],
        n_embd=arch['n_embd']
    )
    model = LogGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    contrastive_criterion = NTXentLoss(temperature=0.1).to(device)
    
    logger.info(f"Train chunks: {len(train_loader.dataset)}")
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    epochs = 30
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3
    
    model_save_path = os.path.join(MODELS_DIR, f"loggpt_exp_{exp_name.lower()}_contrastive.pt")

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_lm_loss = 0
        total_cont_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            
            # --- Mixed Precision Forward Pass ---
            with torch.amp.autocast('cuda'):
                # --- Language Modeling Task ---
                idx = batch['input_ids'].to(device)
                targets = batch['labels'].to(device)
                
                logits, lm_loss, _ = model(idx, targets)
                
                # --- Contrastive Task ---
                v1_idx = batch['view1'].to(device)
                v2_idx = batch['view2'].to(device)
                
                # Forward pass to get hidden states
                _, _, hidden1 = model(v1_idx) # (batch, seq_len, embd)
                _, _, hidden2 = model(v2_idx)
                
                z1 = hidden1.mean(dim=1) # (batch, embd)
                z2 = hidden2.mean(dim=1) # (batch, embd)
                
                cont_loss = contrastive_criterion(z1, z2)
                
                # --- Combined Loss ---
                loss = lm_loss + (lambda_contrastive * cont_loss)
            
            # --- Scaled Backward Pass ---
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            total_lm_loss += lm_loss.item()
            total_cont_loss += cont_loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.3f}", 'lm': f"{lm_loss.item():.3f}", 'nce': f"{cont_loss.item():.3f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        avg_lm = total_lm_loss / len(train_loader)
        avg_cont = total_cont_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                idx = batch['input_ids'].to(device)
                targets = batch['labels'].to(device)
                with torch.amp.autocast('cuda'):
                    logits, v_lm_loss, _ = model(idx, targets)
                
                # In validation we primarily track LM loss for early stopping predictability
                total_val_loss += v_lm_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float('inf')
        
        log_msg = f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f} (LM:{avg_lm:.2f} Cont:{avg_cont:.2f}) | Val LM: {avg_val_loss:.4f} | PPL: {ppl:.2f}"
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            logger.info(log_msg + " |   ✅ Best model saved!")
        else:
            patience_counter += 1
            logger.info(log_msg)
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

if __name__ == '__main__':
    train_contrastive_model('I')
