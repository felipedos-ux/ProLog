"""
RL Trainer - Phase 2: PPO Finetuning for Log Anomaly Detection.
Based on LogGPT (arXiv:2309.14482) PPO approach.

The goal: teach the model that correct Top-K predictions matter,
not just minimizing cross-entropy loss.
"""
import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from model import LogGPT
from config import (
    MODEL_NAME, MODELS_DIR, BLOCK_SIZE, BATCH_SIZE, DEVICE,
    RL_LEARNING_RATE, RL_EPISODES, RL_CLIP_EPSILON, RL_EARLY_STOP_THRESHOLD,
    SEED, set_seeds
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PPOTrainer:
    """
    Proximal Policy Optimization for Log Anomaly Detection.
    Based on LogGPT paper (arXiv:2309.14482).
    
    Key idea: finetune GPT-2 so its Top-K predictions better match actual logs.
    Reward: +1 if actual next log is in Top-K predictions, -1 otherwise.
    """
    
    def __init__(self, model, k_top, lr_rl=RL_LEARNING_RATE, 
                 clip_epsilon=RL_CLIP_EPSILON, device=DEVICE):
        self.model = model
        self.k_top = k_top
        self.clip_epsilon = clip_epsilon
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr_rl)
    
    def compute_rewards_and_logprobs(self, input_ids):
        """
        Forward pass through the model to compute rewards and log probabilities.
        
        For each position t in the sequence:
        - Model predicts distribution over next token
        - Reward = +1 if actual next token is in Top-K predictions
        - Reward = -1 otherwise
        
        Args:
            input_ids: [batch_size, seq_len] tensor
        
        Returns:
            rewards: [total_positions] tensor
            log_probs: [total_positions] tensor (differentiable)
        """
        self.model.eval()
        batch_size, seq_len = input_ids.size()
        
        with torch.no_grad():
            logits, _ = self.model(input_ids)
        # logits: [batch_size, seq_len, vocab_size]
        
        rewards = []
        log_probs = []
        
        # For each position, check if next token is in Top-K
        for t in range(seq_len - 1):
            pred_logits = logits[:, t, :]  # [batch_size, vocab_size]
            actual_next = input_ids[:, t + 1]  # [batch_size]
            
            # Compute Top-K
            probs = torch.softmax(pred_logits, dim=-1)
            topk_indices = torch.topk(probs, self.k_top, dim=-1).indices  # [batch_size, k_top]
            
            # Check if actual is in Top-K
            is_in_topk = (topk_indices == actual_next.unsqueeze(-1)).any(dim=-1)  # [batch_size]
            
            # Reward: +1 if in Top-K, -1 otherwise
            reward = torch.where(
                is_in_topk,
                torch.tensor(1.0, device=self.device),
                torch.tensor(-1.0, device=self.device)
            )
            rewards.append(reward)
            
            # Log probability of actual action (for PPO ratio)
            action_probs = probs.gather(1, actual_next.unsqueeze(-1)).squeeze(-1)
            log_prob = torch.log(action_probs + 1e-10)
            log_probs.append(log_prob)
        
        rewards = torch.stack(rewards, dim=1)  # [batch_size, seq_len-1]
        log_probs = torch.stack(log_probs, dim=1)  # [batch_size, seq_len-1]
        
        return rewards, log_probs
    
    def ppo_update(self, input_ids, old_log_probs, rewards):
        """
        PPO update step.
        
        L^{CLIP}(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
        """
        self.model.train()
        
        # Forward pass with gradient
        logits, _ = self.model(input_ids)
        
        batch_size, seq_len = input_ids.size()
        new_log_probs = []
        
        for t in range(seq_len - 1):
            pred_logits = logits[:, t, :]
            actual_next = input_ids[:, t + 1]
            probs = torch.softmax(pred_logits, dim=-1)
            action_probs = probs.gather(1, actual_next.unsqueeze(-1)).squeeze(-1)
            log_prob = torch.log(action_probs + 1e-10)
            new_log_probs.append(log_prob)
        
        new_log_probs = torch.stack(new_log_probs, dim=1)  # [batch_size, seq_len-1]
        
        # Advantages (simplified: reward - mean)
        advantages = rewards - rewards.mean()
        
        # PPO ratio
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        
        # Clipped surrogate
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        
        # Loss (negative because we want to maximize)
        loss = -torch.min(surr1, surr2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_episode(self, dataloader, max_batches=None):
        """
        Train one RL episode (one pass through training data).
        
        Returns:
            avg_reward: average reward across all positions
            avg_loss: average PPO loss
        """
        total_reward = 0.0
        total_loss = 0.0
        total_positions = 0
        batch_count = 0
        
        for batch in dataloader:
            if max_batches and batch_count >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(self.device)
            
            # Step 1: Collect rewards and old log probs (no gradient)
            rewards, old_log_probs = self.compute_rewards_and_logprobs(input_ids)
            
            # Step 2: PPO update (with gradient)
            loss = self.ppo_update(input_ids, old_log_probs, rewards)
            
            total_reward += rewards.sum().item()
            total_positions += rewards.numel()
            total_loss += loss
            batch_count += 1
        
        avg_reward = total_reward / total_positions if total_positions > 0 else 0.0
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        
        return avg_reward, avg_loss
    
    def train(self, dataloader, num_episodes=RL_EPISODES, 
              early_stop_threshold=RL_EARLY_STOP_THRESHOLD,
              max_batches_per_episode=50):
        """
        Full RL training loop.
        
        Args:
            dataloader: training DataLoader
            num_episodes: max episodes (LogGPT uses 20)
            early_stop_threshold: stop when avg_reward >= threshold
            max_batches_per_episode: limit batches per episode for speed
        
        Returns:
            history: list of episode stats
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: RL FINETUNING WITH PPO")
        logger.info(f"K_top={self.k_top}, LR={self.optimizer.param_groups[0]['lr']}")
        logger.info(f"Episodes={num_episodes}, Clip_ε={self.clip_epsilon}")
        logger.info("=" * 60)
        
        history = []
        
        for ep in range(1, num_episodes + 1):
            avg_reward, avg_loss = self.train_episode(
                dataloader, max_batches=max_batches_per_episode
            )
            
            history.append({
                'episode': ep,
                'avg_reward': round(avg_reward, 4),
                'avg_loss': round(avg_loss, 4),
            })
            
            logger.info(
                f"Episode {ep}/{num_episodes} | "
                f"Avg Reward: {avg_reward:.4f} | "
                f"Loss: {avg_loss:.4f}"
            )
            
            if avg_reward >= early_stop_threshold:
                logger.info(f"✅ Early stopping: reward {avg_reward:.4f} >= {early_stop_threshold}")
                break
        
        logger.info("=" * 60)
        logger.info("✅ RL FINETUNING COMPLETE")
        logger.info("=" * 60)
        
        return history


def run_rl_finetuning(
    experiment_id: str,
    model: LogGPT,
    config,
    k_top: int,
    deduplicate: bool = False,
):
    """
    Complete Phase 2 RL pipeline.
    
    Args:
        experiment_id: experiment identifier
        model: pre-trained LogGPT model (from Phase 1)
        config: GPTConfig
        k_top: number of top predictions for reward
        deduplicate: whether training data was deduplicated
    
    Returns:
        model: RL-finetuned model
        rl_meta: training metadata
    """
    from dataset import prepare_llm_dataset
    
    set_seeds()
    
    logger.info(f"Starting RL finetuning for experiment {experiment_id}")
    logger.info(f"Top-K for reward: {k_top}")
    
    # Prepare dataloader (same data as phase 1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    lm_datasets, _ = prepare_llm_dataset(
        tokenizer,
        block_size=BLOCK_SIZE,
        deduplicate=deduplicate,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(
        lm_datasets, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator
    )
    
    # PPO Training
    t0 = time.time()
    
    ppo = PPOTrainer(
        model=model,
        k_top=k_top,
        lr_rl=RL_LEARNING_RATE,
        clip_epsilon=RL_CLIP_EPSILON,
        device=DEVICE,
    )
    
    history = ppo.train(
        train_loader,
        num_episodes=RL_EPISODES,
        early_stop_threshold=RL_EARLY_STOP_THRESHOLD,
        max_batches_per_episode=50,
    )
    
    rl_time = time.time() - t0
    
    # Save RL model
    exp_model_dir = MODELS_DIR / f"exp_{experiment_id.lower()}"
    os.makedirs(exp_model_dir, exist_ok=True)
    torch.save(model.state_dict(), str(exp_model_dir / "model_rl.pt"))
    
    rl_meta = {
        'k_top': k_top,
        'lr': RL_LEARNING_RATE,
        'episodes_run': len(history),
        'rl_time_seconds': round(rl_time, 1),
        'history': history,
        'final_avg_reward': history[-1]['avg_reward'] if history else 0.0,
    }
    
    with open(str(exp_model_dir / "rl_meta.json"), 'w') as f:
        json.dump(rl_meta, f, indent=2)
    
    logger.info(f"RL finetuning complete in {rl_time:.0f}s")
    
    return model, rl_meta
