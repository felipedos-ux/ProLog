
import polars as pl
from transformers import AutoTokenizer
import torch
from dataset import prepare_llm_dataset, load_bgl_data
from model import LogGPT, GPTConfig
from detect_custom import evaluate_session
from config import MODEL_DIR, DEVICE, MODEL_NAME, SESSION_ID_COL, LABEL_COL
import shutil
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split

def run_optimized_training():
    print("🚀 STARTING OPTIMIZED BGL PIPELINE (Smart Downsampling)")
    
    # 0. Cleanup
    if os.path.exists(MODEL_DIR):
        try:
            shutil.rmtree(MODEL_DIR)
        except:
             print("⚠️ Warning: Could not clear model dir (files in use?)")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load Data with Downsampling
    print("\n📦 Loading Data...")
    df = load_bgl_data()
    
    # Get ALL Anomalies (Critical to keep 100% pattern coverage)
    anom_ids = df.filter(pl.col(LABEL_COL) == 1)[SESSION_ID_COL].unique().to_list()
    print(f"🔹 Anomalies found: {len(anom_ids)} sessions (Keeping 100%)")
    
    # Get Subset of Normals (Downsampling)
    # Strategy: Random sample 2000 normal sessions (enough to learn patterns)
    all_normal_ids = df.filter(pl.col(LABEL_COL) == 0)[SESSION_ID_COL].unique().to_list()
    
    if len(all_normal_ids) > 2000:
        import random
        random.seed(42)
        normal_ids = random.sample(all_normal_ids, 2000)
        print(f"🔹 Normals downsampled: {len(all_normal_ids)} -> {len(normal_ids)} sessions")
    else:
        normal_ids = all_normal_ids
        
    # Filter DF
    dataset_ids = set(normal_ids + anom_ids)
    subset_df = df.filter(pl.col(SESSION_ID_COL).is_in(dataset_ids))
    
    # Save temp CSV for dataset.py compatibility
    print("💾 Saving optimized dataset buffer...")
    subset_df.write_csv("temp_optimized.csv")
    
    # 2. Tokenize
    print("\n🔤 Tokenizing (DistilGPT2)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare only NORMAL data for training (Unsupervised)
    # We need to filter prepare_llm_dataset to only use normals, 
    # but prepare_llm_dataset in dataset.py calls load_bgl_data which loads FILE.
    # So we should save ONLY normals to the temp file for training prep.
    
    subset_df.filter(pl.col(LABEL_COL) == 0).write_csv("temp_train_optimized.csv")
    
    # Import locally to use patched file path
    from dataset import prepare_llm_dataset
    lm_datasets = prepare_llm_dataset(tokenizer, data_path="temp_train_optimized.csv")
    
    # 3. Train (10 Epochs - should be fast now)
    print("\n🏋️ Training (10 Epochs)...")
    config = GPTConfig(
        vocab_size=len(tokenizer)+100, 
        block_size=128, 
        n_layer=4, 
        n_head=4, 
        n_embd=256,
        dropout=0.1
    )
    model = LogGPT(config)
    model.to(DEVICE)
    
    # Save config
    torch.save(config, f"{MODEL_DIR}/config.pt")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4) # Slightly higher LR for smaller dataset
    model.train()
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(lm_datasets["train"], batch_size=32, shuffle=True)
    
    print(f"   Batches per epoch: {len(train_loader)}")
    
    for epoch in range(10):
        epoch_loss = 0
        model.train()
        start = time.time()
        
        for batch in train_loader:
            input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
            labels = torch.tensor(batch["labels"]).to(DEVICE)
            
            logits, loss = model(input_ids, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"   Epoch {epoch+1}/10 | Loss: {avg_loss:.4f} | Time: {time.time()-start:.1f}s")
        
    torch.save(model.state_dict(), f"{MODEL_DIR}/loggpt_weights.pt")
    print("✅ Model Saved.")
    
    # 4. Calibration (Use Val set from Normals)
    print("\n⚖️ Calibrating Adaptive Threshold...")
    model.eval()
    
    # Split Normals: Train (used above) vs Val for calibration
    # We used all 2000 for train? Oops. We should have split.
    # Let's take a fresh 200 normals from the ORIGINAL pool for calibration (unseen)
    remaining_normals = list(set(all_normal_ids) - set(normal_ids))
    if len(remaining_normals) > 200:
         val_ids = random.sample(remaining_normals, 200)
    else:
         val_ids = normal_ids[:200] # Fallback (optimistic bias)
         
    val_df = df.filter(pl.col(SESSION_ID_COL).is_in(val_ids))
    
    # Calculate Mean/Std of Loss on Val
    from calibrate_adaptive import collect_losses
    losses = collect_losses(model, tokenizer, val_df, desc="Calibrating")
    
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    k_sigma = 3.0 # Standard
    threshold = mean_loss + k_sigma * std_loss
    
    print(f"   Stats: Mean={mean_loss:.4f}, Std={std_loss:.4f} => Th={threshold:.4f}")
    
    import json
    with open("threshold_config.json", "w") as f:
        json.dump({"threshold": threshold, "k_sigma": k_sigma, "mean": mean_loss, "std": std_loss}, f)
        
    # 5. Full Evaluation (All Anomalies + 2000 Normals)
    print("\n🔎 Evaluating...")
    # Test set: All Anom + 2000 Normals (can use same as train for "train acc" or new ones)
    # Log anomaly detection usually evaluates on "Test" set.
    # Let's pick ANOTHER 2000 normals for pure Test
    test_normal_ids = remaining_normals[:2000] if len(remaining_normals) > 2000 else remaining_normals
    
    eval_ids = anom_ids + test_normal_ids
    print(f"   Eval Set: {len(anom_ids)} Anomalies + {len(test_normal_ids)} Normals")
    
    results = []
    
    # Pre-filter DF for speed
    eval_df = df.filter(pl.col(SESSION_ID_COL).is_in(eval_ids))
    
    for tid in eval_ids:
        # Determine label (slow logic, better map)
        # We know anom_ids
        label = 1 if tid in anom_ids else 0
        s_df = eval_df.filter(pl.col(SESSION_ID_COL) == tid)
        
        res = evaluate_session(tid, label, s_df, model, tokenizer, threshold, DEVICE)
        if res:
            results.append(res)
            
    # Metrics
    tp = sum(1 for r in results if r['label']==1 and r['is_detected'])
    fp = sum(1 for r in results if r['label']==0 and r['is_detected'])
    fn = sum(1 for r in results if r['label']==1 and not r['is_detected'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n📊 FINAL OPTIMIZED RESULTS:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    with open("optimized_results.txt", "w") as f:
        f.write(f"Precision: {precision}\nRecall: {recall}\nF1: {f1}\nTP: {tp}\nFP: {fp}")

if __name__ == "__main__":
    run_optimized_training()
