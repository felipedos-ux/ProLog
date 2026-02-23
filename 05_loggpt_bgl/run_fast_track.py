
import polars as pl
from transformers import AutoTokenizer
import torch
from dataset import prepare_llm_dataset, load_bgl_data
from model import LogGPT, GPTConfig
from train_custom import Trainer
from detect_custom import evaluate_session
from config import MODEL_DIR, DEVICE, MODEL_NAME, SESSION_ID_COL, LABEL_COL
import shutil
import os
import numpy as np

def run_fast_track():
    print("🚀 STARTING FAST TRACK VALIDATION (Proof of Concept)")
    
    # 0. Cleanup old weights
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load Tiny Subset of Data
    print("\n📦 Loading Subset Data...")
    df = load_bgl_data() # Loads everything
    
    # Take first 500 Normal sessions for Train
    normal_ids = df.filter(pl.col(LABEL_COL) == 0)[SESSION_ID_COL].unique().head(500).to_list()
    
    # Take 50 Anomaly sessions for Test
    anom_ids = df.filter(pl.col(LABEL_COL) == 1)[SESSION_ID_COL].unique().head(50).to_list()
    
    subset_df = df.filter(pl.col(SESSION_ID_COL).is_in(normal_ids + anom_ids))
    print(f"✅ Subset Logic: {len(normal_ids)} train sessions, {len(anom_ids)} test sessions.")
    
    # Save subset to disk effectively for the "dataset.py" which might reload it?
    # Actually, let's just patch the functions or pass df directly implies refactoring.
    # Easiest: Overwrite load_bgl_data to return subset? No.
    # Let's manually run the steps here instead of calling scripts.
    
    # 2. Prepare Dataset
    print("\n🔤 Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # We need to hack prepare_llm_dataset to accept DF or save temp CSV
    # Let's save a temp CSV
    subset_df.write_csv("temp_subset.csv")
    
    # Imports inside check config
    # We need to monkeypatch DATA_FILE in config?
    # Easier: Just use the function exposed
    from dataset import prepare_llm_dataset
    lm_datasets = prepare_llm_dataset(tokenizer, data_path="temp_subset.csv")
    
    # 3. Train (1 Epoch)
    print("\n🏋️ Training (1 Epoch)...")
    config = GPTConfig(
        vocab_size=len(tokenizer)+100, 
        block_size=128, 
        n_layer=2,  # Tiny model for speed
        n_head=2, 
        n_embd=128
    )
    model = LogGPT(config)
    model.to(DEVICE)
    
    # Save config
    torch.save(config, f"{MODEL_DIR}/config.pt")
    
    # Simplified Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(lm_datasets["train"], batch_size=4, shuffle=True)
    
    for i, batch in enumerate(train_loader):
        input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
        labels = torch.tensor(batch["labels"]).to(DEVICE)
        
        logits, loss = model(input_ids, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"   Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        if i > 50: break # Super fast track
            
    torch.save(model.state_dict(), f"{MODEL_DIR}/loggpt_weights.pt")
    print("✅ Model Saved.")
    
    # 4. Calibrate (Mock)
    print("\n⚖️ Calibrating...")
    # Just use a fixed threshold for demo
    threshold = 5.0
    import json
    with open("threshold_config.json", "w") as f:
        json.dump({"threshold": threshold, "k_sigma": 0, "method": "fast_track"}, f)
        
    # 5. Detect
    print("\n🔎 Detecting...")
    results = []
    
    # Evaluate 10 anomalies and 10 normals
    test_ids = anom_ids[:10] + normal_ids[:10]
    
    for tid in test_ids:
        s_df = subset_df.filter(pl.col(SESSION_ID_COL) == tid)
        label = 1 if tid in anom_ids else 0
        res = evaluate_session(tid, label, s_df, model, tokenizer, threshold, DEVICE)
        if res:
            results.append(res)
            
    print(f"\n🎉 Processed {len(results)} sessions.")
    tp = sum(1 for r in results if r['label']==1 and r['is_detected'])
    fp = sum(1 for r in results if r['label']==0 and r['is_detected'])
    print(f"TP: {tp} | FP: {fp}")
    
    with open("fast_track_results.txt", "w") as f:
        f.write(f"Fast Track Complete.\nTP: {tp}\nFP: {fp}\nTotal: {len(results)}")

if __name__ == "__main__":
    run_fast_track()
