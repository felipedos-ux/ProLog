
import polars as pl
from transformers import AutoTokenizer
import torch
from model import LogGPT, GPTConfig
from config import MODEL_DIR, DEVICE, MODEL_NAME
import shutil
import os
import numpy as np
import time

# Custom Config for 2k
DATA_FILE_2K = "D:/ProLog/data/BGL_2k.log_structured.csv"
BLOCK_SIZE = 128
SESSION_COL = "Node"
LABEL_COL = "Label"
TEMPLATE_COL = "EventTemplate"
TIMESTAMP_COL = "Timestamp"

def run_2k_pipeline():
    print("🚀 STARTING BGL 2K PIPELINE (Proof of Concept)")
    
    # 0. Cleanup
    if os.path.exists(MODEL_DIR):
        try: shutil.rmtree(MODEL_DIR)
        except: pass
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load Data
    print(f"\n📦 Loading {DATA_FILE_2K}...")
    try:
        df = pl.read_csv(DATA_FILE_2K, infer_schema_length=0)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Filter Normals vs Anomalies
    # Normal = '-'
    # Anomaly = anything else
    
    # Add binary label for easier processing
    df = df.with_columns(
        pl.when(pl.col(LABEL_COL) == "-").then(0).otherwise(1).alias("binary_label")
    )
    
    normal_df = df.filter(pl.col("binary_label") == 0)
    anom_df = df.filter(pl.col("binary_label") == 1)
    
    print(f"   Total: {len(df)}")
    print(f"   Normals: {len(normal_df)}")
    print(f"   Anomalies: {len(anom_df)}")
    
    # 2. Prepare Sessions (Group by Node)
    print("\n🔄 Grouping by Node...")
    
    def df_to_sessions(dataframe):
        try:
            sessions = (
                dataframe.sort(TIMESTAMP_COL)
                .group_by(SESSION_COL)
                .agg([
                    pl.col(TEMPLATE_COL),
                    pl.col("binary_label").max().alias("session_label")
                ])
            )
            return sessions
        except Exception as e:
            print(f"Grouping Error: {e}")
            return None

    all_sessions_agg = df_to_sessions(df)
    if all_sessions_agg is None: return

    # Filter sessions with label 0 for train
    train_sessions_rows = all_sessions_agg.filter(pl.col("session_label") == 0).rows()
    # Filter sessions with label 1 for test
    test_sessions_rows = all_sessions_agg.filter(pl.col("session_label") == 1).rows()
    
    # Also add some normal sessions to test to measure FP
    # Split train_sessions
    import random
    random.seed(42)
    random.shuffle(train_sessions_rows)
    
    num_train = int(len(train_sessions_rows) * 0.8)
    train_data = train_sessions_rows[:num_train]
    val_data   = train_sessions_rows[num_train:] # Use for calibration & test FP
    
    print(f"   Train Sessions: {len(train_data)}")
    print(f"   Val/Normal Test Sessions: {len(val_data)}")
    print(f"   Anomaly Test Sessions: {len(test_sessions_rows)}")
    
    # 3. Tokenization & Dataset Prep
    print("\n🔤 Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Helper to create HF dataset style dict
    def create_dataset(rows):
        texts = []
        for r in rows:
            templates = r[1] # col 1 is Template list
            if len(templates) < 2: continue
            text = " \n ".join(templates)
            texts.append(text)
        return texts
        
    train_texts = create_dataset(train_data)
    
    # We need to manually tokenize and chunk
    from torch.utils.data import Dataset, DataLoader
    
    class LogDataset(Dataset):
        def __init__(self, texts, tokenizer, block_size):
            self.examples = []
            if not texts: return
            
            encodings = tokenizer(texts, truncation=False, padding=False)
            
            for input_ids in encodings["input_ids"]:
                # Chunking
                for i in range(0, len(input_ids), block_size):
                    chunk = input_ids[i:i + block_size]
                    if len(chunk) < 5: continue 
                    if len(chunk) < block_size:
                        chunk = chunk + [tokenizer.pad_token_id] * (block_size - len(chunk))
                    self.examples.append(torch.tensor(chunk))
                    
        def __len__(self):
            return len(self.examples)
            
        def __getitem__(self, i):
            return self.examples[i]
            
    train_ds = LogDataset(train_texts, tokenizer, BLOCK_SIZE)
    print(f"   Train Blocks: {len(train_ds)}")
    
    if len(train_ds) == 0:
        print("❌ Error: Not enough training data (blocks). Sessions too short?")
        return

    # 4. Train
    print("\n🏋️ Training (10 Epochs)...")
    config = GPTConfig(
        vocab_size=len(tokenizer)+100, 
        block_size=BLOCK_SIZE, 
        n_layer=2, n_head=2, n_embd=128
    )
    model = LogGPT(config)
    model.to(DEVICE)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    
    for epoch in range(10):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            targets = batch.clone()
            logits, loss = model(batch, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), f"{MODEL_DIR}/loggpt_weights.pt")
    
    # 5. Calibration (Val Data)
    print("\n⚖️ Calibrating...")
    model.eval()
    
    def get_session_loss(templates):
        text = " \n ".join(templates)
        tokens = tokenizer.encode(text)
        if len(tokens) < 2: return 0.0
        
        input_ids = torch.tensor(tokens).to(DEVICE).unsqueeze(0)
        
        # SLIDING WINDOW LOGIC (Explicit loop)
        chunk_losses = []
        for i in range(0, input_ids.size(1), BLOCK_SIZE):
            chunk = input_ids[:, i:i + BLOCK_SIZE]
            # Skip tiny leftovers if they are mostly padding or single token
            if chunk.size(1) < 2: continue
            
            with torch.no_grad():
                # Pass chunk as both input and target
                _, loss = model(chunk, chunk)
                chunk_losses.append(loss.item())
        
        if not chunk_losses: return 0.0
        return np.mean(chunk_losses)
            
    val_session_losses = []
    for r in val_data:
        templates = r[1]
        loss = get_session_loss(templates)
        if loss > 0: val_session_losses.append(loss)
        
    if not val_session_losses:
        print("⚠️ No val losses collected.")
        threshold = 5.0
    else:
        mean, std = np.mean(val_session_losses), np.std(val_session_losses)
        threshold = mean + 3 * std
        print(f"   Threshold: {threshold:.4f} (Mean {mean:.4f} + 3*Std {std:.4f})")
        
    # 6. Evaluation
    print("\n🔎 Evaluating...")
    
    detected_anoms = 0
    total_anoms = 0
    
    for r in test_sessions_rows:
        templates = r[1]
        loss = get_session_loss(templates)
        total_anoms += 1
        if loss > threshold:
            detected_anoms += 1
            print(f"   [ANOMALY DETECTED] Node:{r[0]} Loss:{loss:.4f}")
        else:
            print(f"   [FN] Node:{r[0]} Loss:{loss:.4f}")
            
    fp = 0
    total_norm = 0
    for r in val_data:
        templates = r[1]
        loss = get_session_loss(templates)
        total_norm += 1
        if loss > threshold:
            fp += 1
            
    print(f"\n📊 RESULTS:")
    print(f"   Anomalies Detected (TP): {detected_anoms}/{total_anoms}")
    print(f"   False Positives (FP):    {fp}/{total_norm}")
    
    if total_anoms > 0:
        recall = detected_anoms / total_anoms
        print(f"   Recall: {recall:.4f}")
    
    with open("results_2k.txt", "w") as f:
        f.write(f"TP: {detected_anoms}\nTotal Anoms: {total_anoms}\nFP: {fp}\nTotal Norm: {total_norm}\nThreshold: {threshold}")

if __name__ == "__main__":
    run_2k_pipeline()
