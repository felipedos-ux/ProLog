import torch
import torch.nn.functional as F
import polars as pl
import numpy as np
import json
import argparse
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import config_count as config
from model import LogGPT
from utils.logger import setup_logger

# CONFIG
MODEL_PATH = config.MODEL_DIR 
DEVICE = config.DEVICE
BATCH_SIZE = config.BATCH_SIZE
K_PROD = 5
FREQ_FILE = config.MODEL_DIR / "template_freq.json"
RARE_THRESHOLD = 5 

logger = setup_logger(__name__)

def load_resources():
    logger.info(f"Loading Resources...")
    
    # Model
    config_path = MODEL_PATH / "config.pt"
    model_config = torch.load(config_path, weights_only=False)
    model = LogGPT(model_config)
    
    weights_path = MODEL_PATH / "loggpt_weights.pt"
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    
    # Freq Map
    if not FREQ_FILE.exists():
        raise FileNotFoundError("Frequency map not found. Run train_frequency.py first.")
    with open(FREQ_FILE, 'r') as f:
        freq_map = json.load(f)
        
    return model, freq_map

def get_validation_data():
    logger.info(f"Loading raw data from {config.DATA_FILE}...") # Using Train file but filtering Val split logic?
    # Actually, config_count.py defines how to split.
    # We should reuse the logic to ensure we test on the same VAL set.
    # However, for "Official Result", maybe we should use the separate "BGL_processed.csv" 
    # and split consistent with training.
    
    # Reusing logic from detect_count_validation.py to be consistent
    import detect_count_validation
    return detect_count_validation.load_and_preprocess(limit=None) # Load ALL validation data

def is_rare(template, freq_map):
    t_str = str(template)
    if t_str not in freq_map:
        return True
    return freq_map[t_str]['count'] < RARE_THRESHOLD

def evaluate_benchmark(model, freq_map, windows):
    logger.info(f"🚀 Running Official Benchmark on {len(windows)} windows...")
    
    # Storage for predictions
    y_true = []
    y_pred_std = []   # Standard K=5
    y_pred_hybrid = [] # Hybrid
    
    for i in tqdm(range(0, len(windows), BATCH_SIZE), desc="Benchmarking"):
        batch = windows[i:i+BATCH_SIZE]
        
        # Prepare inputs
        input_ids = torch.stack([x['input_ids'] for x in batch]).to(DEVICE)
        labels = [x['label'] for x in batch]
        
        # We need raw templates for Hybrid check
        # But windows dict from detect_count_validation might not have them if using tensors directly?
        # detect_count_validation.preprocess_windows stores 'template_ids' in input_ids, 
        # but 'EventTemplate' was tokenized.
        # LogGPT inputs are token IDs, not template IDs directly unless vocab is 1:1.
        # In this project, we tokenized templates as text?
        # Let's check `dataset_bgl_count.py` -> tokenizer.encode(text). 
        # Wait, if we use Tokenizer, we can't easily map back to "Template ID" for frequency check
        # unless we kept the raw templates in the window object.
        # `detect_count_validation` preprocess DOES NOT keep raw templates inside the 'input_ids' tensor.
        # We need to modify the data loader or use the Token IDs as proxy?
        # NO, frequency map is on "EventTemplate" strings.
        # THE HYBRID SCRIPT `deploy_hybrid_prod.py` kept 'templates' in the dict.
        # We need that here.
        
        y_true.extend(labels)
        
        # 1. MODEL INFERENCE (Standard)
        with torch.no_grad():
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            logits, _ = model(inputs)
            
            probs = F.softmax(logits, dim=-1)
            _, topk_indices = torch.topk(probs, k=K_PROD, dim=-1)
            
            targets_expanded = targets.unsqueeze(-1)
            matches = (topk_indices == targets_expanded)
            hits = matches.any(dim=-1)
            misses = ~hits
            num_misses = misses.sum(dim=1) # [B]
            
            # Standard Prediction
            preds_std_batch = (num_misses > 0).cpu().numpy().astype(int)
            y_pred_std.extend(preds_std_batch)
            
            # 2. HYBRID INFERENCE
            # We need to check if ANY template in the window is rare.
            # Issue: 'input_ids' are sub-word tokens from GPT2 tokenizer, NOT distinct template IDs.
            # We cannot check frequency of sub-word 345 vs 212.
            # Frequency Check MUST happen on the raw Template level before tokenization.
            
            # Since we can't easily reverse tokenization to exact template IDs for frequency check here,
            # AND the `detect_count_validation` loader doesn't store raw templates...
            
            # WE NEED TO RELOAD DATA with Raw Templates.
            # Or assume Hybrid = Standard (since Standard is already 87% F1).
            # But user wants "Improvement".
            # The current Validation script only saved tensors.
            
            # If I want to benchmark Hybrid, I need the raw templates.
            # I will assume for this benchmark script, we will skip the exact "template string" frequency check
            # and rely on the Model's "Unseen Token" implicit penalty (which fits into Standard).
            # OR, I admit I can't run Hybrid Benchmark easily without refactoring the Loader.
            
            # WAIT. `deploy_hybrid_prod.py` loads CSV and keeps `chunk`.
            # I should use that logic.
            
    return y_true, y_pred_std

from transformers import AutoTokenizer

# REDEFINING BENCHMARK TO USE RAW DATA LOADING
def full_benchmark_pipeline():
    model, freq_map = load_resources()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME) # Fix
    tokenizer.pad_token = tokenizer.eos_token
    
    # LOAD VAL DATA (Raw)
    # Reuse `detect_count_validation.load_and_preprocess` but we need to intercept before tokenization?
    # No, let's just use the `deploy_hybrid_prod.py` logic but with Labels.
    # We need to read `BGL_count_train.csv` (since it has labels) and filter for VALIDATION set.
    
    logger.info("Regenerating Validation Set with Raw Templates...")
    import train_count
    # train_count.prepare_llm_dataset splits by Session ID.
    # We need to replicate that Split completely to identify Val Sessions.
    
    RAW_DATA = config.DATA_DIR / "BGL_processed.csv"
    logger.info(f"Loading Raw Data from {RAW_DATA}...")
    df = pl.read_csv(str(RAW_DATA), infer_schema_length=10000)
    session_ids = df[config.SESSION_ID_COL].unique().to_list()
    # SEED is critical here.
    import random
    random.seed(config.SEED)
    random.shuffle(session_ids)
    
    n_train = int(len(session_ids) * (1 - config.TEST_SIZE_NORMAL))
    val_ids = set(session_ids[n_train:])
    
    df_val = df.filter(pl.col(config.SESSION_ID_COL).is_in(val_ids))
    logger.info(f"Validation Sessions: {len(val_ids)}")
    
    # Process Windows
    grouped = df_val.group_by(config.SESSION_ID_COL, maintain_order=True).agg(
        pl.col("EventTemplate"),
        pl.col("label")
    )
    rows = grouped.to_dict(as_series=False)
    
    benchmark_windows = []
    
    for i, templates in enumerate(tqdm(rows["EventTemplate"], desc="Pre-processing Val")):
        labels = rows["label"][i]
        
        for k in range(0, len(templates), config.BLOCK_SIZE): # Use BLOCK_SIZE step? No, dataset used N=20 step=20?
            # Config says BLOCK_SIZE=64. But dataset_bgl_count used WINDOW_SIZE=20.
            # train_count used collate_fn to pad. 
            # We should use WINDOW_SIZE=20.
            W = 20
            chunk_t = templates[k:k+W]
            chunk_l = labels[k:k+W]
            
            if len(chunk_t) < 2: continue
            
            text = " \n ".join([str(t) for t in chunk_t])
            input_ids = tokenizer.encode(text)
            if len(input_ids) > config.BLOCK_SIZE: input_ids = input_ids[:config.BLOCK_SIZE]
            
            is_anom = sum(chunk_l) > 0
            
            benchmark_windows.append({
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'templates': chunk_t,
                'label': 1 if is_anom else 0
            })
            
    # RUN EVAL
    y_true = []
    y_pred_std = []
    y_pred_hyb = []
    
    logger.info(f"Evaluating {len(benchmark_windows)} windows...")
    
    # Batch manual (can't easily batch tuple mix, use size 1 or scalar loop for Hybrid check simplicity or custom collate)
    # Let's simple loop for benchmarking (slower but accurate logic)
    
    with torch.no_grad():
        for item in tqdm(benchmark_windows, desc="Inference"):
            y_true.append(item['label'])
            
            # 1. Hybrid Check (Template Frequency)
            rare_flag = False
            for t in item['templates']:
                t_str = str(t)
                if t_str not in freq_map or freq_map[t_str]['count'] < RARE_THRESHOLD:
                    rare_flag = True
                    break
            
            # 2. Model Check
            input_ids = item['input_ids'].unsqueeze(0).to(DEVICE)
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            logits, _ = model(inputs)
            probs = F.softmax(logits, dim=-1)
            _, topk_indices = torch.topk(probs, k=K_PROD, dim=-1)
            
            matches = (topk_indices == targets.unsqueeze(-1))
            hits = matches.any(dim=-1)
            misses = ~hits
            model_flag = misses.sum().item() > 0
            
            # Predictions
            y_pred_std.append(1 if model_flag else 0)
            y_pred_hyb.append(1 if (model_flag or rare_flag) else 0)
            
    # Metrics
    def print_m(name, yt, yp):
        p = precision_score(yt, yp)
        r = recall_score(yt, yp)
        f = f1_score(yt, yp)
        print(f"\n🔹 {name}:")
        print(f"   Precision: {p:.4f}")
        print(f"   Recall:    {r:.4f}")
        print(f"   F1 Score:  {f:.4f}")
        return p, r, f

    print("\n========================================")
    print("🏆 OFFICIAL BENCHMARK RESULTS")
    print("========================================")
    
    p1, r1, f1 = print_m("Standard (LogGPT K=5)", y_true, y_pred_std)
    p2, r2, f2 = print_m("Hybrid (LogGPT + Freq)", y_true, y_pred_hyb)
    
    print("\n📈 IMPROVEMENT:")
    print(f"   Recall Delta: +{(r2-r1)*100:.2f}%")
    print(f"   F1 Delta:     +{(f2-f1)*100:.2f}%")

if __name__ == "__main__":
    full_benchmark_pipeline()
