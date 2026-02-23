import torch
import polars as pl
import numpy as np
import os
from pathlib import Path

def calculate_metrics(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\n📊 Manual Metrics:")
    print(f"   TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    return f1


    return f1

import universal_detector
print(f"DEBUG: universal_detector layout: {universal_detector}")
print(f"DEBUG: universal_detector file: {getattr(universal_detector, '__file__', 'unknown')}")

from universal_detector.detector import UniversalAnomalyDetector
from universal_detector.model import LogGPT, GPTConfig
from universal_detector.dataset import get_tokenizer
from universal_detector.feature_extractor import MultiSignalExtractor
import inspect
print(f"DEBUG: MultiSignalExtractor file: {inspect.getfile(MultiSignalExtractor)}")

# Configuration matching LogGPT-Large training
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
BLOCK_SIZE = 128
VOCAB_SIZE = 50357 # GPT2 tokenizer + 100 buffer used in training

MODEL_PATH = r"D:\ProLog\05_loggpt_large_bgl\models\loggpt_large_bgl\loggpt_bgl.pt"
DATA_PATH = r"D:\ProLog\data\BGL_consistent.csv"

def load_model(device):
    print(f"🔄 Loading LogGPT-Large from {MODEL_PATH}...")
    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD
    )
    model = LogGPT(config)
    
    # Load weights
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
    state_dict = torch.load(MODEL_PATH, map_location=device)
    # Handle state dict keys if they have "module." prefix or similar
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully.")
    return model

def load_data():
    print(f"📂 Loading BGL data from {DATA_PATH}...")
    # Load with Polars for speed
    # We expect columns: 'node_id', 'timestamp', 'EventTemplate', 'label'
    # EventTemplate might be null if parsing failed? Should filter.
    
    df = pl.read_csv(DATA_PATH)
    print(f"   - Total rows: {len(df)}")
    
    # Ensure columns exist
    required_cols = ["node_id", "timestamp", "EventTemplate", "label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
            
    # Drop nulls
    df = df.drop_nulls(subset=["EventTemplate", "label"])
    
    # Rename node_id to session_id for compatibility with UniversalDetector
    df = df.rename({"node_id": "session_id"})
    
    # Sort by timestamp - DISABLED for now to test Random Split (Concept Drift Check)
    # try:
    #     df = df.sort("timestamp")
    # except:
    #     pass 
        
    # Shuffle for better template coverage (Common in BGL benchmarks)
    df = df.sample(fraction=1.0, shuffle=True, seed=42)
    
    # Split Data
    # 60% Train, 20% Val, 20% Test
    total = len(df)
    train_idx = int(total * 0.6)
    val_idx = int(total * 0.8)
    
    df_train = df.slice(0, train_idx)
    df_val = df.slice(train_idx, val_idx - train_idx)
    df_test = df.slice(val_idx, total - val_idx)
    
    print(f"   - Train: {len(df_train)}")
    print(f"   - Val: {len(df_val)}")
    print(f"   - Test: {len(df_test)}")
    
    # DEBUG: Check Template Coverage
    train_templates = set(df_train["EventTemplate"].unique().to_list())
    test_templates = set(df_test["EventTemplate"].unique().to_list())
    
    print(f"   - Unique Templates in Train: {len(train_templates)}")
    print(f"   - Unique Templates in Test: {len(test_templates)}")
    
    missing = test_templates - train_templates
    print(f"   - Templates in Test but NOT in Train: {len(missing)}")
    
    # Check for the specific problematic template
    problem_keyword = "critical input interrupts"
    found_in_train = any(problem_keyword in t for t in train_templates)
    found_in_test = any(problem_keyword in t for t in test_templates)
    print(f"   - '{problem_keyword}' in Train: {found_in_train}")
    print(f"   - '{problem_keyword}' in Test: {found_in_test}")
    
    if found_in_train:
         # Find exact key
         key = next(t for t in train_templates if problem_keyword in t)
         print(f"   - Key in Train: '{key}' (Len: {len(key)})")
         
    return df_train, df_val, df_test

def main():
    import os
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Universal Detector Evaluation on BGL (Device: {device})")
    
    # 1. Load Data
    df_train, df_val, df_test = load_data()
    
    # 2. Load Model
    model = load_model(device)
    
    # 3. Fit Universal Detector
    detector = UniversalAnomalyDetector(model, device=device)
    detector.fit(df_train, df_val)
    
    # DEBUG: Inspect Calibration
    print("\n🐛 DEBUG: Fusion Calibration Results")
    print(f"   - Weights: {detector.fusion.weights}")
    print(f"   - Threshold: {detector.fusion.threshold:.4f}")
    print(f"   - Best Validation F1: {detector.fusion.best_f1:.4f}")
    print(f"   - Template Freq Size: {len(detector.extractor.template_freq)}")
    if len(detector.extractor.template_freq) > 0:
        print(f"   - Sample Template: {list(detector.extractor.template_freq.keys())[0]}")

    
    # 4. Evaluate on Test Set
    print("\n🔬 Evaluating on TEST set...")
    
    # Prepare Test Data
    print("⚠️ Truncating test sequences to block size 128 for safety in prototype...")
    
    test_sessions = df_test["session_id"].unique().to_list()
    if len(test_sessions) > 500:
        print(f"ℹ️ Under-sampling test set to 500 sessions (from {len(test_sessions)}) for quick evaluation.")
        test_sessions = test_sessions[:500]
        
    y_true = []
    y_pred = []
    raw_scores = []
    
    debug_limit = 5
    
    for i, sess_id in enumerate(test_sessions):
        sess_df = df_test.filter(pl.col("session_id") == sess_id)
        seq = sess_df["EventTemplate"].to_list()
        label = sess_df["label"].max()
        
        # Truncate
        if len(seq) > 100:
            seq = seq[-100:]
            
        try:
            pred = detector.predict(seq)
            score = detector.predict_proba(seq)
            
            y_true.append(label)
            y_pred.append(int(pred))
            raw_scores.append(score)
            
            # Print first few examples
            if i < debug_limit or (label == 1 and i < 20): # Print some anomalies too
                signals = detector.extractor.extract_signals(seq)
                print(f"   SESSION {sess_id} (Label: {label}) -> Pred: {int(pred)} | Score: {score:.4f}")
                print(f"      Signals: PPL={signals['perplexity']:.4f}, RAR={signals['rarity']:.4f}, CTX={signals['context']:.4f}")
                
                if signals['rarity'] > 0.5:
                    import sys
                    print(f"      ⚠️ High Rarity! Last Template: '{seq[-1]}' (Type: {type(seq[-1])})")
                    if seq[-1] in detector.extractor.template_freq:
                         print(f"         But found in freq map! Freq: {detector.extractor.template_freq[seq[-1]]}")
                    else:
                         print(f"         NOT found in freq map. FreqMap Size: {len(detector.extractor.template_freq)}")
                         if len(detector.extractor.template_freq) > 0:
                             print(f"         Sample Key: '{list(detector.extractor.template_freq.keys())[0]}' (Type: {type(list(detector.extractor.template_freq.keys())[0])})")
                    sys.stdout.flush()
                
        except Exception as e:
            print(f"❌ Error predicting session {sess_id}: {e}")
            
            
    # Metrics
    f1 = calculate_metrics(y_true, y_pred)
    
    # Save Results
    with open("bgl_universal_results.txt", "w") as f:
        f.write(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
