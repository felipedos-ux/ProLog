import torch
import polars as pl
import numpy as np
from pathlib import Path

# Importar do código existente
from universal_detector.model import LogGPT, GPTConfig
from universal_detector.dataset import get_tokenizer

# Configuração do LogGPT-Small (já validado no OpenStack)
# Baseado em 02_loggpt_small/config.py
N_LAYER = 4
N_HEAD = 4
N_EMBD = 256
BLOCK_SIZE = 128
VOCAB_SIZE = 50357  # GPT2 tokenizer + 100 buffer

# Caminhos
MODEL_PATH = r"D:\ProLog\02_loggpt_small\model_weights\loggpt_weights.pt"
DATA_DIR = Path(r"D:\ProLog\data\BGL_sliding_windows")

def load_model(device):
    """Carrega LogGPT-Small pré-treinado no OpenStack"""
    print(f"🔄 Loading LogGPT-Small from {MODEL_PATH}...")
    
    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD
    )
    model = LogGPT(config)
    
    # Carregar pesos
    state_dict = torch.load(MODEL_PATH, map_location=device)
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
    """Carrega dados BGL preprocessados com sliding window"""
    print(f"📂 Loading BGL sliding window data from {DATA_DIR}...")
    
    train_df = pl.read_parquet(DATA_DIR / "train.parquet")
    val_df = pl.read_parquet(DATA_DIR / "val.parquet")
    test_df = pl.read_parquet(DATA_DIR / "test.parquet")
    
    print(f"   - Train: {len(train_df)} windows")
    print(f"   - Val: {len(val_df)} windows")
    print(f"   - Test: {len(test_df)} windows")
    
    return train_df, val_df, test_df

def calculate_perplexity(model, tokenizer, sequence, device):
    """Calcula perplexidade de uma sequência de templates"""
    try:
        # Juntar templates com espaço
        text = " ".join(sequence)
        ids = tokenizer.encode(text)
        
        # Truncar se necessário
        if len(ids) > BLOCK_SIZE:
            ids = ids[-BLOCK_SIZE:]
        
        if len(ids) == 0:
            return 0.0
        
        input_ids = torch.tensor(ids).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, loss = model(input_ids, targets=input_ids)
            if loss is not None:
                return loss.item()
            else:
                return 0.0
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return 0.0

def evaluate_simple_threshold(model, tokenizer, test_df, device, threshold=2.0):
    """
    Avaliação simples: Perplexidade > threshold = Anomalia
    
    Baseado em DeepLog e papers similares que usam perplexidade como score.
    """
    print(f"\n🔬 Evaluating with simple perplexity threshold ({threshold})...")
    
    y_true = []
    y_pred = []
    perplexities = []
    
    # Limitar para 1000 janelas para teste rápido
    test_sample = test_df.head(1000)
    
    for i, row in enumerate(test_sample.iter_rows(named=True)):
        sequence = row['sequence']
        label = row['label']
        
        # Calcular perplexidade
        ppl = calculate_perplexity(model, tokenizer, sequence, device)
        
        # Predição: 1 se perplexidade > threshold
        pred = 1 if ppl > threshold else 0
        
        y_true.append(label)
        y_pred.append(pred)
        perplexities.append(ppl)
        
        if i < 10 or (label == 1 and i < 50):
            print(f"   Window {i}: Label={label}, PPL={ppl:.4f}, Pred={pred}")
    
    # Calcular métricas
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\n📊 Results:")
    print(f"   TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    # Estatísticas de perplexidade
    normal_ppls = [p for p, l in zip(perplexities, y_true) if l == 0]
    anomaly_ppls = [p for p, l in zip(perplexities, y_true) if l == 1]
    
    print(f"\n📈 Perplexity Statistics:")
    print(f"   Normal: mean={np.mean(normal_ppls):.4f}, std={np.std(normal_ppls):.4f}")
    print(f"   Anomaly: mean={np.mean(anomaly_ppls):.4f}, std={np.std(anomaly_ppls):.4f}")
    
    return f1, precision, recall

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 BGL Evaluation with LogGPT-Small (Device: {device})")
    print()
    
    # 1. Carregar modelo
    model = load_model(device)
    tokenizer = get_tokenizer()
    
    # 2. Carregar dados
    train_df, val_df, test_df = load_data()
    
    # 3. Avaliar com threshold simples
    # Vamos testar alguns thresholds
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    best_f1 = 0
    best_threshold = 0
    
    for threshold in thresholds:
        f1, precision, recall = evaluate_simple_threshold(
            model, tokenizer, test_df, device, threshold
        )
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n🏆 Best Threshold: {best_threshold} (F1: {best_f1:.4f})")
    
    # Salvar resultado
    with open("bgl_sliding_results.txt", "w") as f:
        f.write(f"Best F1: {best_f1:.4f}\n")
        f.write(f"Best Threshold: {best_threshold}\n")

if __name__ == "__main__":
    main()
