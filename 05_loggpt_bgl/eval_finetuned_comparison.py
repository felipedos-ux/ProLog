"""
Avaliação do LogGPT-Small Fine-tuned no BGL

Compara:
1. Modelo Universal (LogGPT-Small do OpenStack)
2. Modelo Fine-tuned (LogGPT-Small treinado no BGL)

Métricas: Precision, Recall, F1 usando perplexidade simples
"""

import torch
import polars as pl
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Importar componentes
import sys
sys.path.append(str(Path(__file__).parent / "universal_detector"))
from universal_detector.model import LogGPT, GPTConfig
from universal_detector.dataset import get_tokenizer

# Configuração
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path(r"D:\ProLog\data\BGL_sliding_windows")
UNIVERSAL_MODEL = r"D:\ProLog\02_loggpt_small\model_weights\loggpt_weights.pt"
FINETUNED_MODEL = r"D:\ProLog\07_loggpt_small_bgl_finetuned\loggpt_small_bgl_finetuned.pt"

# Arquitetura
N_LAYER = 4
N_HEAD = 4
N_EMBD = 256
BLOCK_SIZE = 128
VOCAB_SIZE = 50357

# Avaliação
TEST_SAMPLE_SIZE = 1000  # Mesmas 1000 janelas da Fase 1

def load_model(model_path, device):
    """Carrega LogGPT-Small"""
    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD
    )
    model = LogGPT(config)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def calculate_perplexity(model, sequence, tokenizer, device):
    """Calcula perplexidade de uma sequência"""
    text = " ".join(sequence)
    ids = tokenizer.encode(text)
    
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

def evaluate_model(model, test_df, tokenizer, device, threshold):
    """Avalia modelo usando perplexidade simples"""
    perplexities = []
    labels = []
    
    print(f"   Calculating perplexities for {len(test_df)} windows...")
    for row in tqdm(test_df.iter_rows(named=True), total=len(test_df)):
        seq = row['sequence']
        label = row['label']
        
        ppl = calculate_perplexity(model, seq, tokenizer, device)
        perplexities.append(ppl)
        labels.append(label)
    
    # Aplicar threshold
    predictions = [1 if ppl > threshold else 0 for ppl in perplexities]
    
    # Calcular métricas
    tp = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 1)
    tn = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 0)
    fp = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Estatísticas de perplexidade
    normal_ppls = [p for p, l in zip(perplexities, labels) if l == 0]
    anomaly_ppls = [p for p, l in zip(perplexities, labels) if l == 1]
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'normal_ppl_mean': np.mean(normal_ppls) if normal_ppls else 0.0,
        'normal_ppl_std': np.std(normal_ppls) if normal_ppls else 0.0,
        'anomaly_ppl_mean': np.mean(anomaly_ppls) if anomaly_ppls else 0.0,
        'anomaly_ppl_std': np.std(anomaly_ppls) if anomaly_ppls else 0.0,
    }

def find_best_threshold(model, val_df, tokenizer, device):
    """Encontra melhor threshold no validation set"""
    print("   Finding best threshold on validation set...")
    
    perplexities = []
    labels = []
    
    # Usar 1000 janelas do val set
    val_sample = val_df.head(1000)
    
    for row in tqdm(val_sample.iter_rows(named=True), total=len(val_sample)):
        seq = row['sequence']
        label = row['label']
        
        ppl = calculate_perplexity(model, seq, tokenizer, device)
        perplexities.append(ppl)
        labels.append(label)
    
    # Testar thresholds (range ajustado para perplexidades observadas)
    best_f1 = 0
    best_threshold = 0
    
    # Range mais amplo para capturar diferentes escalas de perplexidade
    for threshold in np.arange(0.0, 10.0, 0.05):
        predictions = [1 if ppl > threshold else 0 for ppl in perplexities]
        
        tp = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 1)
        tn = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 0)
        fp = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 1)
        fn = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"   Best threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    return best_threshold

def main():
    print("=" * 80)
    print("🚀 Avaliação: Universal vs Fine-tuned LogGPT-Small")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print()
    
    # 1. Carregar dados
    print("📂 Loading BGL data...")
    val_df = pl.read_parquet(DATA_DIR / "val.parquet")
    test_df = pl.read_parquet(DATA_DIR / "test.parquet").head(TEST_SAMPLE_SIZE)
    print(f"   Val: {len(val_df)} windows")
    print(f"   Test: {len(test_df)} windows")
    
    tokenizer = get_tokenizer()
    
    # 2. Carregar modelos
    print("\n🔄 Loading models...")
    print("   - Universal (OpenStack)")
    universal_model = load_model(UNIVERSAL_MODEL, DEVICE)
    print("   - Fine-tuned (BGL)")
    finetuned_model = load_model(FINETUNED_MODEL, DEVICE)
    
    # 3. Encontrar thresholds
    print("\n🎯 Finding optimal thresholds...")
    print("\nUniversal Model:")
    universal_threshold = find_best_threshold(universal_model, val_df, tokenizer, DEVICE)
    
    print("\nFine-tuned Model:")
    finetuned_threshold = find_best_threshold(finetuned_model, val_df, tokenizer, DEVICE)
    
    # 4. Avaliar no test set
    print("\n📊 Evaluating on test set...")
    
    print("\nUniversal Model:")
    universal_results = evaluate_model(universal_model, test_df, tokenizer, DEVICE, universal_threshold)
    
    print("\nFine-tuned Model:")
    finetuned_results = evaluate_model(finetuned_model, test_df, tokenizer, DEVICE, finetuned_threshold)
    
    # 5. Comparar resultados
    print("\n" + "=" * 80)
    print("📈 RESULTS COMPARISON")
    print("=" * 80)
    
    print("\n🔵 Universal Model (OpenStack):")
    print(f"   Threshold: {universal_threshold:.2f}")
    print(f"   Precision: {universal_results['precision']:.4f}")
    print(f"   Recall:    {universal_results['recall']:.4f}")
    print(f"   F1 Score:  {universal_results['f1']:.4f}")
    print(f"   Normal PPL:  {universal_results['normal_ppl_mean']:.2f} ± {universal_results['normal_ppl_std']:.2f}")
    print(f"   Anomaly PPL: {universal_results['anomaly_ppl_mean']:.2f} ± {universal_results['anomaly_ppl_std']:.2f}")
    
    print("\n🟢 Fine-tuned Model (BGL):")
    print(f"   Threshold: {finetuned_threshold:.2f}")
    print(f"   Precision: {finetuned_results['precision']:.4f}")
    print(f"   Recall:    {finetuned_results['recall']:.4f}")
    print(f"   F1 Score:  {finetuned_results['f1']:.4f}")
    print(f"   Normal PPL:  {finetuned_results['normal_ppl_mean']:.2f} ± {finetuned_results['normal_ppl_std']:.2f}")
    print(f"   Anomaly PPL: {finetuned_results['anomaly_ppl_mean']:.2f} ± {finetuned_results['anomaly_ppl_std']:.2f}")
    
    print("\n📊 Improvement:")
    f1_improvement = finetuned_results['f1'] - universal_results['f1']
    print(f"   F1: {f1_improvement:+.4f} ({f1_improvement/universal_results['f1']*100:+.1f}%)")
    
    precision_improvement = finetuned_results['precision'] - universal_results['precision']
    print(f"   Precision: {precision_improvement:+.4f}")
    
    recall_improvement = finetuned_results['recall'] - universal_results['recall']
    print(f"   Recall: {recall_improvement:+.4f}")
    
    print("\n" + "=" * 80)
    
    # 6. Salvar resultados
    import json
    results = {
        'universal': {
            'threshold': float(universal_threshold),
            **{k: float(v) if isinstance(v, (int, float, np.number)) else v 
               for k, v in universal_results.items()}
        },
        'finetuned': {
            'threshold': float(finetuned_threshold),
            **{k: float(v) if isinstance(v, (int, float, np.number)) else v 
               for k, v in finetuned_results.items()}
        },
        'improvement': {
            'f1': float(f1_improvement),
            'precision': float(precision_improvement),
            'recall': float(recall_improvement)
        }
    }
    
    with open('phase2_finetuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("💾 Results saved to phase2_finetuning_results.json")
    print()

if __name__ == "__main__":
    main()
