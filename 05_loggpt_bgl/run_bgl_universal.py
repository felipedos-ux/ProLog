"""
Fase 1: Universal Detector com Multi-Signal no BGL

Objetivo: Avaliar UniversalAnomalyDetector usando:
- LogGPT-Small (pré-treinado no OpenStack)
- Dados BGL preprocessados com Sliding Window
- 3 sinais: Perplexity + Rarity + Context
- Calibração adaptativa de pesos

Meta: F1 > 0.70 (vs baseline 0.66 com perplexidade simples)
"""

import torch
import polars as pl
import numpy as np
from pathlib import Path
import json

# Importar componentes do Universal Detector
from universal_detector.detector import UniversalAnomalyDetector
from universal_detector.model import LogGPT, GPTConfig
from universal_detector.dataset import get_tokenizer

# Configuração
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path(r"D:\ProLog\data\BGL_sliding_windows")
MODEL_PATH = r"D:\ProLog\02_loggpt_small\model_weights\loggpt_weights.pt"

# LogGPT-Small config (treinado no OpenStack)
N_LAYER = 4
N_HEAD = 4
N_EMBD = 256
BLOCK_SIZE = 128
VOCAB_SIZE = 50357

# Parâmetros de avaliação
TEST_SAMPLE_SIZE = 1000  # Começar com 1k janelas para testes rápidos

def load_loggpt_small(device):
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
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
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

def load_sliding_window_data():
    """Carrega dados BGL preprocessados com sliding window"""
    print(f"📂 Loading BGL sliding window data from {DATA_DIR}...")
    
    train_df = pl.read_parquet(DATA_DIR / "train.parquet")
    val_df = pl.read_parquet(DATA_DIR / "val.parquet")
    test_df = pl.read_parquet(DATA_DIR / "test.parquet")
    
    print(f"   - Train: {len(train_df)} windows")
    print(f"   - Val: {len(val_df)} windows")
    print(f"   - Test: {len(test_df)} windows")
    
    return train_df, val_df, test_df

def convert_to_detector_format(df):
    """
    Converte DataFrame Polars para formato esperado pelo UniversalDetector
    
    Input: DataFrame com colunas [window_id, sequence, label]
    Output: List of dicts com {log_sequence, label}
    """
    data = []
    for row in df.iter_rows(named=True):
        data.append({
            'log_sequence': row['sequence'],  # Lista de templates
            'label': row['label']  # 0 ou 1
        })
    return data

def calculate_metrics(y_true, y_pred):
    """Calcula Precision, Recall, F1"""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def main():
    print("=" * 80)
    print("🚀 FASE 1: Universal Detector Multi-Signal no BGL")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Test Sample Size: {TEST_SAMPLE_SIZE}")
    print()
    
    # 1. Carregar modelo
    loggpt = load_loggpt_small(DEVICE)
    tokenizer = get_tokenizer()
    
    # 2. Carregar dados
    train_df, val_df, test_df = load_sliding_window_data()
    
    # 3. Transformar para formato esperado pelo detector
    # UniversalDetector espera DataFrame Polars com colunas: session_id, EventTemplate, label
    print("\n📦 Transforming data to detector format...")
    
    def sliding_to_detector_df(df_sliding):
        """
        Converte sliding window DataFrame para formato do detector.
        
        Input: DataFrame com [window_id, sequence (list), label]
        Output: DataFrame com [session_id, EventTemplate, label] (uma linha por evento)
        """
        rows = []
        for row in df_sliding.iter_rows(named=True):
            window_id = row['window_id']
            sequence = row['sequence']
            label = row['label']
            
            # Expandir janela: cada template vira uma linha
            for template in sequence:
                rows.append({
                    'session_id': window_id,  # window_id como session_id
                    'EventTemplate': template,
                    'label': label  # Toda janela tem mesmo label
                })
        
        return pl.DataFrame(rows)
    
    train_detector_df = sliding_to_detector_df(train_df)
    val_detector_df = sliding_to_detector_df(val_df)
    test_detector_df = sliding_to_detector_df(test_df.head(TEST_SAMPLE_SIZE))
    
    print(f"   - Train: {len(train_detector_df)} events in {train_df.shape[0]} windows")
    print(f"   - Val: {len(val_detector_df)} events in {val_df.shape[0]} windows")
    print(f"   - Test: {len(test_detector_df)} events in {TEST_SAMPLE_SIZE} windows")
    
    # 4. Criar detector
    print("\n🔧 Initializing UniversalAnomalyDetector...")
    detector = UniversalAnomalyDetector(
        loggpt_model=loggpt,
        device=DEVICE
    )
    
    # 5. Fit (profiling + calibração)
    print("\n🎯 Fitting detector (profiling + calibration)...")
    print("   This will:")
    print("   - Profile dataset characteristics")
    print("   - Calculate template frequencies")
    print("   - Calibrate signal weights on validation set")
    print()
    
    detector.fit(train_detector_df, val_detector_df)
    
    print("\n✅ Detector fitted successfully!")
    print(f"   - Dataset profile: {detector.profiler.profile}")
    print(f"   - Calibrated weights: {detector.fusion.weights}")
    
    # 6. Avaliar no test set
    print("\n📊 Evaluating on test set...")
    results = detector.evaluate(test_detector_df)
    
    print("\n" + "=" * 80)
    print("📈 RESULTS (Test Set - {} samples)".format(TEST_SAMPLE_SIZE))
    print("=" * 80)
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {results['tp']}, FP: {results['fp']}")
    print(f"  TN: {results['tn']}, FN: {results['fn']}")
    
    # 7. Comparar com baseline
    baseline_f1 = 0.66
    improvement = results['f1'] - baseline_f1
    print(f"\n🎯 Comparison with Baseline:")
    print(f"  Baseline F1 (perplexity only): {baseline_f1:.4f}")
    print(f"  Current F1 (multi-signal):     {results['f1']:.4f}")
    print(f"  Improvement: {improvement:+.4f} ({improvement/baseline_f1*100:+.1f}%)")
    
    # 8. Verificar meta
    target_f1 = 0.70
    if results['f1'] >= target_f1:
        print(f"\n✅ SUCCESS! F1 ({results['f1']:.4f}) >= Target ({target_f1:.4f})")
    else:
        print(f"\n⚠️  F1 ({results['f1']:.4f}) < Target ({target_f1:.4f})")
        print("   Recommendations:")
        print("   - Analyze error cases")
        print("   - Adjust calibration parameters")
        print("   - Try different signal combinations")
    
    # 9. Salvar resultados
    output = {
        'model': 'LogGPT-Small (OpenStack)',
        'dataset': 'BGL Sliding Window',
        'test_size': TEST_SAMPLE_SIZE,
        'metrics': results,
        'calibrated_weights': detector.fusion.weights,
        'dataset_profile': detector.profiler.profile,
        'baseline_f1': baseline_f1,
        'improvement': improvement
    }
    
    output_file = "phase1_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    print("\n" + "=" * 80)
    print("🎉 Fase 1 - Tarefa 4 Concluída!")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    main()
