"""
Avaliação do modelo Top-K no BGL

Testa detecção de anomalias usando Top-K prediction ao invés de perplexidade.
Espera-se F1 > 0.90 conforme metodologia LogGPT.
"""

import torch
import polars as pl
from pathlib import Path
import json

# Importar componentes
import sys
sys.path.append(str(Path(__file__).parent / "universal_detector"))
from universal_detector.model import LogGPT, GPTConfig
from topk_detector import TopKAnomalyDetector

# Configuração
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path(r"D:\ProLog\data\BGL_sliding_windows")
VOCAB_PATH = "bgl_template_vocab.json"
MODEL_PATH = r"D:\ProLog\08_loggpt_topk\loggpt_topk_best.pt"

# Arquitetura
N_LAYER = 4
N_HEAD = 4
N_EMBD = 256
BLOCK_SIZE = 19

def eval_topk():
    print("=" * 80)
    print("🚀 Evaluating Top-K Anomaly Detection on BGL")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print()
    
    # 1. Carregar vocabulário
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    VOCAB_SIZE = vocab_data['vocab_size']
    K = vocab_data['num_real_templates'] // 2
    
    print(f"📚 Vocabulary:")
    print(f"   Templates: {vocab_data['num_real_templates']}")
    print(f"   Vocab size: {VOCAB_SIZE}")
    print(f"   K (Top-K): {K}")
    print()
    
    # 2. Carregar modelo
    print(f"🔄 Loading model from {MODEL_PATH}...")
    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD
    )
    model = LogGPT(config)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded")
    print()
    
    # 3. Criar detector
    detector = TopKAnomalyDetector(model, VOCAB_PATH, device=DEVICE)
    print()
    
    # 4. Carregar test data
    print("📂 Loading test data...")
    test_df = pl.read_parquet(DATA_DIR / "test.parquet").head(1000)  # Avaliar em 1000 janelas
    print(f"   Test windows: {len(test_df)}")
    print(f"   Normal: {len(test_df.filter(pl.col('label') == 0))}")
    print(f"   Anomalous: {len(test_df.filter(pl.col('label') == 1))}")
    print()
    
    # 5. Preparar dados para avaliação
    test_sequences = []
    for row in test_df.iter_rows(named=True):
        templates = row['sequence']
        label = row['label']
        test_sequences.append((templates, label))
    
    # 6. Avaliar
    print("🎯 Evaluating with Top-K detection...")
    results = detector.evaluate(test_sequences)
    
    # 7. Mostrar resultados
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print(f"\n🎯 Target (LogGPT paper): F1 > 0.90")
    print(f"\n✅ Achieved:")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1 Score:  {results['f1']:.4f}")
    print(f"   Accuracy:  {results['accuracy']:.4f}")
    print()
    print(f"📈 Confusion Matrix:")
    print(f"   TP: {results['tp']} | FP: {results['fp']}")
    print(f"   FN: {results['fn']} | TN: {results['tn']}")
    print()
    
    # 8. Comparar com baseline
    baseline_f1 = 0.6568  # Abordagem de perplexidade
    improvement = results['f1'] - baseline_f1
    print(f"📊 Comparison with Baseline (Perplexity):")
    print(f"   Baseline F1: {baseline_f1:.4f}")
    print(f"   Top-K F1:    {results['f1']:.4f}")
    print(f"   Improvement: {improvement:+.4f} ({improvement/baseline_f1*100:+.1f}%)")
    print()
    
    # 9. Salvar resultados
    results_path = "topk_evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'topk': results,
            'baseline': {'f1': baseline_f1},
            'improvement': float(improvement),
            'vocab_size': VOCAB_SIZE,
            'K': K
        }, f, indent=2)
    
    print(f"💾 Results saved to {results_path}")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    results = eval_topk()
    print(f"\n🎉 Evaluation complete!")
    print(f"Final F1: {results['f1']:.4f}")
