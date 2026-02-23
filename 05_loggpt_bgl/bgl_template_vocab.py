"""
Cria vocabulário de templates do BGL para detecção Top-K

Mapeia cada template único para um ID inteiro, permitindo:
1. Tokenização ao nível de template (não BPE)
2. Predição Top-K de próximos templates
3. Vocabulário reduzido (~242 templates vs 50k tokens BPE)
"""

import polars as pl
from pathlib import Path
import json

DATA_DIR = Path(r"D:\ProLog\data\BGL_sliding_windows")
OUTPUT_FILE = Path(r"D:\ProLog\bgl_template_vocab.json")

def build_template_vocab():
    """Constrói vocabulário de templates a partir dos dados BGL"""
    print("🔨 Building BGL Template Vocabulary")
    print("=" * 60)
    
    # Carregar dados
    print("\n📂 Loading BGL data...")
    train_df = pl.read_parquet(DATA_DIR / "train.parquet")
    val_df = pl.read_parquet(DATA_DIR / "val.parquet")
    test_df = pl.read_parquet(DATA_DIR / "test.parquet")
    
    print(f"   Train: {len(train_df)} windows")
    print(f"   Val:   {len(val_df)} windows")
    print(f"   Test:  {len(test_df)} windows")
    
    # Coletar todos os templates únicos
    print("\n🔍 Collecting unique templates...")
    all_templates = set()
    
    for df in [train_df, val_df, test_df]:
        for row in df.iter_rows(named=True):
            sequence = row['sequence']
            all_templates.update(sequence)
    
    # Ordenar templates para consistência
    sorted_templates = sorted(all_templates)
    
    print(f"   Found {len(sorted_templates)} unique templates")
    
    # Criar mapeamento template → ID
    template_to_id = {template: idx for idx, template in enumerate(sorted_templates)}
    id_to_template = {idx: template for template, idx in template_to_id.items()}
    
    # Adicionar tokens especiais
    vocab_size = len(sorted_templates)
    template_to_id["<PAD>"] = vocab_size
    template_to_id["<UNK>"] = vocab_size + 1
    id_to_template[vocab_size] = "<PAD>"
    id_to_template[vocab_size + 1] = "<UNK>"
    
    vocab_data = {
        "template_to_id": template_to_id,
        "id_to_template": id_to_template,
        "vocab_size": vocab_size + 2,  # +2 para <PAD> e <UNK>
        "num_real_templates": vocab_size
    }
    
    # Salvar vocabulário
    print(f"\n💾 Saving vocabulary to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    # Estatísticas
    print("\n📊 Vocabulary Statistics:")
    print(f"   Total templates: {vocab_size}")
    print(f"   With special tokens: {vocab_size + 2}")
    print(f"   K (50% for Top-K): {vocab_size // 2}")
    
    # Mostrar exemplos
    print("\n📝 Sample templates (first 10):")
    for i, template in enumerate(sorted_templates[:10]):
        print(f"   ID {i}: {template}")
    
    print("\n✅ Vocabulary created successfully!")
    return vocab_data

if __name__ == "__main__":
    vocab_data = build_template_vocab()
    print(f"\n🎯 K for Top-K detection: {vocab_data['num_real_templates'] // 2}")
