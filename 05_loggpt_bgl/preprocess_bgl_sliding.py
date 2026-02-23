import polars as pl
from pathlib import Path
import numpy as np

def create_sliding_windows(df, window_size=20, step_size=20):
    """
    Cria janelas deslizantes a partir do dataset BGL.
    
    Baseado na metodologia de LogADEmpirical (ICSE 2022) e DeepLog (CCS 2017).
    
    Args:
        df: DataFrame com colunas [timestamp, EventTemplate, label]
        window_size: Tamanho da janela (número de eventos)
        step_size: Passo da janela (step_size == window_size -> sem overlap)
    
    Returns:
        List of dicts com {window_id, sequence, label}
    """
    print(f"📊 Creating sliding windows (size={window_size}, step={step_size})...")
    
    # Ordenar por timestamp
    df = df.sort("timestamp")
    
    templates = df["EventTemplate"].to_list()
    labels = df["label"].to_list()
    
    print(f"   - Total logs: {len(templates)}")
    
    windows = []
    window_id = 0
    
    for i in range(0, len(templates) - window_size + 1, step_size):
        window_templates = templates[i : i + window_size]
        window_labels = labels[i : i + window_size]
        
        # Label: 1 se QUALQUER log na janela for anômalo
        window_label = 1 if any(window_labels) else 0
        
        windows.append({
            'window_id': window_id,
            'sequence': window_templates,
            'label': window_label
        })
        
        window_id += 1
    
    print(f"   - Total windows created: {len(windows)}")
    
    # Estatísticas
    anomalous = sum(1 for w in windows if w['label'] == 1)
    print(f"   - Normal windows: {len(windows) - anomalous}")
    print(f"   - Anomalous windows: {anomalous}")
    print(f"   - Anomaly rate: {anomalous / len(windows):.2%}")
    
    return windows


def main():
    # Configuração
    DATA_PATH = r"D:\ProLog\data\BGL_consistent.csv"
    OUTPUT_DIR = Path(r"D:\ProLog\data\BGL_sliding_windows")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    WINDOW_SIZE = 20  # Tamanho típico usado em papers
    STEP_SIZE = 20    # Sem overlap (fixed window)
    
    print("🚀 BGL Sliding Window Preprocessing")
    print(f"   Input: {DATA_PATH}")
    print(f"   Output: {OUTPUT_DIR}")
    print()
    
    # 1. Carregar dados
    print("📂 Loading BGL data...")
    df = pl.read_csv(DATA_PATH)
    print(f"   - Total rows: {len(df)}")
    print(f"   - Columns: {df.columns}")
    
    # Verificar colunas necessárias
    required_cols = ["timestamp", "EventTemplate", "label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    
    # Remover nulls
    df = df.drop_nulls(subset=["EventTemplate", "label"])
    print(f"   - After dropping nulls: {len(df)}")
    print()
    
    # 2. Criar janelas
    windows = create_sliding_windows(df, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    print()
    
    # 3. Converter para DataFrame
    print("📦 Converting to DataFrame...")
    windows_df = pl.DataFrame({
        'window_id': [w['window_id'] for w in windows],
        'sequence': [w['sequence'] for w in windows],
        'label': [w['label'] for w in windows]
    })
    
    # 4. Split Train/Val/Test (80/10/10)
    print("✂️ Splitting into Train/Val/Test (80/10/10)...")
    total = len(windows_df)
    train_idx = int(total * 0.8)
    val_idx = int(total * 0.9)
    
    train_df = windows_df[:train_idx]
    val_df = windows_df[train_idx:val_idx]
    test_df = windows_df[val_idx:]
    
    print(f"   - Train: {len(train_df)} windows")
    print(f"   - Val: {len(val_df)} windows")
    print(f"   - Test: {len(test_df)} windows")
    print()
    
    # 5. Salvar (Parquet suporta listas aninhadas)
    print("💾 Saving to disk (Parquet format)...")
    train_df.write_parquet(OUTPUT_DIR / "train.parquet")
    val_df.write_parquet(OUTPUT_DIR / "val.parquet")
    test_df.write_parquet(OUTPUT_DIR / "test.parquet")
    
    print(f"✅ Done! Files saved to {OUTPUT_DIR}")
    print()
    
    # 6. Estatísticas finais
    print("📊 Final Statistics:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        anomalies = split_df.filter(pl.col("label") == 1).shape[0]
        total_split = len(split_df)
        print(f"   {split_name}: {anomalies}/{total_split} anomalous ({anomalies/total_split:.2%})")


if __name__ == "__main__":
    main()
