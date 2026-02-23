"""
Script de Geração de Dataset "Windowed" para Re-treino.

Objetivo:
Criar um CSV de treino onde cada "sessão" é, na verdade, uma Janela de 2h.
Isso alinha o treino (model.fit) com a inferência (slide window), permitindo
que o modelo aprenda a classificar janelas curtas corretamente.

Estratégia:
1. Carregar logs brutos.
2. Agrupar por NodeID.
3. Fazer Sliding Window (2h) em TODOS os dados.
4. Filtrar/Balancear:
   - Manter 100% das janelas com erro (Label 1).
   - Manter 10% das janelas normais (Label 0) para não enviesar (Downsampling).
5. Salvar como `BGL_windowed_train.csv`.
"""

import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta
import os
from pathlib import Path

from config import DATA_FILE, SESSION_ID_COL, LABEL_COL, TIMESTAMP_COL

# CONFIG
WINDOW_HOURS = 2
OUTPUT_FILE = Path("../data/BGL_windowed_train.csv")
DOWNSAMPLE_NORMAL_RATIO = 0.1 # Manter apenas 10% das janelas normais (são muitas!)

def generate_windowed_dataset():
    data_path_str = str(DATA_FILE.resolve())
    print(f"📦 Loading BGL data from {data_path_str}...")
    df = pl.read_csv(data_path_str, infer_schema_length=10000)
    
    # Garantir ordenação
    df = df.sort([SESSION_ID_COL, TIMESTAMP_COL])
    
    # Iterar por grupo
    session_groups = df.group_by(SESSION_ID_COL, maintain_order=True)
    
    new_rows = []
    
    print("⚡ Processing Sliding Windows...")
    
    # Stats
    total_wins = 0
    anom_wins = 0
    norm_wins = 0
    
    for tid, session_df in tqdm(session_groups, total=len(df[SESSION_ID_COL].unique())):
        raw_ts = session_df[TIMESTAMP_COL].to_list()
        templates = session_df["EventTemplate"].to_list()
        labels = session_df[LABEL_COL].to_list()
        content = session_df["Content"].to_list() if "Content" in session_df.columns else [""] * len(templates)
        
        if len(raw_ts) < 2: continue
        
        try:
            ts_datetime = [pd.to_datetime(ts, unit='s') for ts in raw_ts]
        except: continue
        
        # Sliding Logic
        window_delta = timedelta(hours=WINDOW_HOURS)
        window_start = ts_datetime[0]
        current_indices = []
        
        for i, ts in enumerate(ts_datetime):
            if ts <= window_start + window_delta:
                current_indices.append(i)
            else:
                if len(current_indices) >= 2:
                    process_window(new_rows, tid, current_indices, templates, raw_ts, labels, content)
                    total_wins += 1
                
                window_start = ts
                current_indices = [i]
                
        # Last
        if len(current_indices) >= 2:
            process_window(new_rows, tid, current_indices, templates, raw_ts, labels, content)
            total_wins += 1

    print(f"\n📊 Generated {len(new_rows)} windows (after filtering).")
    
    # Converter para DF e Salvar
    print("💾 Saving to CSV...")
    
    # A estrutura do novo CSV deve ser compatível com dataset.py?
    # O dataset.py espera logs individuais.
    # Truque: Vamos salvar LOGS, mas o "SessionId" será "WindowId".
    # Assim o dataset.py vai agrupar por WindowId e tratar como sessão.
    
    # Flattening data
    flat_data = []
    for win in new_rows:
        # Downsampling logic here?
        # É melhor fazer antes para economizar RAM, mas ok.
        is_anom = win['label'] == 1
        
        if is_anom:
            continue # SKIP ANOMALIES (Unsupervised Training)
            
        if np.random.rand() > DOWNSAMPLE_NORMAL_RATIO:
            continue # Skip normal window (Downsampling)
        
        win_id = f"{win['node_id']}_{win['start_ts']}"
        
        for i in range(len(win['templates'])):
            flat_data.append({
                SESSION_ID_COL: win_id, # Window ID vira Session ID
                'EventTemplate': win['templates'][i],
                TIMESTAMP_COL: win['timestamps'][i],
                LABEL_COL: win['log_labels'][i], # Label do log individual
                'window_label': win['label'] # Label da janela inteira (meta)
            })
            
    final_df = pl.DataFrame(flat_data)
    final_df.write_csv(str(OUTPUT_FILE))
    
    print(f"✅ Saved {len(final_df)} logs to {OUTPUT_FILE}")
    print(f"New 'SessionId' is actually WindowId.")

def process_window(rows, node_id, indices, templates, timestamps, labels, content):
    w_labels = [labels[j] for j in indices]
    w_anom = 1 if sum(w_labels) > 0 else 0
    
    rows.append({
        'node_id': node_id,
        'start_ts': timestamps[indices[0]],
        'templates': [templates[j] for j in indices],
        'timestamps': [timestamps[j] for j in indices],
        'log_labels': w_labels,
        'label': int(is_anom)
    })

if __name__ == "__main__":
    generate_windowed_dataset()
