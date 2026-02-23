import polars as pl
import pandas as pd
import numpy as np

# Carregar dados
df = pl.read_csv("D:/ProLog/data/BGL_processed.csv", infer_schema_length=10000)

# Analisar distribuição temporal de sessões
print("=" * 80)
print("DIAGNÓSTICO DE TIMESTAMPS - BGL DATASET")
print("=" * 80)

# 1. Verificar formato de timestamps
print("\n1. AMOSTRA DE TIMESTAMPS:")
print(df.select(["node_id", "timestamp"]).head(10))

# 2. Analisar duração de sessões
session_durations = []
for node_id in df["node_id"].unique().to_list()[:1000]:  # Primeiros 1000 nodes
    session_df = df.filter(pl.col("node_id") == node_id)
    timestamps = session_df["timestamp"].to_list()
    
    if len(timestamps) > 1:
        try:
            ts_parsed = [pd.to_datetime(ts, unit='s') for ts in timestamps]
            duration_hours = (max(ts_parsed) - min(ts_parsed)).total_seconds() / 3600
            session_durations.append(duration_hours)
        except:
            pass

print("\n2. DISTRIBUIÇÃO DE DURAÇÃO DE SESSÕES (horas):")
print(f"   Min:    {np.min(session_durations):.2f}h")
print(f"   25%:    {np.percentile(session_durations, 25):.2f}h")
print(f"   Median: {np.percentile(session_durations, 50):.2f}h")
print(f"   75%:    {np.percentile(session_durations, 75):.2f}h")
print(f"   95%:    {np.percentile(session_durations, 95):.2f}h")
print(f"   99%:    {np.percentile(session_durations, 99):.2f}h")
print(f"   Max:    {np.max(session_durations):.2f}h ({np.max(session_durations)/24:.1f} dias)")

# 3. Contar sessões com duração > 1 dia
long_sessions = sum(1 for d in session_durations if d > 24)
print(f"\n3. SESSÕES COM DURAÇÃO > 24H: {long_sessions}/{len(session_durations)} ({long_sessions/len(session_durations)*100:.1f}%)")

# 4. Analisar sessões anômalas
anom_durations = []
anom_ids = df.filter(pl.col("label") == 1)["node_id"].unique().to_list()[:500]
for node_id in anom_ids:
    session_df = df.filter(pl.col("node_id") == node_id)
    timestamps = session_df["timestamp"].to_list()
    
    if len(timestamps) > 1:
        try:
            ts_parsed = [pd.to_datetime(ts, unit='s') for ts in timestamps]
            duration_hours = (max(ts_parsed) - min(ts_parsed)).total_seconds() / 3600
            anom_durations.append(duration_hours)
        except:
            pass

print("\n4. DISTRIBUIÇÃO DE DURAÇÃO DE SESSÕES ANÔMALAS (horas):")
print(f"   Median: {np.percentile(anom_durations, 50):.2f}h")
print(f"   95%:    {np.percentile(anom_durations, 95):.2f}h")
print(f"   Max:    {np.max(anom_durations):.2f}h ({np.max(anom_durations)/24:.1f} dias)")

print("\n" + "=" * 80)
print("CONCLUSÃO:")
print("Se muitas sessões têm duração > 24h, o lead time calculado está incorreto.")
print("Solução: Usar janela temporal (sliding window) ao invés de node_id completo.")
print("=" * 80)
