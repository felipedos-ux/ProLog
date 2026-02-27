"""
Script para verificar se as linhas do HDFS_data_processed.csv
estão na mesma ordem dos eventos no HDFS_full.log
"""

import polars as pl

# Ler CSV processado
print("=== Carregando CSV processado ===")
df_csv = pl.read_csv('data/HDFS/HDFS_data_processed.csv', n_rows=50)
print(f"Total de linhas no CSV: {len(df_csv)}\n")

print("=== HDFS_data_processed.csv (primeiras 50 linhas) ===")
for i, row in enumerate(df_csv.iter_rows(named=True)):
    session_id = row['session_id']
    template = row['EventTemplate']
    print(f"{i+1:3d}. {session_id[:30]:<30} | {template[:50]}")

print("\n" + "="*80)

# Ler structured para obter EventIds
print("\n=== Carregando HDFS_full.log_structured.csv ===")
structured_df = pl.read_csv('data/HDFS/HDFS_full.log_structured.csv', n_rows=50)

# Carregar templates
templates_df = pl.read_csv('data/HDFS/HDFS_full.log_templates.csv')
eid_to_template = dict(zip(
    templates_df['EventId'].to_list(),
    templates_df['EventTemplate'].to_list()
))

print("\n=== HDFS_full.log_structured.csv (primeiras 50 linhas) ===")
for i, row in enumerate(structured_df.iter_rows(named=True)):
    event_id = row['EventId']
    template = eid_to_template.get(event_id, 'Unknown')
    print(f"{i+1:3d}. EventId={event_id:<6} | {template[:50]}")

print("\n" + "="*80)

# Comparação detalhada das primeiras 20 linhas
print("\n=== COMPARAÇÃO DIRETA (primeiras 20 linhas) ===")
print(f"{'CSV #':<6} | {'Log #':<6} | {'CSV Template':<60} | {'Log Template':<60} | {'Match?'}")
print("-" * 160)

for i in range(min(20, len(df_csv), len(structured_df))):
    csv_template = df_csv[i, 'EventTemplate']
    log_event_id = structured_df[i, 'EventId']
    log_template = eid_to_template.get(log_event_id, 'Unknown')
    
    match = "✅" if csv_template == log_template else "❌"
    print(f"{i+1:<6} | {i+1:<6} | {csv_template[:60]:<60} | {log_template[:60]:<60} | {match}")