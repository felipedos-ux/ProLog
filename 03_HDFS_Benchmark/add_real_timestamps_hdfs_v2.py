"""
Script V2 para adicionar timestamps reais do HDFS_full.log ao HDFS_data_processed.csv

Estratégia melhorada:
1. Parse HDFS_full.log para extrair timestamp + BlockId + EventId
2. Criar mapeamento por ORDEM de ocorrência: BlockId -> [timestamp1, timestamp2, ...]
3. Para cada linha no CSV, atribuir o timestamp correspondente à mesma posição
"""

import polars as pl
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

def parse_hdfs_log_by_blockid(log_path):
    """
    Parse HDFS_full.log e extrai timestamps organizados por BlockId
    
    Retorna: {BlockId: [timestamp1, timestamp2, ...]}
    """
    print("📂 Parsing HDFS_full.log...")
    
    # Ler structured para mapear LineId -> EventId
    print("  - Carregando HDFS_full.log_structured.csv...")
    structured_df = pl.read_csv('../data/HDFS/HDFS_full.log_structured.csv')
    lineid_to_eid = dict(zip(
        structured_df['LineId'].to_list(),
        structured_df['EventId'].to_list()
    ))
    
    # Ler templates
    print("  - Carregando templates...")
    templates_df = pl.read_csv('../data/HDFS/HDFS_full.log_templates.csv')
    eid_to_template = dict(zip(
        templates_df['EventId'].to_list(),
        templates_df['EventTemplate'].to_list()
    ))
    
    # Parse log linha por linha
    print("  - Extraindo timestamps por BlockId...")
    blockid_to_timestamps = {}  # {BlockId: [ts1, ts2, ...]}
    
    with open(log_path, 'r') as f:
        for line_num, line in enumerate(tqdm(f, desc="Parsing log"), 1):
            # Extrair timestamp
            ts_match = re.match(r'(\d{6})\s+(\d{6})\s+(\d+)', line)
            if not ts_match:
                continue
            
            # Converter timestamp
            date_str = ts_match.group(1)
            time_str = ts_match.group(2)
            ms = int(ts_match.group(3))
            
            year = 2000 + int(date_str[0:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            hour = int(time_str[0:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            
            timestamp = datetime(year, month, day, hour, minute, second, ms)
            
            # Extrair BlockId
            blk_match = re.search(r'blk_[-\d]+', line)
            if not blk_match:
                continue
            block_id = blk_match.group(0)
            
            # Adicionar timestamp à lista do BlockId
            if block_id not in blockid_to_timestamps:
                blockid_to_timestamps[block_id] = []
            blockid_to_timestamps[block_id].append(timestamp)
    
    print(f"  ✅ Mapeamento criado: {len(blockid_to_timestamps)} BlockIds")
    total_events = sum(len(ts) for ts in blockid_to_timestamps.values())
    print(f"  ✅ Total de eventos: {total_events}")
    return blockid_to_timestamps


def update_csv_with_ordered_timestamps(csv_path, blockid_to_timestamps):
    """
    Atualiza HDFS_data_processed.csv com timestamps reais por ordem
    """
    print(f"\n📂 Lendo {csv_path}...")
    
    # Ler CSV
    df = pl.read_csv(csv_path)
    print(f"  - Total de linhas: {len(df)}")
    
    # Atualizar timestamps
    print("  - Atualizando timestamps por BlockId e ordem...")
    updated_timestamps = []
    matched = 0
    not_matched = 0
    
    # Processar cada sessão (BlockId) separadamente
    block_ids = df['session_id'].unique().to_list()
    
    for block_id in tqdm(block_ids, desc="Processing BlockIds"):
        # Filtrar linhas deste BlockId
        block_df = df.filter(pl.col('session_id') == block_id)
        
        # Obter timestamps do log para este BlockId
        log_timestamps = blockid_to_timestamps.get(block_id, None)
        
        if not log_timestamps:
            # Manter timestamps sintéticos originais
            for _ in range(len(block_df)):
                updated_timestamps.append(block_df[0, 'timestamp'])
            not_matched += len(block_df)
            continue
        
        # Atribuir timestamps por ordem
        for i in range(len(block_df)):
            if i < len(log_timestamps):
                updated_timestamps.append(log_timestamps[i].isoformat())
                matched += 1
            else:
                # Se tiver mais eventos no CSV que no log, manter o último timestamp
                updated_timestamps.append(log_timestamps[-1].isoformat())
                not_matched += 1
    
    # Criar DataFrame atualizado
    df_updated = df.with_columns(
        pl.Series('timestamp', updated_timestamps)
    )
    
    print(f"\n  ✅ Matched: {matched} ({matched/len(df)*100:.2f}%)")
    print(f"  ⚠️  Not matched: {not_matched} ({not_matched/len(df)*100:.2f}%)")
    
    # Fazer backup
    backup_path = csv_path.replace('.csv', '_backup_v2.csv')
    print(f"\n💾 Criando backup: {backup_path}")
    df.write_csv(backup_path)
    
    # Salvar CSV atualizado
    print(f"💾 Salvando CSV atualizado: {csv_path}")
    df_updated.write_csv(csv_path)
    
    print("✅ Concluído!")
    
    return df_updated, matched, not_matched


def main():
    # Caminhos (ajustados para rodar de 03_HDFS_Benchmark/)
    log_path = '../data/HDFS/HDFS_full.log'
    csv_path = '../data/HDFS/HDFS_data_processed.csv'
    
    # Parse log e criar mapeamento por BlockId
    blockid_to_timestamps = parse_hdfs_log_by_blockid(log_path)
    
    # Atualizar CSV
    df_updated, matched, not_matched = update_csv_with_ordered_timestamps(csv_path, blockid_to_timestamps)
    
    # Estatísticas finais
    print("\n" + "="*60)
    print("📊 RESUMO")
    print("="*60)
    print(f"Total de linhas processadas: {len(df_updated)}")
    print(f"Timestamps atualizados com sucesso: {matched}")
    print(f"Timestamps não encontrados (mantidos originais): {not_matched}")
    print(f"Taxa de sucesso: {matched/len(df_updated)*100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()