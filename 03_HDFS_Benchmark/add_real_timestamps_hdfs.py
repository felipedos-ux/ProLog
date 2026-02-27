"""
Script para adicionar timestamps reais do HDFS_full.log ao HDFS_data_processed.csv

Estratégia:
1. Parse HDFS_full.log para extrair timestamp + BlockId + EventId
2. Criar mapeamento: (BlockId, EventId) -> timestamp
3. Para cada linha no CSV processado, encontrar o timestamp correspondente
4. Atualizar a coluna timestamp com o valor real
"""

import polars as pl
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

def parse_hdfs_log(log_path):
    """
    Parse HDFS_full.log e extrai timestamp, BlockId e EventId
    
    Formato do log:
    081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_XXXXX src: ...
    """
    print("📂 Parsing HDFS_full.log...")
    
    # Ler structured para mapear LineId -> EventId
    print("  - Carregando HDFS_full.log_structured.csv...")
    structured_df = pl.read_csv('data/HDFS/HDFS_full.log_structured.csv')
    lineid_to_eid = dict(zip(
        structured_df['LineId'].to_list(),
        structured_df['EventId'].to_list()
    ))
    
    # Ler templates
    print("  - Carregando templates...")
    templates_df = pl.read_csv('data/HDFS/HDFS_full.log_templates.csv')
    eid_to_template = dict(zip(
        templates_df['EventId'].to_list(),
        templates_df['EventTemplate'].to_list()
    ))
    
    # Parse log linha por linha
    print("  - Extraindo timestamps e mapeando...")
    mapping = {}  # (BlockId, EventId) -> timestamp
    
    with open(log_path, 'r') as f:
        for line_num, line in enumerate(tqdm(f, desc="Parsing log"), 1):
            # Extrair timestamp
            ts_match = re.match(r'(\d{6})\s+(\d{6})\s+(\d+)', line)
            if not ts_match:
                continue
            
            # Converter timestamp
            date_str = ts_match.group(1)  # 081109 -> 2008-11-09
            time_str = ts_match.group(2)  # 203518 -> 20:35:18
            ms = int(ts_match.group(3))
            
            year = 2000 + int(date_str[0:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            hour = int(time_str[0:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            
            # Converter ms para microsegundos (ms já está em microsegundos no log)
            timestamp = datetime(year, month, day, hour, minute, second, ms)
            
            # Extrair BlockId
            blk_match = re.search(r'blk_[-\d]+', line)
            if not blk_match:
                continue
            block_id = blk_match.group(0)
            
            # Obter EventId do structured
            event_id = lineid_to_eid.get(line_num, None)
            if not event_id:
                continue
            
            # Obter template
            template = eid_to_template.get(event_id, None)
            if not template:
                continue
            
            # Criar chave única (BlockId, Template)
            key = (block_id, template)
            
            # Armazenar timestamp (se já existe, manter o primeiro)
            if key not in mapping:
                mapping[key] = timestamp
    
    print(f"  ✅ Mapeamento criado: {len(mapping)} eventos únicos")
    return mapping, eid_to_template


def update_csv_with_real_timestamps(csv_path, mapping, eid_to_template):
    """
    Atualiza HDFS_data_processed.csv com timestamps reais
    """
    print(f"\n📂 Lendo {csv_path}...")
    
    # Ler CSV
    df = pl.read_csv(csv_path)
    print(f"  - Total de linhas: {len(df)}")
    
    # Atualizar timestamps
    print("  - Atualizando timestamps...")
    updated_timestamps = []
    matched = 0
    not_matched = 0
    
    for row in tqdm(df.iter_rows(named=True), desc="Updating timestamps"):
        block_id = row['session_id']
        template = row['EventTemplate']
        
        # Buscar timestamp no mapeamento
        key = (block_id, template)
        timestamp = mapping.get(key, None)
        
        if timestamp:
            updated_timestamps.append(timestamp.isoformat())
            matched += 1
        else:
            # Manter o timestamp sintético original
            updated_timestamps.append(row['timestamp'])
            not_matched += 1
    
    # Criar DataFrame atualizado
    df_updated = df.with_columns(
        pl.Series('timestamp', updated_timestamps)
    )
    
    print(f"\n  ✅ Matched: {matched} ({matched/len(df)*100:.2f}%)")
    print(f"  ⚠️  Not matched: {not_matched} ({not_matched/len(df)*100:.2f}%)")
    
    # Fazer backup
    backup_path = csv_path.replace('.csv', '_backup.csv')
    print(f"\n💾 Criando backup: {backup_path}")
    df.write_csv(backup_path)
    
    # Salvar CSV atualizado
    print(f"💾 Salvando CSV atualizado: {csv_path}")
    df_updated.write_csv(csv_path)
    
    print("✅ Concluído!")
    
    return df_updated, matched, not_matched


def main():
    # Caminhos
    log_path = 'data/HDFS/HDFS_full.log'
    csv_path = 'data/HDFS/HDFS_data_processed.csv'
    
    # Parse log e criar mapeamento
    mapping, eid_to_template = parse_hdfs_log(log_path)
    
    # Atualizar CSV
    df_updated, matched, not_matched = update_csv_with_real_timestamps(csv_path, mapping, eid_to_template)
    
    # Estatísticas finais
    print("\n" + "="*60)
    print("📊 RESUMO")
    print("="*60)
    print(f"Total de linhas processadas: {len(df_updated)}")
    print(f"Timestamps atualizados com sucesso: {matched}")
    print(f"Timestamps não encontrados (mantidos originais): {not_matched}")
    print(f"Taxa de sucesso: {matched/len(df_updated)*100:.2f}%")
    print("="*60)
    
    if not_matched > 0:
        print("\n⚠️  AVISO: Alguns timestamps não foram encontrados.")
        print("Isso pode acontecer se:")
        print("  - O CSV processado contém templates que não estão no log original")
        print("  - Houve filtragem durante o preprocessamento")
        print("  - O mapeamento (BlockId, Template) não é único")


if __name__ == "__main__":
    main()