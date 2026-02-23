# -*- coding: utf-8 -*-
"""
Análise de Viabilidade: Dataset SIAT
=====================================
Avalia se o dataset SIAT (logs reais de produção) é adequado para
detecção de anomalias com LogGPT, comparando com HDFS e OpenStack.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

print("=" * 70)
print("  ANÁLISE DE VIABILIDADE: Dataset SIAT")
print("=" * 70)

# ====================================================================
# 1. CARREGAMENTO E ESTRUTURA BÁSICA
# ====================================================================
print("\n[1] CARREGANDO DADOS...")

# Tentar diferentes encodings
encodings = ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8']
df = None

for enc in encodings:
    try:
        df = pd.read_csv('D:/ProLog/data/siat.csv', encoding=enc, header=None, low_memory=False)
        print(f"  ✓ Arquivo carregado com encoding: {enc}")
        break
    except Exception as e:
        continue

if df is None:
    print("  ✗ ERRO: Não foi possível carregar o arquivo")
    exit(1)

# Nomear colunas baseado no padrão observado
df.columns = ['timestamp', 'status_code', 'method', 'endpoint', 'service', 'server', 'ip', 'city', 'country', 'user_agent']

print(f"\n  Total de registros: {len(df):,}")
print(f"  Total de colunas: {len(df.columns)}")
print(f"  Tamanho em memória: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ====================================================================
# 2. ANÁLISE DE CAMPOS
# ====================================================================
print("\n[2] ANÁLISE DE CAMPOS...")

print("\n  Campos disponíveis:")
for i, col in enumerate(df.columns, 1):
    non_null = df[col].notna().sum()
    unique = df[col].nunique()
    print(f"    {i}. {col:15s} — {non_null:,} não-nulos ({non_null/len(df)*100:.1f}%), {unique:,} únicos")

# ====================================================================
# 3. ANÁLISE TEMPORAL
# ====================================================================
print("\n[3] ANÁLISE TEMPORAL...")

try:
    # Remove header if present
    if df['timestamp'].iloc[0] == 'timestamp':
        df = df.iloc[1:].reset_index(drop=True)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    min_time = df['timestamp'].min()
    max_time = df['timestamp'].max()
    duration = max_time - min_time
    
    print(f"  Período: {min_time} a {max_time}")
    print(f"  Duração: {duration.days} dias ({duration.days/30:.1f} meses)")
    
    # Logs por dia
    df['date'] = df['timestamp'].dt.date
    logs_per_day = df.groupby('date').size()
    print(f"  Logs/dia: média={logs_per_day.mean():,.0f}, min={logs_per_day.min():,}, max={logs_per_day.max():,}")
    
except Exception as e:
    print(f"  ⚠️ Erro ao processar timestamps: {e}")

# ====================================================================
# 4. ANÁLISE DE SESSÕES
# ====================================================================
print("\n[4] ANÁLISE DE SESSÕES...")

# Estratégia 1: IP como identificador de sessão
sessions_by_ip = df.groupby('ip').size()
print(f"\n  Sessões por IP:")
print(f"    Total de IPs únicos: {len(sessions_by_ip):,}")
print(f"    Logs/IP: média={sessions_by_ip.mean():.1f}, mediana={sessions_by_ip.median():.0f}")
print(f"    Min={sessions_by_ip.min()}, Max={sessions_by_ip.max():,}")

# Estratégia 2: IP + Janela temporal (5 min)
print(f"\n  Sessões por IP + Janela Temporal (5 min):")
try:
    df = df.sort_values(['ip', 'timestamp']).reset_index(drop=True)
    df['time_diff'] = df.groupby('ip')['timestamp'].diff().dt.total_seconds()
    df['new_session'] = (df['time_diff'].isna()) | (df['time_diff'] > 300)  # 5 min
    df['session_id'] = df.groupby('ip')['new_session'].cumsum()
    df['global_session_id'] = df['ip'].astype(str) + '_' + df['session_id'].astype(str)
    
    sessions = df.groupby('global_session_id').size()
    print(f"    Total de sessões: {len(sessions):,}")
    print(f"    Logs/sessão: média={sessions.mean():.1f}, mediana={sessions.median():.0f}")
    print(f"    Min={sessions.min()}, Max={sessions.max():,}")
except Exception as e:
    print(f"    ⚠️ Erro ao calcular sessões temporais: {e}")
    print(f"    Usando apenas IP como identificador de sessão")
    sessions = sessions_by_ip


# Distribuição de tamanho de sessões
print(f"\n  Distribuição de tamanho de sessões:")
bins = [0, 1, 2, 5, 10, 20, 50, 100, 500, 1000, float('inf')]
labels = ['1', '2', '3-5', '6-10', '11-20', '21-50', '51-100', '101-500', '501-1000', '>1000']
session_size_dist = pd.cut(sessions, bins=bins, labels=labels).value_counts().sort_index()
for label, count in session_size_dist.items():
    pct = count / len(sessions) * 100
    print(f"    {label:>10s} logs: {count:>6,} sessões ({pct:>5.1f}%)")

# ====================================================================
# 5. ANÁLISE DE ENDPOINTS (Templates)
# ====================================================================
print("\n[5] ANÁLISE DE ENDPOINTS...")

endpoint_counts = df['endpoint'].value_counts()
print(f"  Total de endpoints únicos: {len(endpoint_counts):,}")
print(f"\n  Top 20 endpoints mais frequentes:")
for i, (endpoint, count) in enumerate(endpoint_counts.head(20).items(), 1):
    pct = count / len(df) * 100
    print(f"    {i:2d}. {count:>8,} ({pct:>5.1f}%) — {endpoint[:60]}")

# ====================================================================
# 6. ANÁLISE DE STATUS CODES (Potenciais Anomalias)
# ====================================================================
print("\n[6] ANÁLISE DE STATUS CODES...")

status_counts = df['status_code'].value_counts()
print(f"  Total de status codes únicos: {len(status_counts)}")
print(f"\n  Distribuição de status codes:")
for status, count in status_counts.items():
    pct = count / len(df) * 100
    status_type = "✓ Normal" if str(status).startswith('2') else ("⚠️ Redirect" if str(status).startswith('3') else "❌ Erro")
    print(f"    {status}: {count:>8,} ({pct:>5.1f}%) — {status_type}")

# Calcular taxa de erro
error_codes = df['status_code'].astype(str).str.startswith(('4', '5'))
error_rate = error_codes.sum() / len(df) * 100
print(f"\n  Taxa de erro (4xx + 5xx): {error_rate:.2f}%")

# ====================================================================
# 7. COMPARAÇÃO COM HDFS E OPENSTACK
# ====================================================================
print("\n[7] COMPARAÇÃO COM DATASETS VALIDADOS...")

comparison = {
    'HDFS': {
        'logs': 11_175_629,
        'sessions': 72_661,
        'templates': 29,
        'anomaly_rate': 23.17,
        'avg_session_size': 153.7,
        'type': 'Structured logs (Drain parsing)',
        'label_source': 'Ground truth labels'
    },
    'OpenStack': {
        'logs': 207_000,
        'sessions': 507,
        'templates': 48,
        'anomaly_rate': 41.4,
        'avg_session_size': 408.3,
        'type': 'Structured logs (Drain parsing)',
        'label_source': 'Ground truth labels'
    },
    'SIAT': {
        'logs': len(df),
        'sessions': len(sessions),
        'templates': len(endpoint_counts),
        'anomaly_rate': error_rate,
        'avg_session_size': sessions.mean(),
        'type': 'HTTP access logs',
        'label_source': 'Status codes (4xx/5xx)'
    }
}

print("\n  Comparação:")
print(f"  {'Métrica':<25} {'HDFS':>15} {'OpenStack':>15} {'SIAT':>15}")
print("  " + "-" * 72)
for metric in ['logs', 'sessions', 'templates', 'anomaly_rate', 'avg_session_size']:
    hdfs_val = comparison['HDFS'][metric]
    os_val = comparison['OpenStack'][metric]
    siat_val = comparison['SIAT'][metric]
    
    if metric == 'anomaly_rate':
        print(f"  {metric:<25} {hdfs_val:>14.1f}% {os_val:>14.1f}% {siat_val:>14.2f}%")
    elif metric == 'avg_session_size':
        print(f"  {metric:<25} {hdfs_val:>15.1f} {os_val:>15.1f} {siat_val:>15.1f}")
    else:
        print(f"  {metric:<25} {hdfs_val:>15,} {os_val:>15,} {siat_val:>15,}")

# ====================================================================
# 8. REQUISITOS PARA LOGGPT
# ====================================================================
print("\n[8] AVALIAÇÃO DE REQUISITOS PARA LOGGPT...")

requirements = {
    'Sequências temporais': {
        'required': True,
        'met': df['timestamp'].notna().all(),
        'details': 'Timestamps presentes em todos os registros'
    },
    'Agrupamento em sessões': {
        'required': True,
        'met': True,
        'details': f'{len(sessions):,} sessões identificadas via IP + janela temporal'
    },
    'Templates/Padrões': {
        'required': True,
        'met': len(endpoint_counts) > 10,
        'details': f'{len(endpoint_counts):,} endpoints únicos (equivalente a templates)'
    },
    'Labels de anomalia': {
        'required': True,
        'met': error_rate > 0,
        'details': f'Status codes 4xx/5xx como proxy ({error_rate:.2f}% de erro)'
    },
    'Volume suficiente': {
        'required': True,
        'met': len(sessions) > 1000,
        'details': f'{len(sessions):,} sessões (>1000 mínimo recomendado)'
    },
    'Sessões normais para treino': {
        'required': True,
        'met': (100 - error_rate) > 50,
        'details': f'{100-error_rate:.1f}% de sessões normais disponíveis'
    }
}

print("\n  Requisito                          Status    Detalhes")
print("  " + "-" * 70)
for req, info in requirements.items():
    status = "✓ OK" if info['met'] else "✗ FALTA"
    print(f"  {req:<35} {status:<10} {info['details']}")

all_met = all(r['met'] for r in requirements.values())

# ====================================================================
# 9. DESAFIOS E LIMITAÇÕES
# ====================================================================
print("\n[9] DESAFIOS E LIMITAÇÕES IDENTIFICADOS...")

challenges = []

if error_rate < 5:
    challenges.append(f"Taxa de erro muito baixa ({error_rate:.2f}%) — pode dificultar calibração")
elif error_rate > 30:
    challenges.append(f"Taxa de erro alta ({error_rate:.2f}%) — pode indicar dados ruidosos")

if sessions.median() < 3:
    challenges.append(f"Sessões muito curtas (mediana={sessions.median():.0f}) — limitação conhecida do LogGPT")

if len(endpoint_counts) > 1000:
    challenges.append(f"Muitos endpoints únicos ({len(endpoint_counts):,}) — pode precisar de agrupamento")

if not challenges:
    print("  ✓ Nenhum desafio crítico identificado")
else:
    for i, challenge in enumerate(challenges, 1):
        print(f"  {i}. ⚠️ {challenge}")

# ====================================================================
# 10. RECOMENDAÇÃO FINAL
# ====================================================================
print("\n[10] RECOMENDAÇÃO FINAL...")
print("  " + "=" * 70)

if all_met and len(challenges) < 3:
    verdict = "✅ VIÁVEL"
    recommendation = """
  O dataset SIAT ATENDE aos requisitos mínimos para detecção de anomalias
  com LogGPT. Recomenda-se prosseguir com:
  
  1. Pré-processamento: Agrupar endpoints similares (e.g., /api/user/123 → /api/user/{id})
  2. Definição de sessões: Validar janela temporal de 5 min ou ajustar
  3. Labeling: Usar status codes 4xx/5xx como proxy de anomalias
  4. Pipeline: Adaptar código de HDFS/OpenStack para HTTP logs
  5. Validação: Executar os mesmos 6 testes de validação científica
  """
else:
    verdict = "⚠️ VIÁVEL COM RESSALVAS"
    recommendation = """
  O dataset SIAT pode ser usado, mas requer atenção aos desafios identificados.
  Recomenda-se:
  
  1. Resolver limitações críticas primeiro (ver seção 9)
  2. Considerar coleta de mais dados se volume for insuficiente
  3. Validar definição de "anomalia" (status codes podem não ser suficientes)
  4. Executar experimento piloto antes de produção
  """

print(f"\n  VEREDICTO: {verdict}")
print(recommendation)

# Salvar resultados
results = {
    'dataset': 'SIAT',
    'analysis_date': datetime.now().isoformat(),
    'basic_stats': {
        'total_logs': int(len(df)),
        'total_sessions': int(len(sessions)),
        'unique_endpoints': int(len(endpoint_counts)),
        'error_rate_pct': float(error_rate),
        'avg_session_size': float(sessions.mean()),
        'median_session_size': float(sessions.median())
    },
    'requirements': {k: v['met'] for k, v in requirements.items()},
    'challenges': challenges,
    'verdict': verdict.split()[1],  # VIÁVEL ou VIÁVEL COM RESSALVAS
    'comparison': comparison
}

with open('siat_viability_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n  Resultados salvos em: siat_viability_analysis.json")
print("=" * 70)
