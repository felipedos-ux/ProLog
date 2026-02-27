import json
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def load_results(json_path):
    """Carrega resultados do JSON gerado pelo detect.py"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_lead_time(data):
    """Analisa lead time para HDFS (medido em minutos usando timestamps reais)"""
    results = data['results']
    metrics = data['metrics']
    lead_time_stats = data['lead_time_metrics']
    
    # Filtrar apenas anomalias corretamente detectadas com lead time positivo
    true_positives = [r for r in results if r['label'] == 1 and r['is_detected'] and r['lead_time_minutes'] is not None]
    
    # Analisar lead time em minutos
    lead_times = [r['lead_time_minutes'] for r in true_positives]
    
    stats = {
        'num_tp': len(true_positives),
        'num_fp': sum(1 for r in results if r['label'] == 0 and r['is_detected']),
        'num_fn': sum(1 for r in results if r['label'] == 1 and not r['is_detected']),
        'num_tn': sum(1 for r in results if r['label'] == 0 and not r['is_detected']),
        'lead_times': lead_times,
        'mean_lead_time': lead_time_stats.get('avg_lead_minutes', 0),
        'median_lead_time': lead_time_stats.get('median_lead_minutes', 0),
        'min_lead_time': min(lead_times) if lead_times else 0,
        'max_lead_time': lead_time_stats.get('max_lead_minutes', 0),
        'std_lead_time': pd.Series(lead_times).std() if lead_times else 0,
        'anticipated_count': lead_time_stats.get('anticipated_count', 0),
        'not_anticipated_count': lead_time_stats.get('not_anticipated_count', 0),
        'threshold': data.get('threshold', 0)
    }
    
    return stats

def generate_html_report(stats, metrics, output_path):
    """Gera relatório HTML completo"""
    
    html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório Avançado de Detecção de Anomalias - HDFS</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .section {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .lead-time-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .lead-time-table th, .lead-time-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .lead-time-table th {{
            background-color: #667eea;
            color: white;
        }}
        .lead-time-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 20px 0;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
        .success {{
            background-color: #d4edda;
            padding: 15px;
            border-left: 4px solid #28a745;
            margin: 20px 0;
        }}
        .warning {{
            background-color: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Relatório Avançado de Detecção de Anomalias</h1>
        <p>Dataset: HDFS (Hadoop Distributed File System)</p>
        <p>Modelo: LogGPT (GPT-2 Small) com Threshold-Based Detection</p>
        <p>Threshold: {stats['threshold']:.4f}</p>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{metrics['precision']:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Recall</div>
            <div class="metric-value">{metrics['recall']:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">F1-Score</div>
            <div class="metric-value">{metrics['f1']:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{metrics['accuracy']:.4f}</div>
        </div>
    </div>

    <div class="section">
        <h2>🎯 Matriz de Confusão</h2>
        <table class="lead-time-table">
            <tr>
                <th></th>
                <th>Predito Normal</th>
                <th>Predito Anômalo</th>
            </tr>
            <tr>
                <th>Real Normal</th>
                <td><strong>{stats['num_tn']}</strong></td>
                <td><strong>{stats['num_fp']}</strong></td>
            </tr>
            <tr>
                <th>Real Anômalo</th>
                <td><strong>{stats['num_fn']}</strong></td>
                <td><strong>{stats['num_tp']}</strong></td>
            </tr>
        </table>
        
        <div class="success">
            <strong>✅ True Positives (TP):</strong> {stats['num_tp']} anomalias corretamente detectadas
        </div>
        
        <div class="warning">
            <strong>⚠️ False Positives (FP):</strong> {stats['num_fp']} blocos normais incorretamente classificados como anômalos
        </div>
        
        <div class="highlight">
            <strong>❌ False Negatives (FN):</strong> {stats['num_fn']} anomalias não detectadas (escapes)
        </div>
    </div>

    <div class="section">
        <h2>⏱️ Análise de Lead Time (HDFS)</h2>
        
        <div class="highlight">
            <strong>Nota sobre Lead Time no HDFS:</strong> O HDFS possui timestamps nos logs (Date e Time). 
            O lead time é medido em <strong>minutos</strong> usando timestamps reais, representando quanto tempo antes da falha 
            a anomalia foi detectada.
        </div>
        
        <table class="lead-time-table">
            <tr>
                <th>Métrica</th>
                <th>Valor</th>
                <th>Interpretação</th>
            </tr>
            <tr>
                <td>Média</td>
                <td><strong>{stats['mean_lead_time']:.2f}</strong> minutos</td>
                <td>Tempo médio até a detecção</td>
            </tr>
            <tr>
                <td>Mediana</td>
                <td><strong>{stats['median_lead_time']:.2f}</strong> minutos</td>
                <td>Valor central da distribuição</td>
            </tr>
            <tr>
                <td>Mínimo</td>
                <td><strong>{stats['min_lead_time']:.2f}</strong> minutos</td>
                <td>Detecção mais rápida</td>
            </tr>
            <tr>
                <td>Máximo</td>
                <td><strong>{stats['max_lead_time']:.2f}</strong> minutos</td>
                <td>Detecção mais tardia</td>
            </tr>
            <tr>
                <td>Desvio Padrão</td>
                <td><strong>{stats['std_lead_time']:.2f}</strong> minutos</td>
                <td>Variação nos tempos de detecção</td>
            </tr>
        </table>
        
        <div class="success">
            <strong>📊 Análise de Antecipação:</strong>
            <ul>
                <li>✅ Antecipadas (Lead > 0): {stats['anticipated_count']} ({stats['anticipated_count']/stats['num_tp']*100:.1f}% dos TPs)</li>
                <li>⚠️ Não Antecipadas (Lead ≤ 0): {stats['not_anticipated_count']} ({stats['not_anticipated_count']/stats['num_tp']*100:.1f}% dos TPs)</li>
            </ul>
        </div>
        
        <div class="highlight">
            <strong>Interpretação:</strong> Lead times maiores indicam detecção mais precoce. 
            Lead time positivo significa que a anomalia foi detectada antes da falha ocorrer.
            Lead time zero ou negativo significa que a anomalia foi detectada no momento ou após a falha.
        </div>
    </div>

    <div class="section">
        <h2>📈 Distribuição de Lead Times</h2>
        <div class="chart-container">
            <canvas id="leadTimeChart"></canvas>
        </div>
    </div>

    <div class="section">
        <h2>🔍 Insights e Recomendações</h2>
        
        <div class="success">
            <h3>Pontos Fortes</h3>
            <ul>
                <li>Recall de <strong>{metrics['recall']:.2%}</strong>: {stats['num_fn'] == 0 and 'Zero escapes de anomalias!' or 'Baixa taxa de escapes.'}</li>
                <li>Lead time médio de <strong>{stats['mean_lead_time']:.1f}</strong> minutos indica detecção relativamente precoce</li>
                <li><strong>{stats['anticipated_count']}/{stats['num_tp']}</strong> ({stats['anticipated_count']/stats['num_tp']*100:.1f}%) anomalias foram antecipadas antes da falha</li>
                <li>Desvio padrão de <strong>{stats['std_lead_time']:.2f}</strong> minutos mostra consistência nas detecções</li>
            </ul>
        </div>
        
        <div class="warning">
            <h3>Pontos de Atenção</h3>
            <ul>
                <li>{stats['num_fp']} falsos positivos podem gerar alertas desnecessários</li>
                <li>Precision de <strong>{metrics['precision']:.2%}</strong> pode ser melhorada com calibração de threshold</li>
                <li><strong>{stats['not_anticipated_count']}</strong> anomalias não foram antecipadas (detectadas na hora da falha ou depois)</li>
                <li>Considerar análise de blocos com lead time muito alto para identificar padrões de antecipação muito precoce</li>
            </ul>
        </div>
    </div>

    <script>
        // Chart.js configuration
        const ctx = document.getElementById('leadTimeChart').getContext('2d');
        
        const leadTimes = {json.dumps(stats['lead_times'])};
        
        // Create histogram bins
        const bins = {{}};
        leadTimes.forEach(lt => {{
            bins[lt] = (bins[lt] || 0) + 1;
        }});
        
        const labels = Object.keys(bins).sort((a, b) => a - b);
        const data = labels.map(l => bins[l]);
        
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'Frequência',
                    data: data,
                    backgroundColor: 'rgba(102, 126, 234, 0.6)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Distribuição de Lead Times (em minutos)',
                        font: {{
                            size: 16
                        }}
                    }},
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Lead Time (minutos)'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Frequência'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Relatório HTML gerado: {output_path}")

def main():
    # Caminhos
    json_path = Path("../data/HDFS_test_results.json")
    output_path = Path("docs/relatorio_avancado_hdfs.html")
    
    # Criar diretório de saída se não existir
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Carregar dados
    print("📂 Carregando resultados...")
    data = load_results(json_path)
    
    # Analisar lead time
    print("📊 Analisando lead times...")
    stats = analyze_lead_time(data)
    
    # Obter métricas
    metrics = data['metrics']
    
    # Gerar relatório
    print("📝 Gerando relatório HTML...")
    generate_html_report(stats, metrics, output_path)
    
    # Imprimir resumo no console
    print("\n" + "="*60)
    print("📊 RESUMO - DETECÇÃO DE ANOMALIAS HDFS")
    print("="*60)
    print(f"✅ True Positives: {stats['num_tp']}")
    print(f"⚠️  False Positives: {stats['num_fp']}")
    print(f"❌ False Negatives: {stats['num_fn']}")
    print(f"✅ True Negatives: {stats['num_tn']}")
    print("-"*60)
    print(f"Lead Time Médio: {stats['mean_lead_time']:.2f} minutos")
    print(f"Lead Time Mediano: {stats['median_lead_time']:.2f} minutos")
    print(f"Lead Time Mínimo: {stats['min_lead_time']:.2f} minutos")
    print(f"Lead Time Máximo: {stats['max_lead_time']:.2f} minutos")
    print(f"Desvio Padrão: {stats['std_lead_time']:.2f} minutos")
    print(f"✅ Antecipadas: {stats['anticipated_count']}/{stats['num_tp']} ({stats['anticipated_count']/stats['num_tp']*100:.1f}%)")
    print(f"⚠️  Não Antecipadas: {stats['not_anticipated_count']}/{stats['num_tp']} ({stats['not_anticipated_count']/stats['num_tp']*100:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    main()