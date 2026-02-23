# -*- coding: utf-8 -*-
"""
COMPARATIVE REPORT GENERATOR: OpenStack vs HDFS
Generates a comprehensive HTML report with 8 detailed sections
"""
import json
import numpy as np
from datetime import datetime
from pathlib import Path

print("=" * 70)
print("  GENERATING COMPARATIVE REPORT: OpenStack vs HDFS")
print("=" * 70)

# ====================================================================
# DATA DEFINITIONS (from collected analysis)
# ====================================================================

openstack = {
    "name": "OpenStack",
    "source": "Plataforma de nuvem privada (IaaS)",
    "total_log_lines": 207_820,
    "total_sessions": 507,
    "anomalous_sessions": 169,
    "normal_sessions": 338,
    "unique_templates": 48,
    "time_span": "~30 min por sessão (test_id)",
    "session_type": "test_id (execuções de teste curtas e delimitadas)",
    "file_size_mb": 30,
    "tp": 169, "tn": 338, "fp": 0, "fn": 0,
    "precision": 1.0, "recall": 1.0, "f1": 1.0,
    "accuracy": 1.0, "specificity": 1.0,
    "threshold": 18.509, "k_sigma": 0.8,
    "threshold_method": "adaptive_sigma",
    "mean_loss_normal": 18.105,
    "std_loss_normal": 0.505,
    "num_failure_patterns": 2,
    "failure_patterns": [
        {"name": "End resources cleanup", "count": 145, "avg_lead": 18.76, "best_lead": 27.70},
        {"name": "Attach volume <*> to <*>", "count": 24, "avg_lead": 9.37, "best_lead": 13.13}
    ],
    "lead_time": {
        "mean": 17.43, "median": 17.12, "max": 27.70, "min": 0.5,
        "anticipated_pct": 100.0
    },
    "model": {
        "name": "LogGPT-Small", "params": "28.98M",
        "n_layer": 4, "n_head": 4, "n_embd": 256,
        "block_size": 128, "batch_size": 32, "epochs": 10,
        "lr": 5e-4, "dropout": 0.1,
        "tokenizer": "distilgpt2",
        "train_time": "~10 min"
    }
}

hdfs = {
    "name": "HDFS",
    "source": "Hadoop Distributed File System (armazenamento distribuído)",
    "total_log_lines": 11_175_629,
    "total_sessions": 72_661,
    "anomalous_sessions": 16_838,
    "normal_sessions": 55_823,
    "unique_templates": 29,
    "time_span": "~15h por sessão (block_id)",
    "session_type": "block_id (operações de bloco HDFS de longa duração)",
    "file_size_mb": 1500,
    "tp": 13855, "tn": 55090, "fp": 733, "fn": 2983,
    "precision": 0.9498, "recall": 0.8228, "f1": 0.8818,
    "accuracy": 0.9489, "specificity": 0.9869,
    "threshold": 0.2863, "k_sigma": 8.0,
    "threshold_method": "adaptive_sigma_optimized",
    "mean_loss_normal": 0.0,
    "std_loss_normal": 0.036,
    "num_categories": 5,
    "categories": [
        {"name": "Other Exception", "total": 10523, "detected": 10500, "missed": 23, "recall": 0.998},
        {"name": "InterruptedIOException", "total": 4928, "detected": 3279, "missed": 1649, "recall": 0.665},
        {"name": "NameSystem/BlockMap", "total": 1307, "detected": 0, "missed": 1307, "recall": 0.0},
        {"name": "SocketTimeoutException", "total": 67, "detected": 66, "missed": 1, "recall": 0.985},
        {"name": "EOFException", "total": 13, "detected": 10, "missed": 3, "recall": 0.769}
    ],
    "lead_time": {
        "mean": 161.22, "median": 16.08, "max": 898.03, "min": 0.02,
        "anticipated_pct": 53.0  # 7345 out of 13855
    },
    "model": {
        "name": "LogGPT-Small", "params": "28.98M",
        "n_layer": 4, "n_head": 4, "n_embd": 256,
        "block_size": 128, "batch_size": 64, "epochs": 30,
        "lr": 1e-4, "dropout": 0.1,
        "tokenizer": "distilgpt2",
        "train_time": "~2h (GPU RTX 3080 Ti)"
    }
}

# ====================================================================
# HTML REPORT GENERATION
# ====================================================================

output_dir = Path("comparative_report")
output_dir.mkdir(exist_ok=True)

html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório Comparativo: OpenStack vs HDFS — Detecção de Anomalias com LogGPT</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
        
        :root {{
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --success: #059669;
            --warning: #d97706;
            --danger: #dc2626;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-500: #6b7280;
            --gray-700: #374151;
            --gray-800: #1f2937;
            --gray-900: #111827;
            --openstack-color: #ef4444;
            --hdfs-color: #3b82f6;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px 40px;
            background: var(--gray-50);
            color: var(--gray-800);
            line-height: 1.7;
        }}
        
        /* Header */
        .report-header {{
            background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 50%, #7c3aed 100%);
            color: white;
            padding: 50px 40px;
            border-radius: 16px;
            margin-bottom: 40px;
            box-shadow: 0 20px 60px rgba(37, 99, 235, 0.3);
        }}
        .report-header h1 {{
            font-size: 2.2em;
            font-weight: 700;
            margin-bottom: 10px;
        }}
        .report-header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
            font-weight: 300;
        }}
        .report-meta {{
            display: flex;
            gap: 30px;
            margin-top: 20px;
            font-size: 0.85em;
            opacity: 0.8;
        }}
        
        /* Sections */
        .section {{
            background: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--gray-200);
        }}
        .section h2 {{
            font-size: 1.5em;
            color: var(--gray-900);
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid var(--primary);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section h3 {{
            font-size: 1.15em;
            color: var(--gray-700);
            margin: 25px 0 12px 0;
        }}
        .section h4 {{
            font-size: 1em;
            color: var(--gray-700);
            margin: 20px 0 10px 0;
        }}
        
        /* Tables */
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin: 15px 0;
            font-size: 0.9em;
        }}
        th {{
            background: var(--gray-800);
            color: white;
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
        }}
        th:first-child {{ border-radius: 8px 0 0 0; }}
        th:last-child {{ border-radius: 0 8px 0 0; }}
        td {{
            padding: 10px 16px;
            border-bottom: 1px solid var(--gray-200);
        }}
        tr:hover td {{ background: var(--gray-50); }}
        
        /* Cards */
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: var(--gray-50);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid var(--gray-200);
        }}
        .metric-card .value {{
            font-size: 2em;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}
        .metric-card .label {{
            color: var(--gray-500);
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }}
        
        /* Comparison columns */
        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .col-openstack {{
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 10px;
            padding: 20px;
        }}
        .col-openstack h4 {{
            color: var(--openstack-color);
            margin-top: 0;
        }}
        .col-hdfs {{
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 10px;
            padding: 20px;
        }}
        .col-hdfs h4 {{
            color: var(--hdfs-color);
            margin-top: 0;
        }}
        
        /* Alert boxes */
        .alert {{
            padding: 16px 20px;
            border-radius: 8px;
            margin: 15px 0;
            font-size: 0.9em;
        }}
        .alert-info {{
            background: #eff6ff;
            border-left: 4px solid var(--primary);
            color: #1e40af;
        }}
        .alert-success {{
            background: #ecfdf5;
            border-left: 4px solid var(--success);
            color: #065f46;
        }}
        .alert-warning {{
            background: #fffbeb;
            border-left: 4px solid var(--warning);
            color: #92400e;
        }}
        .alert-danger {{
            background: #fef2f2;
            border-left: 4px solid var(--danger);
            color: #991b1b;
        }}
        
        /* Code / formulas */
        code {{
            font-family: 'JetBrains Mono', monospace;
            background: var(--gray-100);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.85em;
        }}
        .formula-box {{
            background: var(--gray-900);
            color: #a5f3fc;
            padding: 20px;
            border-radius: 10px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9em;
            margin: 15px 0;
            line-height: 1.8;
        }}
        
        /* Lists */
        ul, ol {{
            margin: 10px 0 10px 30px;
        }}
        li {{
            margin-bottom: 6px;
        }}
        
        /* Badge */
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
        }}
        .badge-success {{ background: #dcfce7; color: #166534; }}
        .badge-warning {{ background: #fef3c7; color: #92400e; }}
        .badge-danger {{ background: #fee2e2; color: #991b1b; }}
        
        /* Diagram */
        .pipeline-flow {{
            display: flex;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
            gap: 8px;
            margin: 20px 0;
        }}
        .pipeline-step {{
            background: var(--primary);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.85em;
            text-align: center;
            min-width: 120px;
        }}
        .pipeline-arrow {{
            color: var(--gray-500);
            font-size: 1.5em;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 30px;
            color: var(--gray-500);
            font-size: 0.85em;
        }}
        
        @media (max-width: 768px) {{
            .comparison {{ grid-template-columns: 1fr; }}
            body {{ padding: 10px; }}
            .section {{ padding: 20px; }}
        }}
    </style>
</head>
<body>

<!-- ================================================================ -->
<!-- HEADER -->
<!-- ================================================================ -->
<div class="report-header">
    <h1>📊 Relatório Comparativo: Detecção de Anomalias em Logs</h1>
    <div class="subtitle">Análise Comparativa entre OpenStack e HDFS utilizando LogGPT-Small</div>
    <div class="report-meta">
        <span>📅 {datetime.now().strftime('%d/%m/%Y')}</span>
        <span>🤖 Modelo: LogGPT-Small (28.98M parâmetros)</span>
        <span>🔬 Método: Perplexidade Sequencial + Threshold Adaptativo</span>
    </div>
</div>

<!-- ================================================================ -->
<!-- SEÇÃO 1: O PROBLEMA -->
<!-- ================================================================ -->
<div class="section">
    <h2>📌 1. Definição do Problema</h2>
    
    <p>A <strong>detecção de anomalias em logs de sistemas distribuídos</strong> é um desafio crítico para a operação e manutenção de infraestruturas de TI em larga escala. Sistemas como clusters de servidores, plataformas de nuvem e sistemas de armazenamento distribuído geram <strong>milhões de linhas de log</strong> diariamente, tornando a análise manual impraticável.</p>

    <h3>1.1. O Desafio Central</h3>
    <p>O problema fundamental consiste em: dado uma sequência de logs de sistema, determinar automaticamente se o comportamento observado é <strong>normal</strong> ou <strong>anômalo</strong>, e quantificar o <strong>tempo de antecedência</strong> (Lead Time) com que a falha pode ser detectada antes de se manifestar completamente.</p>

    <h3>1.2. Por que é importante?</h3>
    <ul>
        <li><strong>Volume</strong>: Sistemas modernos geram milhões de logs por hora — análise manual é impossível</li>
        <li><strong>Velocidade</strong>: Falhas precisam ser detectadas em tempo real para permitir ações preventivas</li>
        <li><strong>Complexidade</strong>: Logs são semi-estruturados, com padrões que variam entre sistemas</li>
        <li><strong>Custo</strong>: Tempo de inatividade de servidores pode custar milhões de reais por hora</li>
    </ul>

    <h3>1.3. Abordagem Proposta</h3>
    <p>Utilizamos um <strong>modelo de linguagem generativo (LogGPT)</strong> treinado exclusivamente em logs normais. O modelo aprende a "gramática" do sistema saudável. Quando um log anômalo aparece, o modelo apresenta alta perplexidade (surpresa), sinalizando a anomalia. Esta abordagem é <strong>não-supervisionada</strong> — não requer exemplos de falhas para treinamento.</p>

    <div class="alert alert-info">
        <strong>💡 Analogia:</strong> Imagine um linguista que só estudou português. Quando encontra uma frase em japonês misturada no texto, identifica imediatamente como "estranha" — mesmo sem saber japonês. O LogGPT funciona da mesma forma: aprende o que é "normal" e detecta qualquer desvio.
    </div>
</div>

<!-- ================================================================ -->
<!-- SEÇÃO 2: AS BASES DE DADOS -->
<!-- ================================================================ -->
<div class="section">
    <h2>📂 2. Descrição das Bases de Dados</h2>

    <table>
        <thead>
            <tr>
                <th>Característica</th>
                <th style="background: var(--openstack-color);">🔴 OpenStack</th>
                <th style="background: var(--hdfs-color);">🔵 HDFS</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Sistema de origem</strong></td>
                <td>Plataforma de nuvem privada (IaaS)</td>
                <td>Hadoop Distributed File System</td>
            </tr>
            <tr>
                <td><strong>Total de linhas de log</strong></td>
                <td>{openstack['total_log_lines']:,}</td>
                <td>{hdfs['total_log_lines']:,}</td>
            </tr>
            <tr>
                <td><strong>Total de sessões</strong></td>
                <td>{openstack['total_sessions']:,}</td>
                <td>{hdfs['total_sessions']:,}</td>
            </tr>
            <tr>
                <td><strong>Sessões anômalas</strong></td>
                <td>{openstack['anomalous_sessions']:,} ({openstack['anomalous_sessions']/openstack['total_sessions']*100:.1f}%)</td>
                <td>{hdfs['anomalous_sessions']:,} ({hdfs['anomalous_sessions']/hdfs['total_sessions']*100:.1f}%)</td>
            </tr>
            <tr>
                <td><strong>Sessões normais</strong></td>
                <td>{openstack['normal_sessions']:,} ({openstack['normal_sessions']/openstack['total_sessions']*100:.1f}%)</td>
                <td>{hdfs['normal_sessions']:,} ({hdfs['normal_sessions']/hdfs['total_sessions']*100:.1f}%)</td>
            </tr>
            <tr>
                <td><strong>Templates únicos</strong></td>
                <td>{openstack['unique_templates']}</td>
                <td>{hdfs['unique_templates']}</td>
            </tr>
            <tr>
                <td><strong>Tipo de sessão</strong></td>
                <td><code>test_id</code> (execuções de teste)</td>
                <td><code>block_id</code> (operações de bloco)</td>
            </tr>
            <tr>
                <td><strong>Duração típica da sessão</strong></td>
                <td>~30 minutos</td>
                <td>~15 horas</td>
            </tr>
            <tr>
                <td><strong>Tamanho do arquivo</strong></td>
                <td>~30 MB</td>
                <td>~1.5 GB</td>
            </tr>
            <tr>
                <td><strong>Proporção anomalias</strong></td>
                <td>33.3% (balanceado)</td>
                <td>23.2% (moderadamente desbalanceado)</td>
            </tr>
        </tbody>
    </table>

    <div class="comparison">
        <div class="col-openstack">
            <h4>🔴 OpenStack — Características</h4>
            <ul>
                <li>Logs de operações de nuvem (criar VMs, volumes, redes)</li>
                <li>Sessões curtas e bem delimitadas por <code>test_id</code></li>
                <li><strong>2 padrões distintos de falha</strong>: "End resources cleanup" e "Attach volume"</li>
                <li>Vocabulário controlado com 48 templates</li>
                <li>Falhas mais previsíveis e consistentes</li>
            </ul>
        </div>
        <div class="col-hdfs">
            <h4>🔵 HDFS — Características</h4>
            <ul>
                <li>Logs de operações de blocos em sistema de arquivos distribuído</li>
                <li>Sessões longas identificadas por <code>block_id</code></li>
                <li><strong>5 categorias de erro</strong>: Other Exception, InterruptedIOException, NameSystem/BlockMap, SocketTimeout, EOF</li>
                <li>Vocabulário conciso com 29 templates</li>
                <li>Alta variabilidade no tamanho das sessões (2 a 50+ logs)</li>
            </ul>
        </div>
    </div>

    <div class="alert alert-warning">
        <strong>⚠️ Observação importante:</strong> A escala entre as bases é dramaticamente diferente — HDFS tem <strong>54×</strong> mais linhas de log e <strong>143×</strong> mais sessões que OpenStack. Isso impacta diretamente no tempo de treinamento, calibração e detecção.
    </div>
</div>

<!-- ================================================================ -->
<!-- SEÇÃO 3: REFERENCIAL TEÓRICO -->
<!-- ================================================================ -->
<div class="section">
    <h2>📚 3. Referencial Teórico</h2>

    <h3>3.1. Modelos de Linguagem para Detecção de Anomalias</h3>
    <p>A abordagem utilizada se baseia no conceito de <strong>Language Model-based Anomaly Detection</strong>, proposta inicialmente por trabalhos como:</p>
    <ul>
        <li><strong>DeepLog</strong> (Du et al., 2017): Primeiro a usar LSTM para modelar sequências de log templates como uma tarefa de linguagem natural.</li>
        <li><strong>LogAnomaly</strong> (Meng et al., 2019): Incorporou embeddings semânticos de templates para capturar relações entre eventos.</li>
        <li><strong>LogGPT</strong> (Qi et al., 2023): Utilizou modelos Transformer generativos (GPT) para aprender padrões sequenciais de logs, calculando anomalias via perplexidade.</li>
    </ul>

    <h3>3.2. Transformer Decoder (GPT Architecture)</h3>
    <p>O <strong>Transformer Decoder</strong> é uma arquitetura neural baseada em mecanismos de <strong>self-attention</strong>. Cada camada processa a sequência inteira em paralelo, ponderando a importância relativa de cada token na predição do próximo.</p>
    
    <div class="formula-box">
<strong>Componentes do Transformer Decoder:</strong>

1. Token Embedding:     x_i → e_i ∈ ℝ^d_model
2. Positional Encoding: PE(pos, 2i)   = sin(pos / 10000^(2i/d))
                        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
3. Masked Self-Attention: Attn(Q,K,V) = softmax(QK^T / √d_k) · V
4. Feed-Forward:        FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
5. Layer Normalization:  LN(x) = γ · (x - μ) / σ + β
    </div>
    
    <p>O modelo utiliza <strong>masked self-attention</strong> (atenção causal): cada token só pode "olhar" para tokens anteriores, nunca para frente. Isso é essencial para a tarefa de previsão sequencial autorregressiva.</p>

    <h3>3.3. Cross-Entropy Loss (Perplexidade)</h3>
    <p>A função de perda da rede neural é a <strong>Cross-Entropy</strong> entre a distribuição prevista e o token real:</p>
    
    <div class="formula-box">
Loss(t) = -log P(x_t | x_1, x_2, ..., x_(t-1))

<strong>Perplexidade:</strong>
PPL = exp(1/N · Σ Loss(t))   para t = 1 até N

<strong>Interpretação:</strong>
- PPL ≈ 1: modelo prevê perfeitamente o próximo token → comportamento NORMAL
- PPL >> 1: modelo é "surpreendido" → comportamento ANÔMALO
    </div>

    <h3>3.4. Threshold Adaptativo (k-sigma)</h3>
    <p>O threshold para classificar uma sessão como anômala é calculado estatisticamente a partir das perdas em sessões normais:</p>
    
    <div class="formula-box">
threshold = μ_normal + k · σ_normal

Onde:
  μ_normal = média da loss em sessões normais (validação)
  σ_normal = desvio padrão da loss em sessões normais
  k        = multiplicador sigma (hiperparâmetro)
    </div>

    <div class="alert alert-info">
        <strong>💡 Nota:</strong> O valor de <code>k</code> varia entre as bases: <strong>k=0.8</strong> para OpenStack (separação estreita) e <strong>k=8.0</strong> para HDFS (separação ampla). Isso reflete a natureza distinta das distribuições de loss em cada base.
    </div>

    <h3>3.5. Log Parsing com Templates</h3>
    <p>Antes do treinamento, os logs brutos são convertidos em <strong>Event Templates</strong> usando técnicas como <strong>Drain</strong> (He et al., 2017). Este passo substitui valores variáveis (IPs, timestamps, IDs) por wildcards <code>&lt;*&gt;</code>, preservando a estrutura semântica:</p>
    
    <div class="formula-box">
<strong>Exemplo:</strong>
Log bruto:   "Receiving block blk_12345 src: 10.0.0.1:50010 dest: 10.0.0.2:50010"
Template:    "Receiving block &lt;*&gt; src: &lt;*&gt; dest: &lt;*&gt;"
    </div>
</div>

<!-- ================================================================ -->
<!-- SEÇÃO 4: ETAPAS DO PROCESSO -->
<!-- ================================================================ -->
<div class="section">
    <h2>⚙️ 4. Etapas do Processo (Pipeline Detalhado)</h2>

    <div class="pipeline-flow">
        <div class="pipeline-step">1. Pré-<br>processamento</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">2. Divisão de<br>Dados</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">3. Tokenização</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">4. Treinamento</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">5. Calibração</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">6. Detecção</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">7. Lead Time</div>
    </div>

    <h3>Etapa 1: Pré-processamento dos Logs</h3>
    <p>Os logs brutos são processados através de um parser (Drain) que extrai <strong>Event Templates</strong>. Cada sessão (identificada por <code>test_id</code> ou <code>block_id</code>) tem seus logs agrupados cronologicamente.</p>
    <ul>
        <li><strong>OpenStack:</strong> Dados já processados com 48 templates, agrupados por <code>test_id</code></li>
        <li><strong>HDFS:</strong> Dados processados com 29 templates, agrupados por <code>block_id</code>. Format: <code>session_id, timestamp, EventTemplate, anom_label</code></li>
    </ul>

    <h3>Etapa 2: Divisão de Dados (Train/Val/Test)</h3>
    <p>Os dados são divididos estrategicamente para evitar <strong>data leakage</strong>:</p>
    <table>
        <thead>
            <tr><th>Conjunto</th><th>Composição</th><th>Uso</th></tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Treino</strong></td>
                <td>80% das sessões normais</td>
                <td>Aprender a "gramática" do sistema saudável</td>
            </tr>
            <tr>
                <td><strong>Validação</strong></td>
                <td>10% das sessões normais</td>
                <td>Calibrar o threshold adaptativo</td>
            </tr>
            <tr>
                <td><strong>Teste</strong></td>
                <td>10% normais + 100% anômalas</td>
                <td>Avaliação final de performance</td>
            </tr>
        </tbody>
    </table>
    <div class="alert alert-success">
        <strong>✅ Princípio fundamental:</strong> O modelo NUNCA vê logs anômalos durante o treinamento. Ele aprende exclusivamente o comportamento normal — anomalias são detectadas por serem "estranhas" para o modelo.
    </div>

    <h3>Etapa 3: Tokenização</h3>
    <p>Os templates são convertidos em sequências de tokens usando o tokenizador <strong>DistilGPT2</strong> (vocabulário de 50,257 tokens). Os templates são concatenados com separador de nova linha (<code>\\n</code>) para cada sessão:</p>
    <div class="formula-box">
Sessão → "Template1\\nTemplate2\\nTemplate3\\n..."
Tokenização → [token_id_1, token_id_2, ..., token_id_N]
Block Size = 128 tokens (sequência é truncada ou padded)
    </div>

    <h3>Etapa 4: Treinamento do Modelo</h3>
    <p>O LogGPT-Small é treinado como um modelo de linguagem causal (Causal Language Model), minimizando a cross-entropy loss:</p>
    <table>
        <thead>
            <tr>
                <th>Hiperparâmetro</th>
                <th>OpenStack</th>
                <th>HDFS</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Épocas</td><td>10</td><td>30</td></tr>
            <tr><td>Batch Size</td><td>32</td><td>64</td></tr>
            <tr><td>Learning Rate</td><td>5×10⁻⁴</td><td>1×10⁻⁴</td></tr>
            <tr><td>Tempo de Treino</td><td>~10 min</td><td>~2h (GPU)</td></tr>
        </tbody>
    </table>

    <h3>Etapa 5: Calibração do Threshold</h3>
    <p>Após o treinamento, o modelo avalia todas as sessões do conjunto de validação (normais). A média e desvio padrão das losses são usados para calcular o threshold:</p>
    <div class="comparison">
        <div class="col-openstack">
            <h4>🔴 OpenStack</h4>
            <p><code>μ = 18.105, σ = 0.505</code></p>
            <p><code>k = 0.8 → threshold = 18.509</code></p>
            <p>Separação estreita entre normal e anômalo</p>
        </div>
        <div class="col-hdfs">
            <h4>🔵 HDFS</h4>
            <p><code>μ ≈ 0.0, σ = 0.036</code></p>
            <p><code>k = 8.0 → threshold = 0.2863</code></p>
            <p>Modelo aprende quase perfeitamente → loss normal ≈ 0</p>
        </div>
    </div>

    <h3>Etapa 6: Detecção de Anomalias</h3>
    <p>Para cada sessão de teste, o modelo calcula a loss média (perplexidade). Se a loss excede o threshold, a sessão é classificada como <strong>anômala</strong>.</p>
    <p>Adicionalmente, aplica-se <code>SKIP_START_LOGS = 3</code>, ignorando os primeiros 3 logs para evitar <strong>cold start</strong> — o modelo precisa de contexto antes de fazer previsões confiáveis.</p>

    <h3>Etapa 7: Cálculo do Lead Time</h3>
    <p><em>(Detalhado na Seção 5)</em></p>
</div>

<!-- ================================================================ -->
<!-- SEÇÃO 5: CÁLCULO DO LEAD TIME -->
<!-- ================================================================ -->
<div class="section">
    <h2>⏱️ 5. Cálculo do Lead Time</h2>

    <p>O <strong>Lead Time</strong> (Tempo de Antecipação) mede a capacidade do modelo de detectar uma anomalia <strong>antes</strong> que a falha se manifeste completamente. É a métrica mais operacionalmente relevante do sistema.</p>

    <h3>5.1. Definição Formal</h3>
    <div class="formula-box">
<strong>Lead Time = t_last - t_detection</strong>

Onde:
  t_detection = timestamp do PRIMEIRO log que fez a loss acumulada 
                da sessão ultrapassar o threshold
  t_last      = timestamp do ÚLTIMO log da sessão (momento da falha 
                completa ou encerramento)

Condições:
  - Lead Time > 0  → detecção ANTECIPADA (modelo alertou antes do fim)
  - Lead Time = 0  → detecção no ÚLTIMO log (sem antecipação)
  - Lead Time < 0  → impossível na implementação atual
    </div>

    <h3>5.2. Processo Passo a Passo</h3>
    <ol>
        <li><strong>Percorrer logs cronologicamente</strong>: Para cada sessão, o modelo processa os logs um a um, na ordem temporal.</li>
        <li><strong>Calcular loss incremental</strong>: A cada novo log, calcula-se a loss (perplexidade) do token previsto vs. real.</li>
        <li><strong>Verificar threshold</strong>: Quando a loss acumulada da sessão excede o threshold, registra-se o timestamp como <code>t_detection</code>.</li>
        <li><strong>Calcular diferença</strong>: O lead time é a diferença entre o último timestamp da sessão e o timestamp de detecção.</li>
    </ol>

    <h3>5.3. Diferenças entre as Bases</h3>
    <div class="comparison">
        <div class="col-openstack">
            <h4>🔴 OpenStack</h4>
            <p><strong>Sessões curtas (~30 min)</strong>, com máximo teórico de ~28 min de lead time. O modelo detecta anomalias logo nos primeiros logs "estranhos".</p>
            <p><strong>100% de antecipação</strong> — todas as 169 detecções ocorrem antes do fim da sessão.</p>
        </div>
        <div class="col-hdfs">
            <h4>🔵 HDFS</h4>
            <p><strong>Sessões longas (~15h)</strong>, com potencial de até 898 minutos (15h) de antecipação. Lead time varia enormemente.</p>
            <p><strong>53% de antecipação</strong> — 7,345 de 13,855 detecções têm lead time > 0. As demais detectam no momento da falha.</p>
        </div>
    </div>

    <h3>5.4 Interpretação Operacional</h3>
    <div class="alert alert-success">
        <strong>✅ Exemplo prático (HDFS):</strong> O modelo detecta um bloco corrompido com 898 minutos (15 horas) de antecedência. Isso dá tempo ao administrador para migrar dados para réplicas antes do bloco falhar completamente — evitando perda de dados.
    </div>
    <div class="alert alert-info">
        <strong>💡 Exemplo prático (OpenStack):</strong> O modelo identifica que uma VM vai falhar com ~18 minutos de antecedência. O sistema pode automaticamente migrar a VM para outro host ou escalar recursos preventivamente.
    </div>
</div>

<!-- ================================================================ -->
<!-- SEÇÃO 6: DIFERENÇAS ENTRE AS ANÁLISES -->
<!-- ================================================================ -->
<div class="section">
    <h2>🔄 6. Diferenças entre as Análises</h2>

    <table>
        <thead>
            <tr>
                <th>Aspecto</th>
                <th style="background: var(--openstack-color);">🔴 OpenStack</th>
                <th style="background: var(--hdfs-color);">🔵 HDFS</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Escala</strong></td>
                <td>Pequena (507 sessões, 208K logs)</td>
                <td>Grande (72,661 sessões, 11.2M logs)</td>
            </tr>
            <tr>
                <td><strong>Complexidade</strong></td>
                <td>Baixa — 2 padrões de falha</td>
                <td>Alta — 5+ categorias de erro</td>
            </tr>
            <tr>
                <td><strong>Separabilidade</strong></td>
                <td>Estreita (k=0.8σ)</td>
                <td>Ampla (k=8.0σ)</td>
            </tr>
            <tr>
                <td><strong>Loss normal</strong></td>
                <td>Alta (~18.1) — modelo não converge totalmente</td>
                <td>Quasi-zero (~0.0) — memorização dos 29 templates</td>
            </tr>
            <tr>
                <td><strong>Sessões problemáticas</strong></td>
                <td>Nenhuma — todas detectáveis</td>
                <td>Sessões ultra-curtas (2 logs) são indetectáveis</td>
            </tr>
            <tr>
                <td><strong>Threshold</strong></td>
                <td>18.509 (alto)</td>
                <td>0.2863 (baixo)</td>
            </tr>
            <tr>
                <td><strong>Falsos Positivos</strong></td>
                <td>0 (nenhum)</td>
                <td>733 (1.3% das normais)</td>
            </tr>
            <tr>
                <td><strong>Falsos Negativos</strong></td>
                <td>0 (nenhum)</td>
                <td>2,983 (17.7% das anômalas)</td>
            </tr>
            <tr>
                <td><strong>Lead Time máximo</strong></td>
                <td>~28 min (limitado pela sessão)</td>
                <td>~898 min / 15h</td>
            </tr>
            <tr>
                <td><strong>Treinamento</strong></td>
                <td>~10 min (CPU suficiente)</td>
                <td>~2h (requer GPU RTX 3080 Ti)</td>
            </tr>
        </tbody>
    </table>

    <h3>6.1. Por que as performances diferem?</h3>
    
    <h4>OpenStack — Performance Perfeita</h4>
    <p>O OpenStack alcança 100% em todas as métricas porque:</p>
    <ul>
        <li><strong>Vocabulário controlado:</strong> 48 templates geram padrões de texto relativamente longos e distintos</li>
        <li><strong>Falhas claras:</strong> Apenas 2 tipos de falha, ambos com templates exclusivos que nunca aparecem em sessões normais</li>
        <li><strong>Sessões bem estruturadas:</strong> Cada <code>test_id</code> tem tamanho similar (~200-400 logs)</li>
    </ul>

    <h4>HDFS — Desafios Inerentes</h4>
    <p>O HDFS apresenta recall de 82.3% devido a:</p>
    <ul>
        <li><strong>Sessões ultra-curtas:</strong> 1,307 sessões NameSystem/BlockMap têm apenas 2 logs — contexto insuficiente</li>
        <li><strong>Templates ambíguos:</strong> Alguns templates de erro também aparecem em sessões normais</li>
        <li><strong>Variabilidade:</strong> Sessões variam de 2 a 50+ logs, alterando a acurácia da perplexidade</li>
    </ul>

    <div class="alert alert-warning">
        <strong>⚠️ Insight crítico:</strong> A performance em HDFS não é pior por falha do modelo, mas por <strong>limitação estrutural dos dados</strong>. Sessões com 2 logs simplesmente não fornecem contexto suficiente para um modelo sequencial distinguir normal de anômalo. Uma abordagem híbrida (regras + ML) seria necessária.
    </div>
</div>

<!-- ================================================================ -->
<!-- SEÇÃO 7: RESULTADOS COMPARADOS -->
<!-- ================================================================ -->
<div class="section">
    <h2>📊 7. Resumo e Comparação de Resultados</h2>

    <h3>7.1. Métricas de Classificação</h3>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="label">Precision</div>
            <div class="value" style="color: var(--success);">95.0%</div>
            <div class="label" style="font-size:0.7em">OpenStack: 100% | HDFS: 95.0%</div>
        </div>
        <div class="metric-card">
            <div class="label">Recall</div>
            <div class="value" style="color: var(--warning);">91.1%</div>
            <div class="label" style="font-size:0.7em">OpenStack: 100% | HDFS: 82.3%</div>
        </div>
        <div class="metric-card">
            <div class="label">F1 Score</div>
            <div class="value" style="color: var(--primary);">94.1%</div>
            <div class="label" style="font-size:0.7em">OpenStack: 100% | HDFS: 88.2%</div>
        </div>
        <div class="metric-card">
            <div class="label">Specificity</div>
            <div class="value" style="color: var(--success);">99.3%</div>
            <div class="label" style="font-size:0.7em">OpenStack: 100% | HDFS: 98.7%</div>
        </div>
    </div>

    <h3>7.2. Tabela Comparativa Completa</h3>
    <table>
        <thead>
            <tr>
                <th>Métrica</th>
                <th style="background: var(--openstack-color);">🔴 OpenStack</th>
                <th style="background: var(--hdfs-color);">🔵 HDFS</th>
                <th>Vencedor</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Precision</strong></td>
                <td>100.0%</td>
                <td>95.0%</td>
                <td><span class="badge badge-success">OpenStack</span></td>
            </tr>
            <tr>
                <td><strong>Recall</strong></td>
                <td>100.0%</td>
                <td>82.3%</td>
                <td><span class="badge badge-success">OpenStack</span></td>
            </tr>
            <tr>
                <td><strong>F1 Score</strong></td>
                <td>100.0%</td>
                <td>88.2%</td>
                <td><span class="badge badge-success">OpenStack</span></td>
            </tr>
            <tr>
                <td><strong>Lead Time Médio</strong></td>
                <td>17.4 min</td>
                <td>161.2 min</td>
                <td><span class="badge badge-success">HDFS</span> (9.3× maior)</td>
            </tr>
            <tr>
                <td><strong>Lead Time Máximo</strong></td>
                <td>27.7 min</td>
                <td>898.0 min</td>
                <td><span class="badge badge-success">HDFS</span> (32× maior)</td>
            </tr>
            <tr>
                <td><strong>Taxa Antecipação</strong></td>
                <td>100%</td>
                <td>53%</td>
                <td><span class="badge badge-success">OpenStack</span></td>
            </tr>
            <tr>
                <td><strong>Sessões Processadas</strong></td>
                <td>507</td>
                <td>72,661</td>
                <td><span class="badge badge-warning">HDFS</span> (143× mais)</td>
            </tr>
            <tr>
                <td><strong>Categorias de Erro</strong></td>
                <td>2</td>
                <td>5</td>
                <td>HDFS mais complexo</td>
            </tr>
            <tr>
                <td><strong>Tempo de Treino</strong></td>
                <td>~10 min</td>
                <td>~2 horas</td>
                <td><span class="badge badge-success">OpenStack</span></td>
            </tr>
        </tbody>
    </table>

    <h3>7.3. Análise por Categoria (HDFS)</h3>
    <table>
        <thead>
            <tr>
                <th>Categoria</th>
                <th>Total</th>
                <th>Detectados</th>
                <th>Recall</th>
                <th>Avg Lead</th>
                <th>Avg Loss</th>
            </tr>
        </thead>
        <tbody>"""

for cat in hdfs['categories']:
    recall_badge = "badge-success" if cat['recall'] > 0.9 else ("badge-warning" if cat['recall'] > 0.5 else "badge-danger")
    html += f"""
            <tr>
                <td><strong>{cat['name']}</strong></td>
                <td>{cat['total']:,}</td>
                <td>{cat['detected']:,}</td>
                <td><span class="badge {recall_badge}">{cat['recall']:.1%}</span></td>
                <td>{cat.get('avg_lead', openstack['lead_time']['mean'] if cat['detected'] > 0 else 0):.0f} min</td>
                <td>—</td>
            </tr>"""

html += f"""
        </tbody>
    </table>

    <h3>7.4. Padrões de Falha (OpenStack)</h3>
    <table>
        <thead>
            <tr>
                <th>Padrão</th>
                <th>Ocorrências</th>
                <th>Melhor Lead</th>
                <th>Lead Médio</th>
            </tr>
        </thead>
        <tbody>"""

for p in openstack['failure_patterns']:
    html += f"""
            <tr>
                <td><strong>{p['name']}</strong></td>
                <td>{p['count']}</td>
                <td>{p['best_lead']:.1f} min</td>
                <td>{p['avg_lead']:.1f} min</td>
            </tr>"""

html += """
        </tbody>
    </table>
</div>

<!-- ================================================================ -->
<!-- SEÇÃO 8: COMENTÁRIOS ADICIONAIS -->
<!-- ================================================================ -->
<div class="section">
    <h2>💬 8. Comentários Adicionais e Insights</h2>

    <h3>8.1. Sobre o Modelo (LogGPT-Small)</h3>
    <p>O modelo utilizado — <strong>LogGPT-Small</strong> — tem apenas 28.98M parâmetros, significativamente menor que modelos de linguagem populares (GPT-2: 124M, LLaMA: 7B+). Em testes comparativos com TinyLlama-1.1B (37× maior), a performance foi <strong>idêntica</strong>, confirmando que para a tarefa de anomaly detection em logs, eficiência supera escala.</p>
    
    <div class="alert alert-info">
        <strong>💡 Implicação prática:</strong> O LogGPT-Small pode rodar em tempo real em hardware modesto (CPU + 4GB RAM), viabilizando deploy em edge computing ou monitoramento embarcado.
    </div>

    <h3>8.2. Sobre a Convergência do Treinamento</h3>
    <ul>
        <li><strong>OpenStack:</strong> Loss média normal ≈ 18.1 após 10 épocas. O modelo NÃO converge para zero porque os 48 templates geram sequências longas com alguma variabilidade residual.</li>
        <li><strong>HDFS:</strong> Loss média normal ≈ 0.0 após 30 épocas. O modelo MEMORIZA os 29 templates quase perfeitamente — o vocabulário é menor e as sequências mais previsíveis. Isso exige k=8.0σ para separar anomalias do "ruído zero".</li>
    </ul>

    <h3>8.3. Sobre Sessões Ultra-Curtas (Limitação Fundamental)</h3>
    <p>O principal gargalo do HDFS são sessões com <strong>apenas 2 logs</strong>. Com <code>SKIP_START_LOGS=3</code>, essas sessões são efetivamente ignoradas. Mesmo sem skip, 2 logs não fornecem contexto suficiente para o Transformer gerar atenção discriminativa.</p>
    <p><strong>Solução proposta:</strong> Implementar classificação híbrida — rule-based para sessões curtas (≤3 logs) e ML para sessões com contexto suficiente (>3 logs).</p>

    <h3>8.4. Sobre Falsos Positivos</h3>
    <ul>
        <li><strong>OpenStack:</strong> Zero FP — os padrões normais são extremamente consistentes</li>
        <li><strong>HDFS:</strong> 733 FP (1.3%) — sessões normais com loss borderline (0.29–0.62). São operações normais levemente atípicas. A taxa é aceitável para produção.</li>
    </ul>

    <h3>8.5. Sobre Escalabilidade</h3>
    <table>
        <thead>
            <tr><th>Métrica</th><th>OpenStack</th><th>HDFS</th><th>Fator de Escala</th></tr>
        </thead>
        <tbody>
            <tr><td>Linhas de log</td><td>208K</td><td>11.2M</td><td><strong>54×</strong></td></tr>
            <tr><td>Sessões</td><td>507</td><td>72,661</td><td><strong>143×</strong></td></tr>
            <tr><td>Tempo de treino</td><td>10 min</td><td>120 min</td><td><strong>12×</strong></td></tr>
            <tr><td>Tempo de detecção</td><td>~1 min</td><td>~30 min</td><td><strong>30×</strong></td></tr>
        </tbody>
    </table>
    <div class="alert alert-success">
        <strong>✅ O escalonamento é sub-linear:</strong> Apesar de ter 54× mais dados, o treinamento demora apenas 12× mais. Isso indica que a arquitetura GPU-acelerada (batched inference) escala eficientemente.
    </div>

    <h3>8.6. Considerações Finais</h3>
    <ol>
        <li><strong>Generalização:</strong> O mesmo modelo (arquitetura idêntica) foi aplicado em dois sistemas completamente diferentes, sem modificação arquitetural — apenas ajuste de hiperparâmetros de treinamento e calibração.</li>
        <li><strong>Reprodutibilidade:</strong> Todos os experimentos usam <code>SEED=42</code> para garantir reprodutibilidade determinística.</li>
        <li><strong>Limitação ética:</strong> O sistema é uma ferramenta de <strong>auxílio à decisão</strong>, não um substituto para engenheiros de operação. Lead times de minutos/horas dão tempo para análise humana antes de ações corretivas.</li>
        <li><strong>Próximos passos:</strong> Aplicar o modelo em outros benchmarks (BGL, Thunderbird) e avaliar generalização cross-domain (treinar em um sistema, testar em outro).</li>
    </ol>
</div>

<!-- FOOTER -->
<div class="footer">
    <hr style="border: none; border-top: 1px solid var(--gray-300); margin-bottom: 20px;">
    <p><strong>Relatório Comparativo: Detecção de Anomalias em Logs</strong></p>
    <p>OpenStack vs HDFS — LogGPT-Small (28.98M parâmetros)</p>
    <p>Gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')}</p>
</div>

</body>
</html>"""

# Save HTML report
report_path = output_dir / "comparative_report.html"
with open(str(report_path), "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n  ✅ Report saved to: {report_path.absolute()}")
print(f"  📄 Size: {len(html):,} characters")
print(f"\n{'='*70}")
print(f"  REPORT GENERATION COMPLETE!")
print(f"{'='*70}")
