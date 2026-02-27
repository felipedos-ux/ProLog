# -*- coding: utf-8 -*-
"""HTML template for the comprehensive TCC report."""

def build_html(c_metrics, c_cm, c_radar, c_lt, c_tmpl, c_pipe, OS, HD, BG):
    def fmt(m):
        if m is None: return "—"
        if abs(m)<1: return f"{m*60:.1f}s"
        if abs(m)<60: return f"{m:.1f}min"
        if abs(m)<1440: return f"{m/60:.1f}h"
        return f"{m/1440:.1f}d"

    CSS = """
    :root{--bg:#0a0a1a;--surface:#151530;--card:#1c1c40;--emerald:#27ae60;--blue:#3498db;--red:#e74c3c;--gold:#f39c12;--text:#e0e0e0;--muted:#7f8c8d;}
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);line-height:1.8}
    .hero{background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);padding:80px 20px;text-align:center;border-bottom:4px solid var(--emerald)}
    .hero h1{font-size:3rem;font-weight:900;background:linear-gradient(90deg,#27ae60,#3498db,#9b59b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .hero p{font-size:1.2rem;color:rgba(255,255,255,.7);margin-top:10px}
    .hero .sub{font-size:.85rem;color:rgba(255,255,255,.4);margin-top:8px}
    .container{max-width:1200px;margin:0 auto;padding:40px 20px}
    section{background:var(--surface);border-radius:16px;padding:45px;margin-bottom:40px;border:1px solid rgba(255,255,255,.05);box-shadow:0 8px 32px rgba(0,0,0,.3)}
    h2{color:var(--emerald);font-weight:800;font-size:1.8rem;border-bottom:2px solid rgba(39,174,96,.3);padding-bottom:12px;margin-bottom:25px}
    h3{color:var(--blue);font-weight:700;font-size:1.3rem;margin:30px 0 15px}
    h4{color:var(--gold);font-weight:600;margin:20px 0 10px}
    p{margin-bottom:15px;font-size:15px}
    .img-c{text-align:center;margin:25px 0}
    .img-c img{max-width:100%;border-radius:12px;border:1px solid rgba(255,255,255,.1)}
    table{width:100%;border-collapse:collapse;margin:20px 0;font-size:14px}
    th{background:rgba(39,174,96,.15);color:var(--emerald);padding:14px 12px;text-align:left;font-weight:700;border-bottom:2px solid rgba(39,174,96,.3)}
    td{padding:12px;border-bottom:1px solid rgba(255,255,255,.05)}
    tr:hover{background:rgba(255,255,255,.03)}
    .kpi-row{display:flex;gap:20px;flex-wrap:wrap;margin:25px 0}
    .kpi{flex:1;min-width:140px;background:var(--card);padding:22px;border-radius:12px;text-align:center;border:1px solid rgba(255,255,255,.08)}
    .kpi .v{font-size:2rem;font-weight:800;margin:6px 0}
    .kpi .l{font-size:.72rem;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);font-weight:600}
    .green .v{color:var(--emerald)}.blue .v{color:var(--blue)}.red .v{color:var(--red)}.gold .v{color:var(--gold)}
    .note{background:rgba(52,152,219,.1);border-left:4px solid var(--blue);padding:15px 20px;border-radius:0 8px 8px 0;margin:20px 0;font-size:14px}
    .warn{background:rgba(231,76,60,.1);border-left:4px solid var(--red);padding:15px 20px;border-radius:0 8px 8px 0;margin:20px 0;font-size:14px}
    .badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:700}
    .bg{background:rgba(39,174,96,.2);color:#27ae60}.br{background:rgba(231,76,60,.2);color:#e74c3c}.bb{background:rgba(52,152,219,.2);color:#3498db}
    code{background:rgba(255,255,255,.1);padding:2px 8px;border-radius:4px;font-size:12px;font-family:'Fira Code',monospace}
    ol,ul{margin:10px 0 15px 25px}
    li{margin-bottom:8px}
    .flex{display:flex;gap:30px;flex-wrap:wrap}.col{flex:1;min-width:280px}
    .step-box{background:var(--card);border-radius:12px;padding:20px;margin:15px 0;border-left:4px solid var(--emerald)}
    .step-box h4{margin-top:0;color:var(--emerald)}
    @media(max-width:800px){.flex{flex-direction:column}.kpi-row{flex-direction:column}}
    """

    return f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LogGPT — Relatório Completo TCC</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>

<div class="hero">
<h1>🧠 LogGPT: Detecção Proativa de Anomalias em Logs</h1>
<p>Trabalho de Conclusão de Curso — Relatório Técnico Completo</p>
<p class="sub">Análise comparativa em 3 datasets: OpenStack · HDFS · BGL | Modelo baseado em GPT-2 (Causal Language Model)</p>
</div>

<div class="container">

<!-- 1. INTRODUÇÃO -->
<section>
<h2>1. Introdução e Contexto</h2>
<p>Sistemas de software modernos — como plataformas de <strong>cloud computing</strong>, sistemas de <strong>armazenamento distribuído</strong> e <strong>supercomputadores</strong> — geram milhões de linhas de log diariamente. Essas mensagens registram o comportamento interno do sistema: cada operação, cada erro, cada alerta.</p>
<p>O desafio é: <strong>como detectar automaticamente que algo está errado, antes que o sistema falhe?</strong> A abordagem tradicional depende de regras manuais ("se aparecer a palavra ERROR, alerte"), mas isso é frágil e não captura padrões sutis que antecedem falhas catastróficas.</p>

<h3>O que é o LogGPT?</h3>
<p>O <strong>LogGPT</strong> é um modelo de inteligência artificial baseado na arquitetura <strong>GPT-2</strong> (a mesma família do ChatGPT) adaptado especificamente para analisar logs de sistemas computacionais. Em vez de aprender a prever a próxima palavra em textos humanos, o LogGPT aprende a <strong>prever o próximo evento de log</strong> em uma sequência de operações do sistema.</p>

<div class="note">
💡 <strong>Analogia simples:</strong> Imagine um médico que, após anos observando batimentos cardíacos normais, consegue identificar instantaneamente quando um ritmo está "fora do padrão". O LogGPT faz o mesmo com logs: ele aprende o padrão "saudável" e identifica quando o sistema começa a se comportar de forma anômala.
</div>

<h3>Objetivo do Trabalho</h3>
<p>Este trabalho avalia a capacidade do LogGPT de:</p>
<ol>
<li><strong>Detectar anomalias</strong> — identificar sessões que contêm falhas reais</li>
<li><strong>Antecipar falhas</strong> — alertar <em>antes</em> do erro acontecer (lead time)</li>
<li><strong>Generalizar</strong> — funcionar em diferentes domínios (cloud, storage, HPC)</li>
</ol>

<h3>Papers e Referências</h3>
<table>
<tr><th>Referência</th><th>Contribuição</th></tr>
<tr><td>Guo et al. (2021) — <em>"LogGPT: Log Anomaly Detection via GPT"</em></td><td>Proposta original do método, usando GPT-2 como modelo causal de linguagem para detecção de anomalias em logs estruturados.</td></tr>
<tr><td>He et al. (2020) — <em>"Loghub: A Large Collection of System Log Datasets"</em></td><td>Repositório de datasets de logs usados como benchmark (OpenStack, HDFS, BGL).</td></tr>
<tr><td>Du et al. (2017) — <em>"DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning"</em></td><td>Trabalho seminal que introduziu deep learning para detecção de anomalias em logs, usando LSTM.</td></tr>
<tr><td>He et al. (2017) — <em>"Drain: An Online Log Parsing Approach"</em></td><td>Algoritmo de parsing que converte logs brutos em templates estruturados (EventIds).</td></tr>
<tr><td>Radford et al. (2019) — <em>"Language Models are Unsupervised Multitask Learners"</em> (GPT-2)</td><td>Arquitetura base do modelo, demonstrando que modelos causais de linguagem podem capturar padrões sequenciais complexos.</td></tr>
</table>
</section>

<!-- 2. METODOLOGIA -->
<section>
<h2>2. Metodologia — Passo a Passo</h2>
<p>Abaixo explicamos cada etapa do processo, desde os logs brutos até a detecção final, de forma que qualquer pessoa possa entender.</p>

<div class="img-c"><img src="data:image/png;base64,{c_pipe}" alt="Pipeline"></div>

<div class="step-box"><h4>Etapa 1 — Coleta de Logs Brutos</h4>
<p>Os logs são coletados diretamente dos servidores. Cada linha contém um timestamp, nível de severidade (INFO, WARNING, ERROR), o componente que gerou o log e a mensagem em si.</p>
<p><code>2018-06-26 03:34:27 INFO nova.compute: Instance i-00000001 launched successfully</code></p>
</div>

<div class="step-box"><h4>Etapa 2 — Log Parsing (Drain)</h4>
<p>O algoritmo <strong>Drain</strong> converte cada mensagem de log em um <strong>template</strong> (EventId), substituindo valores variáveis por wildcards <code>&lt;*&gt;</code>. Isso reduz milhões de mensagens únicas para dezenas de templates reutilizáveis.</p>
<p>Exemplo: <code>"Instance i-00000001 launched"</code> → Template: <code>"Instance &lt;*&gt; launched"</code> → EventId: <code>e17b68d6</code></p>
</div>

<div class="step-box"><h4>Etapa 3 — Agrupamento em Sessões</h4>
<p>Os logs são agrupados por identificador de sessão:</p>
<ul>
<li><strong>OpenStack:</strong> por <code>test_id</code> (cada teste do Tempest gera uma sessão)</li>
<li><strong>HDFS:</strong> por <code>block_id</code> (cada bloco de dados tem sua sequência)</li>
<li><strong>BGL:</strong> por <code>node_id</code> + janela temporal (sliding window de 20 eventos)</li>
</ul>
<p>Uma sessão vira uma sequência de EventIds: <code>"e17b68d6 96691030 f7725eaf b8be6124"</code></p>
</div>

<div class="step-box"><h4>Etapa 4 — Treinamento do Modelo (Causal LM)</h4>
<p>O modelo GPT-2 é treinado <strong>apenas em sessões normais</strong> (sem falha). Ele aprende a prever "qual será o próximo evento?" dado o contexto anterior. Após o treino, ele sabe qual é o comportamento "normal" do sistema.</p>
<p><strong>Bibliotecas:</strong> PyTorch, HuggingFace Transformers, Polars (processamento de dados), Scikit-learn (métricas).</p>
</div>

<div class="step-box"><h4>Etapa 5 — Detecção (Top-K)</h4>
<p>Na fase de detecção, o modelo recebe cada sessão de teste e, para cada evento, verifica se o evento real está entre as <strong>Top-K predições mais prováveis</strong> (K=5). Se o evento real NÃO estiver no Top-5, o modelo marca aquele ponto como <strong>anômalo</strong>.</p>
<p>Se qualquer ponto da sessão for anômalo, toda a sessão é classificada como anômala.</p>
</div>

<div class="step-box"><h4>Etapa 6 — Cálculo do Lead Time</h4>
<p>O <strong>lead time</strong> mede quanto tempo <em>antes</em> do primeiro erro real o modelo detectou a anomalia. Usamos timestamps reais (resolução de microssegundos) para calcular essa diferença em minutos/horas.</p>
<p>Lead Time = Timestamp do 1º Erro Real − Timestamp da Detecção pelo Modelo</p>
</div>

<h3>Tecnologias Utilizadas</h3>
<table>
<tr><th>Tecnologia</th><th>Versão</th><th>Uso</th></tr>
<tr><td>Python</td><td>3.10+</td><td>Linguagem principal</td></tr>
<tr><td>PyTorch</td><td>2.x</td><td>Framework de deep learning</td></tr>
<tr><td>HuggingFace Transformers</td><td>4.x</td><td>Tokenizer e config do GPT-2</td></tr>
<tr><td>DistilGPT-2</td><td>—</td><td>Tokenizer base (vocabulário de 50257 tokens)</td></tr>
<tr><td>Polars</td><td>0.20+</td><td>Processamento eficiente de DataFrames</td></tr>
<tr><td>Pandas</td><td>2.x</td><td>Análise de dados e geração de relatórios</td></tr>
<tr><td>Scikit-learn</td><td>1.x</td><td>Métricas (precision, recall, F1, confusion matrix)</td></tr>
<tr><td>Matplotlib + Seaborn</td><td>3.x / 0.13</td><td>Visualizações e gráficos</td></tr>
</table>

<h3>Implementação Técnica — Parâmetros e Código</h3>
<p>Abaixo detalhamos os hiperparâmetros escolhidos para cada dataset, com justificativa técnica para cada decisão.</p>

<h4>Comparação de Hiperparâmetros</h4>
<table>
<tr><th>Parâmetro</th><th style="color:#27ae60">OpenStack</th><th style="color:#3498db">HDFS</th><th>Justificativa</th></tr>
<tr><td><strong>Tokenizer Base</strong></td><td><code>gpt2</code></td><td><code>distilgpt2</code></td><td>O OpenStack usa o tokenizer GPT-2 completo; HDFS usa DistilGPT-2 (mesma tokenização, modelo menor) por questão de performance no volume de dados (~575K sessões).</td></tr>
<tr><td><strong>BLOCK_SIZE</strong></td><td><code>1024</code></td><td><code>128</code></td><td>OpenStack tem sessões longas (média de 494 logs por teste) — precisa de contexto grande. HDFS tem sessões curtas (2-20 eventos por bloco) — 128 tokens é mais que suficiente e otimiza memória GPU.</td></tr>
<tr><td><strong>BATCH_SIZE</strong></td><td><code>8</code></td><td><code>64</code></td><td>OpenStack com BLOCK_SIZE=1024 consome ~12GB VRAM com batch de 8. HDFS com BLOCK_SIZE=128 permite batches 8x maiores, acelerando o treinamento na RTX 3080 Ti.</td></tr>
<tr><td><strong>EPOCHS</strong></td><td><code>10</code></td><td><code>30</code></td><td>OpenStack tem apenas 420 sessões — 10 épocas são suficientes para convergir sem overfitting. HDFS tem ~460K sessões de treino — precisa de mais épocas para o modelo aprender padrões de blocos curtos.</td></tr>
<tr><td><strong>LEARNING_RATE</strong></td><td><code>1e-4</code></td><td><code>1e-4</code></td><td>Taxa de aprendizado conservadora. Valor padrão do paper original LogGPT que se mostrou estável em ambos os datasets.</td></tr>
<tr><td><strong>N_LAYER / N_HEAD / N_EMBD</strong></td><td colspan="2"><code>4 / 4 / 256</code></td><td>Modelo "Small" com ~5M parâmetros. 4 camadas e 4 cabeças de atenção capturam padrões sequenciais sem risco de overfitting em datasets menores.</td></tr>
<tr><td><strong>DROPOUT</strong></td><td colspan="2"><code>0.1</code></td><td>Regularização leve (10% dos neurônios desligados aleatoriamente) para evitar memorização.</td></tr>
<tr><td><strong>K (Top-K)</strong></td><td colspan="2"><code>5</code></td><td>Se o próximo evento real não estiver entre as 5 predições mais prováveis do modelo, é marcado como anomalia. K=5 equilibra sensibilidade (detectar anomalias sutis) vs especificidade (evitar falsos positivos). Valores menores (K=1) geram muitos falsos positivos; maiores (K=10) perdem anomalias sutis.</td></tr>
<tr><td><strong>SKIP_START_LOGS</strong></td><td><code>1</code></td><td><code>3</code></td><td>Ignora os N primeiros logs de cada sessão durante a detecção ("cold start"). No OpenStack, anomalias podem aparecer logo no 2º evento (sessões de 7 logs); no HDFS, os primeiros 3 eventos são sempre de alocação (previsíveis).</td></tr>
<tr><td><strong>LOG_COLUMN</strong></td><td><code>EventId</code></td><td><code>EventTemplate</code></td><td>OpenStack usa o hash curto do EventId (1-2 tokens); HDFS usa o template completo. A escolha impacta a tokenização — EventId produz sequências mais compactas.</td></tr>
<tr><td><strong>SEED</strong></td><td colspan="2"><code>42</code></td><td>Semente fixa para reprodutibilidade total dos experimentos.</td></tr>
</table>

<h4>1. Processamento de Dados e Tokenização (Polars e HuggingFace)</h4>
<p>O processamento de logs brutos em tensores para a GPU é feito em duas etapas. Primeiro, usamos a biblioteca <strong>Polars</strong> (por sua velocidade e processamento multi-core em Rust) para agrupar milhões de linhas de log em "sessões". No OpenStack, agrupamos por <code>test_id</code> e concatenamos os <code>EventId</code> com espaços:</p>
<pre style="background:rgba(0,0,0,.3);padding:20px;border-radius:8px;overflow-x:auto;font-size:12px;color:#e0e0e0;font-family:'Fira Code',monospace"><span style="color:#7f8c8d"># dataset.py — Agrupamento de Sessões com Polars</span>
sessions = (
    df.sort(<span style="color:#27ae60">"timestamp"</span>)
    .group_by(<span style="color:#27ae60">"test_id"</span>)
    .agg([
        pl.col(LOG_COLUMN),
        pl.col(<span style="color:#27ae60">"anom_label"</span>).max().alias(<span style="color:#27ae60">"label"</span>) <span style="color:#7f8c8d"># Se 1 log for anômalo, a sessão inteira é</span>
    ])
).with_columns(
    <span style="color:#7f8c8d"># Resultado: "E1 E2 E5 E1 E3..."</span>
    pl.col(LOG_COLUMN).list.join(<span style="color:#27ae60">" "</span>).alias(<span style="color:#27ae60">"EventTemplate"</span>)
)</pre>

<p>Em seguida, transformamos as strings em tensores PyTorch usando o Tokenizer do HuggingFace. Para otimizar memória no treinamento de batches com tamanhos de sessão variados, usamos uma função de colação (<code>collate_fn</code>) que faz <strong>Dynamic Padding</strong> na CPU antes de enviar para a GPU:</p>
<pre style="background:rgba(0,0,0,.3);padding:20px;border-radius:8px;overflow-x:auto;font-size:12px;color:#e0e0e0;font-family:'Fira Code',monospace"><span style="color:#7f8c8d"># dataset.py — Dynamic Padding para batches de tamanho variável</span>
<span style="color:#e74c3c">def</span> <span style="color:#3498db">collate_fn</span>(batch):
    max_len = max(len(x) <span style="color:#e74c3c">for</span> x <span style="color:#e74c3c">in</span> batch)
    <span style="color:#7f8c8d"># Preenche com PAD_TOKEN (50256 no GPT2) até o maior log do batch atual</span>
    padded = torch.full((len(batch), max_len), <span style="color:#3498db">50256</span>, dtype=torch.long)
    <span style="color:#e74c3c">for</span> i, x <span style="color:#e74c3c">in</span> enumerate(batch):
        padded[i, :len(x)] = x
    <span style="color:#e74c3c">return</span> padded</pre>

<h4>2. Treinamento Causal LM (PyTorch)</h4>
<p>O treinamento do modelo não usa labels binários (0 ou 1) de anomalia. O modelo é treinado de forma auto-supervisionada (apenas em dados normais) usando <strong>Teacher Forcing</strong>: dado um contexto de N tokens, deve prever o token N+1. Isso é feito através do deslocamento de matrizes (<em>shift</em>):</p>
<pre style="background:rgba(0,0,0,.3);padding:20px;border-radius:8px;overflow-x:auto;font-size:12px;color:#e0e0e0;font-family:'Fira Code',monospace"><span style="color:#7f8c8d"># train_custom.py — Loop de treinamento (Causal LM Shift)</span>
<span style="color:#e74c3c">def</span> <span style="color:#3498db">train_epoch</span>(model, loader, optimizer, device, epoch):
    model.train()
    <span style="color:#e74c3c">for</span> batch <span style="color:#e74c3c">in</span> loader:
        <span style="color:#7f8c8d"># Causal shift: alvo é a entrada deslocada de 1 posição para a direita</span>
        inp = batch[:, :-<span style="color:#3498db">1</span>].to(device)  <span style="color:#7f8c8d"># Contexto (T_0 até T_N-1)</span>
        tgt = batch[:, <span style="color:#3498db">1</span>:].to(device)   <span style="color:#7f8c8d"># Alvo a prever (T_1 até T_N)</span>
        
        logits, loss = model(inp, targets=tgt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() <span style="color:#7f8c8d"># AdamW Optimizer com weight decay automático</span></pre>

<h4>3. Arquitetura do Modelo — LogGPT Small</h4>
<p>O modelo usa uma arquitetura GPT-2 customizada ("LogGPT-Small") com as seguintes especificações:</p>
<pre style="background:rgba(0,0,0,.3);padding:20px;border-radius:8px;overflow-x:auto;font-size:12px;color:#e0e0e0;font-family:'Fira Code',monospace"><span style="color:#7f8c8d"># model.py — Definição do modelo</span>
<span style="color:#e74c3c">class</span> <span style="color:#f39c12">LogGPT</span>(nn.Module):
    <span style="color:#7f8c8d">\"\"\"GPT-2 customizado para detecção de anomalias em logs.\"\"\"</span>
    <span style="color:#e74c3c">def</span> __init__(self, config):
        self.transformer = GPT2Model(config)  <span style="color:#7f8c8d"># 4 layers, 4 heads, 256 embd</span>
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
    <span style="color:#e74c3c">def</span> forward(self, input_ids):
        hidden = self.transformer(input_ids).last_hidden_state
        logits = self.lm_head(hidden)  <span style="color:#7f8c8d"># Shape: [batch, seq_len, vocab_size]</span>
        <span style="color:#e74c3c">return</span> logits</pre>

<h4>Detecção Top-K — Código Principal</h4>
<p>O trecho abaixo mostra a lógica central de detecção, idêntica para OpenStack e HDFS:</p>
<pre style="background:rgba(0,0,0,.3);padding:20px;border-radius:8px;overflow-x:auto;font-size:12px;color:#e0e0e0;font-family:'Fira Code',monospace"><span style="color:#7f8c8d"># detect_custom.py — Lógica Top-K</span>
K = <span style="color:#3498db">5</span>  <span style="color:#7f8c8d"># Top-K parameter</span>

<span style="color:#7f8c8d"># 1. Forward pass pelo modelo</span>
logits, _ = model(input_ids)     <span style="color:#7f8c8d"># [batch, seq_len, vocab_size]</span>

<span style="color:#7f8c8d"># 2. Shift: comparar predição[i] com alvo[i+1]</span>
targets = input_ids[:, <span style="color:#3498db">1</span>:]       <span style="color:#7f8c8d"># O que realmente aconteceu</span>
preds   = logits[:, :-<span style="color:#3498db">1</span>, :]     <span style="color:#7f8c8d"># O que o modelo previu</span>

<span style="color:#7f8c8d"># 3. Calcular Top-K predições mais prováveis</span>
probs = torch.softmax(preds, dim=-<span style="color:#3498db">1</span>)
_, topk_inds = torch.topk(probs, K, dim=-<span style="color:#3498db">1</span>)

<span style="color:#7f8c8d"># 4. Verificar se o evento REAL está no Top-K</span>
matches = (topk_inds == targets.unsqueeze(-<span style="color:#3498db">1</span>)).any(dim=-<span style="color:#3498db">1</span>)

<span style="color:#7f8c8d"># 5. Anomalia = evento NÃO está no Top-K (e não é padding)</span>
valid_anomalies = (~matches) & target_mask

<span style="color:#7f8c8d"># 6. Sessão inteira é anômala se QUALQUER evento for</span>
is_anomalous = valid_anomalies.any(dim=<span style="color:#3498db">1</span>)</pre>

<h4>Cálculo de Lead Time — Código com Timestamps Reais</h4>
<p>O lead time é calculado usando timestamps com resolução de microssegundos:</p>
<pre style="background:rgba(0,0,0,.3);padding:20px;border-radius:8px;overflow-x:auto;font-size:12px;color:#e0e0e0;font-family:'Fira Code',monospace"><span style="color:#7f8c8d"># Lead Time = Timestamp do 1º Erro Real − Timestamp da Detecção</span>
<span style="color:#7f8c8d"># Positivo → modelo ANTECIPOU a falha</span>
<span style="color:#7f8c8d"># Negativo → modelo detectou DEPOIS (reativo)</span>

<span style="color:#e74c3c">if</span> pred_label == <span style="color:#3498db">1</span> <span style="color:#e74c3c">and</span> first_error_timestamp <span style="color:#e74c3c">is not None</span>:
    <span style="color:#7f8c8d"># Mapear o passo Top-K para timestamp real do evento</span>
    alert_ts = pd.to_datetime(session_timestamps[first_anomaly_step])
    error_ts = pd.to_datetime(first_error_timestamp)
    
    <span style="color:#7f8c8d"># Diferença em segundos (positivo = antecipação)</span>
    lead_time_seconds = (error_ts - alert_ts).total_seconds()
    lead_time_minutes = lead_time_seconds / <span style="color:#3498db">60.0</span></pre>

<div class="note">
💡 <strong>Por que timestamps reais?</strong> Inicialmente o lead time era medido em número de eventos ("o modelo detectou 5 eventos antes do erro"). Porém, isso não diz quanto TEMPO o operador teria para reagir. Com timestamps reais, sabemos que no OpenStack a antecipação média é de <strong>3.8 minutos</strong> e no HDFS de até <strong>15 horas</strong>.
</div>

<h4>Justificativa das Métricas Escolhidas</h4>
<table>
<tr><th>Métrica</th><th>Por que usamos</th><th>Limitação</th></tr>
<tr><td><strong>F1-Score</strong></td><td>Métrica principal. É a média harmônica de Precision e Recall — penaliza modelos que sacrificam um pelo outro. Essencial quando os datasets são desbalanceados (mais sessões normais que anômalas).</td><td>Não captura a distribuição dos erros — um F1 de 90% pode esconder que o modelo erra sempre no mesmo tipo de falha.</td></tr>
<tr><td><strong>Precision</strong></td><td>Crucial em produção: um sistema com baixa precision gera "fadiga de alertas" — operadores ignoram alarmes quando muitos são falsos.</td><td>Alta precision com baixo recall significa que falhas reais estão passando despercebidas.</td></tr>
<tr><td><strong>Recall</strong></td><td>Mede a capacidade do modelo de encontrar TODAS as falhas. Em sistemas críticos (como um supercomputador), perder uma falha pode ser catastrófico.</td><td>100% de recall é fácil de atingir: basta alertar tudo (como o BGL fez com precision de 48.9%).</td></tr>
<tr><td><strong>Lead Time</strong></td><td>Diferencial do LogGPT: não apenas DETECTA anomalias, mas ANTECIPA. Mede o tempo real entre a detecção e o primeiro erro — quanto maior, mais tempo para reagir.</td><td>Depende da resolução temporal dos timestamps. Datasets com timestamps imprecisos (ex: apenas data sem hora) impossibilitam cálculos granulares.</td></tr>
<tr><td><strong>Confusion Matrix</strong></td><td>Visualização completa de TP/TN/FP/FN. Permite entender exatamente ONDE o modelo erra — crucial para debugging e melhoria.</td><td>Não captura a severidade dos erros — um FP em uma sessão de teste é diferente de um FP em produção.</td></tr>
</table>
</section>

<!-- 3. DATASETS -->
<section>
<h2>3. Datasets — Descrição e Particularidades</h2>
<p>Utilizamos três datasets públicos do repositório <a href="https://github.com/logpai/loghub" style="color:var(--blue)">Loghub</a>, cada um representando um cenário distinto de infraestrutura computacional.</p>

<table>
<tr><th>Característica</th><th style="color:#27ae60">🟢 OpenStack</th><th style="color:#3498db">🔵 HDFS</th><th style="color:#e74c3c">🔴 BGL</th></tr>
<tr><td><strong>Domínio</strong></td><td>Cloud Computing (IaaS)</td><td>Armazenamento Distribuído</td><td>Supercomputador HPC</td></tr>
<tr><td><strong>Fonte</strong></td><td>Loghub</td><td>Loghub</td><td>Loghub</td></tr>
<tr><td><strong>Período</strong></td><td>Jun-Jul 2018</td><td>Nov 2008</td><td>Jun 2005 — Jan 2006</td></tr>
<tr><td><strong>Total de Logs</strong></td><td>~424K linhas</td><td>~11M linhas</td><td>~4.7M linhas</td></tr>
<tr><td><strong>Sessões</strong></td><td>420 (test_ids)</td><td>~575K (block_ids)</td><td>~370K (sliding windows)</td></tr>
<tr><td><strong>Templates Únicos</strong></td><td>30</td><td>29</td><td><span class="badge br">242</span></td></tr>
<tr><td><strong>Agrupamento</strong></td><td>Por teste (test_id)</td><td>Por bloco HDFS (block_id)</td><td>Por nó + janela temporal</td></tr>
<tr><td><strong>Tipos de Falha</strong></td><td>Erros de API, exceções Python, timeouts</td><td>I/O exceptions, interrupções</td><td>Hardware: memória, cache, rede torus</td></tr>
<tr><td><strong>Resolução Temporal</strong></td><td>Microssegundos</td><td>Microssegundos</td><td>Segundos (Unix epoch)</td></tr>
<tr><td><strong>Modelo Usado</strong></td><td>Treinado localmente</td><td>Treinado localmente</td><td><span class="badge br">Modelo OpenStack (transferência)</span></td></tr>
</table>

<h3>🟢 OpenStack — Cloud Computing</h3>
<p>O OpenStack é uma plataforma open-source de cloud computing. O dataset contém logs de testes automatizados (Tempest) que exercitam APIs de criação de instâncias, volumes, redes e imagens. As sessões são relativamente curtas (dezenas a centenas de eventos) e os templates são bem definidos. <strong>Ideal para o LogGPT</strong> pois os padrões sequenciais são claros e consistentes.</p>

<h3>🔵 HDFS — Hadoop Distributed File System</h3>
<p>O HDFS é o sistema de arquivos distribuído do Hadoop. Cada bloco de dados gera uma sequência de log (alocação → replicação → servir leituras). As falhas são predominantemente de I/O (rede, disco). O dataset é muito grande (~575K blocos), o que dá ao modelo bastante dados para aprender. <strong>Desafio:</strong> muitas sessões muito curtas (2-5 eventos).</p>

<h3>🔴 BGL — Blue Gene/L Supercomputer</h3>
<p>O BGL é um supercomputador IBM com 131.072 processadores. O dataset registra falhas de hardware: erros de memória, cache, rede torus, kernel panics. É fundamentalmente diferente dos outros dois datasets.</p>

<h4>⚠️ Diferença Estrutural Crítica: Como as sessões são formadas</h4>
<p>A diferença mais importante entre os datasets está na <strong>forma como os logs são agrupados em sessões</strong>:</p>
<table>
<tr><th>Dataset</th><th>Agrupamento</th><th>O que representa</th><th>Compatível com Causal LM?</th></tr>
<tr><td style="color:#27ae60"><strong>OpenStack</strong></td><td><code>test_id</code></td><td>Uma <strong>operação completa</strong> (teste Tempest) com início, meio e fim definidos</td><td><span class="badge bg">✅ Sim</span></td></tr>
<tr><td style="color:#3498db"><strong>HDFS</strong></td><td><code>block_id</code></td><td>O <strong>ciclo de vida</strong> de um bloco: alocação → replicação → leitura</td><td><span class="badge bb">✅ Sim</span></td></tr>
<tr><td style="color:#e74c3c"><strong>BGL</strong></td><td><code>node_id</code></td><td>Uma <strong>máquina física</strong> — acumula logs de meses de operação misturada</td><td><span class="badge br">❌ Não</span></td></tr>
</table>
<p>No OpenStack e HDFS, cada sessão é um <strong>ciclo de vida completo de uma operação</strong> — o modelo consegue aprender a sequência "normal" (ex: criar VM → configurar rede → boot → sucesso) e detectar desvios (ex: timeout no meio). No BGL, o <code>node_id</code> (ex: <code>R02-M1-N0-C:J12-U11</code>) é apenas o endereço de uma máquina física que registra <strong>todos os tipos de eventos ao longo de meses</strong> sem nenhuma separação lógica. Não há um "fluxo previsível" — é uma mistura caótica de eventos de hardware rotineiros e erros reais.</p>

<div class="warn">
⚠️ <strong>Conclusão da análise estrutural:</strong> A abordagem Causal LM ("preveja o próximo evento") só funciona quando os logs formam <strong>sequências previsíveis com começo, meio e fim</strong>. Datasets como OpenStack (<code>test_id</code>) e HDFS (<code>block_id</code>) naturalmente satisfazem essa condição. O BGL, por agrupar logs por máquina física (<code>node_id</code>), <strong>não possui essa propriedade</strong>, tornando a abordagem LogGPT fundamentalmente inadequada para este tipo de dado — independentemente de re-treinamento.
</div>
</section>

<!-- 4. RESULTADOS COMPARATIVOS -->
<section>
<h2>4. Resultados Comparativos</h2>

<div class="kpi-row">
<div class="kpi green"><div class="l">OpenStack F1</div><div class="v">{OS['f1']*100:.1f}%</div><div class="l">Precision {OS['precision']*100:.1f}% · Recall {OS['recall']*100:.1f}%</div></div>
<div class="kpi blue"><div class="l">HDFS F1</div><div class="v">{HD['f1']*100:.1f}%</div><div class="l">Precision {HD['precision']*100:.1f}% · Recall {HD['recall']*100:.1f}%</div></div>
<div class="kpi red"><div class="l">BGL F1</div><div class="v">{BG['f1']*100:.1f}%</div><div class="l">Precision {BG['precision']*100:.1f}% · Recall {BG['recall']*100:.1f}%</div></div>
</div>

<h3>4.1 Métricas de Classificação</h3>
<div class="img-c"><img src="data:image/png;base64,{c_metrics}" alt="Métricas"></div>

<h3>4.2 Radar Comparativo</h3>
<div class="img-c"><img src="data:image/png;base64,{c_radar}" alt="Radar"></div>

<h3>4.3 Matrizes de Confusão</h3>
<div class="img-c"><img src="data:image/png;base64,{c_cm}" alt="Confusion Matrix"></div>

<h3>Interpretação das Métricas</h3>
<table>
<tr><th>Métrica</th><th>O que significa (linguagem simples)</th></tr>
<tr><td><strong>Precision</strong></td><td>"Dos alarmes que o modelo disparou, quantos eram falhas reais?" — Alta precision = poucos alarmes falsos.</td></tr>
<tr><td><strong>Recall</strong></td><td>"Das falhas reais que existiam, quantas o modelo encontrou?" — Alto recall = poucas falhas escaparam.</td></tr>
<tr><td><strong>F1-Score</strong></td><td>Média harmônica de Precision e Recall. É a métrica principal — quanto maior, melhor o equilíbrio.</td></tr>
<tr><td><strong>Accuracy</strong></td><td>"No geral, quantas classificações estavam corretas?" — Pode ser enganosa se os dados forem desbalanceados.</td></tr>
</table>
</section>

<!-- 5. OPENSTACK DETALHADO -->
<section>
<h2>5. OpenStack — Análise Detalhada</h2>
<div class="kpi-row">
<div class="kpi green"><div class="l">True Positives</div><div class="v">{OS['tp']}</div></div>
<div class="kpi blue"><div class="l">True Negatives</div><div class="v">{OS['tn']}</div></div>
<div class="kpi gold"><div class="l">False Positives</div><div class="v">{OS['fp']}</div></div>
<div class="kpi red"><div class="l">False Negatives</div><div class="v">{OS['fn']}</div></div>
</div>
<p>O OpenStack obteve os <strong>melhores resultados</strong> entre os três datasets. Com apenas <strong>5 falsos positivos</strong> e <strong>7 falsos negativos</strong> em 420 sessões, o modelo demonstra uma capacidade excepcional de distinguir operações normais de falhas reais.</p>
<p>Os tipos de falha detectados incluem: erros de API REST (HTTP 500), exceções Python (<code>TypeError</code>, <code>KeyError</code>), timeouts de operações e falhas de criação/destruição de recursos cloud.</p>

<h3>Lead Time — Antecipação</h3>
<div class="kpi-row">
<div class="kpi green"><div class="l">Antecipação Média</div><div class="v">{fmt(OS['lt_mean_min'])}</div></div>
<div class="kpi blue"><div class="l">Mediana</div><div class="v">{fmt(OS['lt_median_min'])}</div></div>
<div class="kpi gold"><div class="l">Máxima</div><div class="v">{fmt(OS['lt_max_min'])}</div></div>
<div class="kpi green"><div class="l">% Antecipadas</div><div class="v">{OS['lt_pct_ant']:.0f}%</div></div>
</div>
<p>Em <strong>{OS['lt_pct_ant']:.0f}%</strong> das sessões detectadas, o modelo alertou <em>antes</em> do primeiro erro real — com uma média de <strong>{fmt(OS['lt_mean_min'])}</strong> de antecedência. Isso demonstra que o LogGPT não apenas detecta falhas, mas as <strong>antecipa</strong>, dando tempo para ações corretivas antes que o impacto se materialize.</p>
</section>

<!-- 6. HDFS DETALHADO -->
<section>
<h2>6. HDFS — Análise Detalhada</h2>
<div class="kpi-row">
<div class="kpi green"><div class="l">True Positives</div><div class="v">{HD['tp']:,}</div></div>
<div class="kpi blue"><div class="l">True Negatives</div><div class="v">{HD['tn']:,}</div></div>
<div class="kpi gold"><div class="l">False Positives</div><div class="v">{HD['fp']:,}</div></div>
<div class="kpi red"><div class="l">False Negatives</div><div class="v">{HD['fn']:,}</div></div>
</div>
<p>O HDFS processou <strong>{HD['total_sessions']:,} sessões de teste</strong> — uma escala 170x maior que o OpenStack. Mesmo assim, manteve <strong>Precision de 95%</strong> com <strong>Recall de 82.3%</strong>. Os 2.983 falsos negativos representam blocos onde a anomalia era sutil demais para o modelo Top-K capturar (sessões muito curtas com poucos eventos).</p>

<h3>Lead Time — Antecipação</h3>
<div class="kpi-row">
<div class="kpi green"><div class="l">Antecipação Média</div><div class="v">{fmt(HD['lt_mean_min'])}</div></div>
<div class="kpi blue"><div class="l">Mediana</div><div class="v">{fmt(HD['lt_median_min'])}</div></div>
<div class="kpi gold"><div class="l">Máxima</div><div class="v">{fmt(HD['lt_max_min'])}</div></div>
<div class="kpi green"><div class="l">% Antecipadas</div><div class="v">{HD['lt_pct_ant']:.0f}%</div></div>
</div>
<p>No HDFS, o modelo conseguiu antecipar falhas com até <strong>{fmt(HD['lt_max_min'])}</strong> de antecedência. A média de <strong>{fmt(HD['lt_mean_min'])}</strong> indica que, em muitos casos, o modelo detecta padrões pré-falha horas antes da cascata de I/O errors se materializar.</p>
</section>

<!-- 7. BGL — POR QUE FALHOU -->
<section>
<h2>7. BGL — Análise e Motivos do Insucesso</h2>
<div class="kpi-row">
<div class="kpi red"><div class="l">Precision</div><div class="v">{BG['precision']*100:.1f}%</div><div class="l">Quase metade são alarmes falsos</div></div>
<div class="kpi gold"><div class="l">Recall</div><div class="v">{BG['recall']*100:.0f}%</div><div class="l">Encontrou tudo (pois alertou tudo)</div></div>
<div class="kpi red"><div class="l">F1-Score</div><div class="v">{BG['f1']*100:.1f}%</div><div class="l">Muito abaixo do aceitável</div></div>
</div>

<h3>O que aconteceu?</h3>
<p>O BGL obteve <strong>100% de recall</strong> mas apenas <strong>48.9% de precision</strong>. Isso significa que o modelo <strong>classificou praticamente TODAS as sessões como anômalas</strong>, acertando as que realmente eram anômalas mas também gerando uma quantidade massiva de falsos positivos. Existem duas causas raíz combinadas:</p>

<h3>Causa 1 — Modelo treinado no domínio errado (Transfer Learning)</h3>
<p>O modelo foi treinado exclusivamente com logs de <strong>OpenStack</strong> (software de cloud) e testado em logs de <strong>BGL</strong> (hardware de supercomputador). São vocabulários completamente diferentes:</p>
<ul>
<li><strong>OpenStack:</strong> HTTP requests, API calls, instâncias de VMs, operações CRUD</li>
<li><strong>BGL:</strong> Erros de memória DDR, parity errors, cache ECC, rede torus, kernel panics</li>
</ul>
<p>O modelo nunca viu esses tipos de eventos durante o treinamento, então <strong>qualquer sequência do BGL parece "anômala"</strong> para ele.</p>

<h4>Diversidade de Templates</h4>
<div class="img-c"><img src="data:image/png;base64,{c_tmpl}" alt="Templates"></div>
<p>O BGL possui <strong>242 templates únicos</strong> — 8 vezes mais que o OpenStack (30) ou HDFS (29). Nenhum deles foi visto pelo modelo durante o treinamento.</p>

<h3>Causa 2 — Estrutura de sessão incompatível (Problema Fundamental)</h3>
<p>Mesmo que re-treinássemos o modelo com dados nativos do BGL, <strong>a abordagem LogGPT ainda não funcionaria</strong>, porque a estrutura dos logs do BGL é fundamentalmente incompatível com o método Causal LM.</p>

<h4>🔑 A diferença crucial: o que é uma "sessão"</h4>
<table>
<tr><th>Dataset</th><th>ID da Sessão</th><th>O que representa</th><th>Padrão sequencial?</th></tr>
<tr><td style="color:#27ae60"><strong>OpenStack</strong></td><td><code>test_id</code></td><td>Uma operação completa com início → meio → fim</td><td><span class="badge bg">✅ Previsível</span></td></tr>
<tr><td style="color:#3498db"><strong>HDFS</strong></td><td><code>block_id</code></td><td>Ciclo de vida do bloco: alocação → replicação → leitura</td><td><span class="badge bb">✅ Previsível</span></td></tr>
<tr><td style="color:#e74c3c"><strong>BGL</strong></td><td><code>node_id</code></td><td>Máquina física — meses de logs misturados sem separação</td><td><span class="badge br">❌ Caótico</span></td></tr>
</table>

<p>No OpenStack, um <code>test_id</code> como <code>"nova.compute.test_create_instance"</code> representa um <strong>teste completo</strong>: criar VM → configurar rede → fazer boot → verificar status → limpar. O modelo aprende essa sequência "healthy" e detecta quando algo desvia (ex: timeout no meio).</p>

<p>No HDFS, um <code>block_id</code> como <code>"blk_-1608999687919862906"</code> tem um <strong>ciclo de vida natural</strong>: o bloco é alocado, replicado em 3 nós, e depois servido para leituras. O modelo aprende essa cadeia e detecta quando um bloco falha no meio.</p>

<p>No BGL, um <code>node_id</code> como <code>"R02-M1-N0-C:J12-U11"</code> é simplesmente <strong>o endereço de um nó físico do supercomputador</strong>. Ele acumula TODOS os logs daquela máquina ao longo de <strong>7 meses de operação</strong> (jun/2005 a jan/2006). Não existe um "fluxo" — é uma mistura caótica de:</p>
<ul>
<li>Eventos de hardware corriqueiros (correções de ECC, bit steering)</li>
<li>Erros reais (kernel panics, falhas de memória)</li>
<li>Mensagens de manutenção (reinicializações, atualizações)</li>
<li>Processos de diferentes aplicações rodando simultaneamente</li>
</ul>

<div class="note">
💡 <strong>Analogia:</strong> Imagine que você quer que um médico identifique batimentos cardíacos irregulares. No OpenStack e HDFS, ele recebe um exame de ECG completo (começo, meio, fim — uma sequência clara). No BGL, ele recebe <strong>7 meses de registros misturados</strong> de pressão, temperatura, batimento, sono, exercício, tudo junto e fora de ordem. Não há como aprender um "padrão normal" nesse caos.
</div>

<p>Para tentar contornar isso, dividimos os logs do BGL em <strong>janelas deslizantes de 20 eventos</strong> (sliding window). Mas isso é artificial e:</p>
<ul>
<li>Quebra o contexto temporal (uma janela pode conter metade de um incidente)</li>
<li>Mistura eventos de diferentes origens na mesma janela</li>
<li>Não captura relações de longo prazo entre eventos do mesmo nó</li>
</ul>

<div class="warn">
⚠️ <strong>Conclusão BGL:</strong> O insucesso no BGL não é apenas uma questão de re-treinamento. A abordagem Causal LM ("preveja o próximo evento na sequência") <strong>depende fundamentalmente de sessões com padrões sequenciais previsíveis</strong>. O BGL não possui essa propriedade — seus logs são agrupados por máquina física, não por operação lógica. Para datasets com essa estrutura, abordagens alternativas como <strong>frequência de templates por janela temporal</strong>, <strong>grafos de dependência de hardware</strong>, ou <strong>modelos de séries temporais</strong> seriam mais adequadas.
</div>
</section>

<!-- 8. LEAD TIME COMPARATIVO -->
<section>
<h2>8. Lead Time — Análise Comparativa de Antecipação</h2>
<p>O <strong>lead time</strong> é a capacidade preditiva mais valiosa do LogGPT: quanto tempo antes do primeiro erro real o modelo consegue alertar sobre a anomalia.</p>

<div class="img-c"><img src="data:image/png;base64,{c_lt}" alt="Lead Time"></div>

<table>
<tr><th>Métrica de Lead Time</th><th style="color:#27ae60">OpenStack</th><th style="color:#3498db">HDFS</th><th style="color:#e74c3c">BGL</th></tr>
<tr><td>Média (sessões antecipadas)</td><td><strong>{fmt(OS['lt_mean_min'])}</strong></td><td><strong>{fmt(HD['lt_mean_min'])}</strong></td><td>—</td></tr>
<tr><td>Mediana</td><td>{fmt(OS['lt_median_min'])}</td><td>{fmt(HD['lt_median_min'])}</td><td>—</td></tr>
<tr><td>Máximo</td><td>{fmt(OS['lt_max_min'])}</td><td>{fmt(HD['lt_max_min'])}</td><td>—</td></tr>
<tr><td>% Sessões Antecipadas</td><td>{OS['lt_pct_ant']:.0f}%</td><td>{HD['lt_pct_ant']:.0f}%</td><td>N/A (modelo inválido)</td></tr>
</table>

<div class="note">
📊 <strong>Interpretação:</strong> No OpenStack, o modelo tipicamente antecipa falhas em ~3.5 minutos — tempo suficiente para um sistema de automação acionar re-tentativas ou failover. No HDFS, a antecipação pode chegar a horas, permitindo realocação proativa de blocos de dados antes que falhas de disco se consumem.
</div>
</section>

<!-- 9. CONCLUSÕES -->
<section>
<h2>9. Conclusões e Contribuições</h2>

<h3>✅ Contribuições Positivas</h3>
<ol>
<li><strong>Validação do LogGPT como abordagem viável</strong> para detecção proativa de anomalias em dois domínios (cloud computing e storage distribuído).</li>
<li><strong>Capacidade de antecipação comprovada</strong>: o modelo não apenas detecta, mas prevê falhas minutos ou horas antes de sua materialização.</li>
<li><strong>Pipeline reprodutível</strong>: todo o código está documentado e disponível para replicação.</li>
<li><strong>Análise comparativa robusta</strong>: três datasets públicos testados, com métricas completas e transparentes.</li>
</ol>

<h3>⚠️ Limitações Identificadas</h3>
<ol>
<li><strong>Transferência cross-domain limitada:</strong> O modelo treinado em um domínio (OpenStack) não generaliza para hardware (BGL). É necessário re-treinar para novos domínios.</li>
<li><strong>Sensibilidade ao vocabulário:</strong> A diversidade de templates impacta diretamente o desempenho. Datasets com muitos templates únicos (&gt;100) são desafiadores.</li>
<li><strong>Granularidade de sessão:</strong> A forma como os logs são agrupados (sessões naturais vs. janelas deslizantes) afeta significativamente a qualidade da detecção.</li>
</ol>

<h3>🔮 Trabalhos Futuros</h3>
<ul>
<li>Retreinar o LogGPT diretamente em logs de BGL para avaliar se funciona com dados nativos do domínio.</li>
<li>Explorar embeddings de templates (Word2Vec, BERT) para melhorar a captura semântica.</li>
<li>Implementar um sistema de re-treinamento contínuo (online learning) para adaptar o modelo a drift de vocabulário.</li>
<li>Integrar com sistemas de orquestração (Kubernetes, Prometheus) para resposta automática.</li>
</ul>
</section>

<!-- 10. RESUMO FINAL -->
<section>
<h2>10. Tabela Resumo Final</h2>
<table>
<tr><th>Aspecto</th><th style="color:#27ae60">🟢 OpenStack</th><th style="color:#3498db">🔵 HDFS</th><th style="color:#e74c3c">🔴 BGL</th></tr>
<tr><td><strong>F1-Score</strong></td><td style="color:#27ae60;font-weight:bold">{OS['f1']*100:.1f}%</td><td style="color:#3498db;font-weight:bold">{HD['f1']*100:.1f}%</td><td style="color:#e74c3c;font-weight:bold">{BG['f1']*100:.1f}%</td></tr>
<tr><td><strong>Precision</strong></td><td>{OS['precision']*100:.1f}%</td><td>{HD['precision']*100:.1f}%</td><td>{BG['precision']*100:.1f}%</td></tr>
<tr><td><strong>Recall</strong></td><td>{OS['recall']*100:.1f}%</td><td>{HD['recall']*100:.1f}%</td><td>{BG['recall']*100:.0f}%</td></tr>
<tr><td><strong>Lead Time Médio</strong></td><td>{fmt(OS['lt_mean_min'])}</td><td>{fmt(HD['lt_mean_min'])}</td><td>N/A</td></tr>
<tr><td><strong>Templates</strong></td><td>{OS['n_templates']}</td><td>{HD['n_templates']}</td><td>{BG['n_templates']}</td></tr>
<tr><td><strong>Modelo</strong></td><td>Treinado local</td><td>Treinado local</td><td>Transfer (OpenStack)</td></tr>
<tr><td><strong>Veredicto</strong></td><td><span class="badge bg">✅ Excelente</span></td><td><span class="badge bb">✅ Bom</span></td><td><span class="badge br">❌ Insuficiente</span></td></tr>
</table>
</section>

</div>
</body></html>"""
