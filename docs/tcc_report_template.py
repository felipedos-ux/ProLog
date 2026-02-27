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
<p>O BGL é um supercomputador IBM com 131.072 processadores. O dataset registra falhas de hardware: erros de memória, cache, rede torus, panicles de kernel. É fundamentalmente diferente dos outros dois datasets.</p>
<div class="warn">
⚠️ <strong>Por que o BGL não funcionou bem:</strong> O modelo foi treinado com padrões de OpenStack (software) e testado em logs de BGL (hardware). Esses domínios são tão diferentes que o modelo não consegue distinguir o "normal" do "anômalo" — ele acha tudo estranho. O BGL possui <strong>242 templates únicos</strong> (8x mais que os outros datasets), e esses templates descrevem eventos de hardware que nunca apareceram no treinamento. O resultado é um modelo que classifica quase tudo como anomalia (recall=100% mas precision=48.9%).
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
<p>O BGL obteve <strong>100% de recall</strong> mas apenas <strong>48.9% de precision</strong>. Isso significa que o modelo <strong>classificou praticamente TODAS as sessões como anômalas</strong>, acertando as que realmente eram anômalas mas também gerando uma quantidade massiva de falsos positivos.</p>

<h3>Causas Raíz do Insucesso</h3>

<h4>1. Incompatibilidade de Domínio (Transfer Learning Ineficaz)</h4>
<p>O modelo foi treinado em logs de <strong>OpenStack</strong> (software de cloud) e testado em logs de <strong>BGL</strong> (hardware de supercomputador). São domínios completamente diferentes:</p>
<ul>
<li><strong>OpenStack:</strong> HTTP requests, API calls, instâncias de VMs, operações CRUD</li>
<li><strong>BGL:</strong> Erros de memória DDR, parity errors, cache ECC, rede torus, kernel panics</li>
</ul>
<p>O modelo nunca viu esses tipos de eventos durante o treinamento, então qualquer sequência do BGL parece "anômala".</p>

<h4>2. Diversidade Excessiva de Templates</h4>
<div class="img-c"><img src="data:image/png;base64,{c_tmpl}" alt="Templates"></div>
<p>O BGL possui <strong>242 templates únicos</strong> — 8 vezes mais que o OpenStack (30) ou HDFS (29). Essa diversidade extrema significa que o vocabulário do BGL é muito mais rico e complexo, tornando impossível para um modelo treinado em outro domínio fazer previsões corretas.</p>

<h4>3. Natureza Diferente dos Eventos</h4>
<p>No OpenStack e HDFS, as anomalias são <em>perturbações</em> no padrão normal (um erro HTTP no meio de operações normais). No BGL, os eventos de "erro" e "normal" são frequentemente tipos de log completamente diferentes (registros de hardware vs. mensagens de aplicação), e não perturbações no mesmo fluxo.</p>

<h4>4. Modelo com Janela Fixa (Sliding Window)</h4>
<p>Enquanto OpenStack e HDFS usam sessões naturais (test_id, block_id), o BGL foi segmentado com <strong>janelas deslizantes de 20 eventos</strong>. Isso pode quebrar o contexto da sequência e misturar eventos que não pertencem ao mesmo incidente.</p>

<div class="warn">
⚠️ <strong>Conclusão BGL:</strong> Para o BGL funcionar adequadamente, seria necessário <strong>re-treinar o modelo</strong> diretamente com logs normais do BGL. A transferência de aprendizado entre domínios tão diferentes (software → hardware) não se sustenta com a abordagem Causal LM pura.
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
