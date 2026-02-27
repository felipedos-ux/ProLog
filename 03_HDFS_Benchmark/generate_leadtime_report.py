# -*- coding: utf-8 -*-
"""
Relatório de Antecipação por Categoria de Falha — HDFS LogGPT
=============================================================
Gera um HTML report com lead times REAIS (minutos/horas) extraídos
do JSON de detecção + CSV com timestamps de microssegundos.

Seções:
  1. Resumo geral (KPIs com tempo real)
  2. Média de antecipação por categoria de falha
  3. Top 10 sessões com MAIOR antecipação
  4. Top 10 sessões com MENOR antecipação (reativa / zero)
  5. Análise visual (distribuição + faixas)
  6. Por tipo de template de erro
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
WORKSPACE = Path("d:/ProLog/03_HDFS_Benchmark")
DATA_DIR = Path("d:/ProLog/data")
CSV_PATH = DATA_DIR / "HDFS" / "HDFS_data_processed.csv"
RESULTS_JSON = WORKSPACE / "HDFS_test_results.json"
DOCS_DIR = WORKSPACE / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = DOCS_DIR / "relatorio_leadtime_por_falha_hdfs.html"

sns.set_theme(style="whitegrid", palette="muted")


# ============================================================
# HELPERS
# ============================================================
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def simplify_template(raw):
    """Cria um rótulo curto a partir do EventTemplate cru do HDFS."""
    if raw is None or pd.isna(raw):
        return "Desconhecido"
    s = str(raw).strip()
    # HDFS templates são mais curtos, pegar até 100 chars
    if len(s) > 100:
        s = s[:100] + "..."
    return s


def fmt_time(minutes):
    """Formata lead time em texto legível."""
    if minutes is None or pd.isna(minutes):
        return "—"
    if abs(minutes) < 1:
        return f"{minutes * 60:.1f} seg"
    elif abs(minutes) < 60:
        return f"{minutes:.1f} min"
    elif abs(minutes) < 1440:
        h = minutes / 60
        return f"{h:.1f} h"
    else:
        d = minutes / 1440
        return f"{d:.1f} dias"


# ============================================================
# CARREGAMENTO
# ============================================================
def load_data():
    print(f"📦 Loading CSV from {CSV_PATH}...")
    csv_df = pd.read_csv(CSV_PATH)
    csv_df['timestamp'] = pd.to_datetime(csv_df['timestamp'], errors='coerce')
    csv_df = csv_df.sort_values(['session_id', 'timestamp']).reset_index(drop=True)

    # Construir per-session: primeiro template de erro + timestamp do primeiro erro
    print("⏰ Building per-session error info...")
    session_error_info = {}
    for sid, grp in csv_df.groupby('session_id'):
        anom_rows = grp[grp['anom_label'] == 1]
        if len(anom_rows) > 0:
            first_error = anom_rows.iloc[0]
            session_error_info[sid] = {
                'first_error_template': first_error['EventTemplate'],
                'first_error_timestamp': first_error['timestamp'],
                'n_events': len(grp),
                'n_error_events': len(anom_rows),
                'start_time': grp['timestamp'].iloc[0],
                'end_time': grp['timestamp'].iloc[-1],
            }
        else:
            session_error_info[sid] = {
                'first_error_template': None,
                'first_error_timestamp': None,
                'n_events': len(grp),
                'n_error_events': 0,
                'start_time': grp['timestamp'].iloc[0],
                'end_time': grp['timestamp'].iloc[-1],
            }

    # Carregar JSON
    print(f"📦 Loading JSON from {RESULTS_JSON}...")
    with open(RESULTS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filtrar apenas TPs (label=1 e is_detected=true)
    tp_list = [r for r in data['results'] if r['label'] == 1 and r.get('is_detected', False)]
    tp = pd.DataFrame(tp_list)

    # Enriquecer com info do CSV
    tp['first_error_template'] = tp['session_id'].map(
        lambda sid: session_error_info.get(sid, {}).get('first_error_template'))
    tp['first_error_timestamp'] = tp['session_id'].map(
        lambda sid: session_error_info.get(sid, {}).get('first_error_timestamp'))
    tp['n_events'] = tp['session_id'].map(
        lambda sid: session_error_info.get(sid, {}).get('n_events', 0))
    tp['n_error_events'] = tp['session_id'].map(
        lambda sid: session_error_info.get(sid, {}).get('n_error_events', 0))
    tp['start_time'] = tp['session_id'].map(
        lambda sid: session_error_info.get(sid, {}).get('start_time'))
    tp['end_time'] = tp['session_id'].map(
        lambda sid: session_error_info.get(sid, {}).get('end_time'))

    # Flags
    tp['anticipated'] = tp['lead_time_minutes'].apply(
        lambda x: x is not None and not pd.isna(x) and x > 0)
    tp['failure_category'] = tp['first_error_template'].apply(simplify_template)

    # Também pegar info de todas as sessões anômalas (detectadas ou não) para contexto
    all_anom = [r for r in data['results'] if r['label'] == 1]
    all_anom_df = pd.DataFrame(all_anom)

    return data, tp, all_anom_df, session_error_info


# ============================================================
# GRÁFICOS
# ============================================================
def plot_anticipation_histogram(tp):
    """Histograma de lead time real (minutos)."""
    valid = tp[tp['lead_time_minutes'].notna()].copy()
    vals = valid['lead_time_minutes'].sort_values(ascending=False).values

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['#27ae60' if v > 0 else ('#95a5a6' if v == 0 else '#e74c3c') for v in vals]
    ax.bar(range(len(vals)), vals, color=colors, edgecolor='none', alpha=0.8, width=1.0)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title("Lead Time Real por Sessão HDFS (Minutos antes do 1º Erro)", fontsize=16)
    ax.set_xlabel(f"Sessões Detectadas ({len(vals)} TPs — ordenadas por lead time)", fontsize=12)
    ax.set_ylabel("Lead Time (minutos)", fontsize=13)

    mean_val = valid['lead_time_minutes'].mean()
    median_val = valid['lead_time_minutes'].median()
    ax.axhline(mean_val, color='#2980b9', linestyle='dashed', linewidth=2)
    ax.text(len(vals) * 0.6, mean_val + (ax.get_ylim()[1] * 0.03),
            f'Média: {fmt_time(mean_val)}', color='#2980b9', fontweight='bold', fontsize=12)
    ax.axhline(median_val, color='#8e44ad', linestyle='dotted', linewidth=2)
    ax.text(len(vals) * 0.6, median_val - (ax.get_ylim()[1] * 0.05),
            f'Mediana: {fmt_time(median_val)}', color='#8e44ad', fontweight='bold', fontsize=11)
    return fig_to_base64(fig)


def plot_category_boxplot(tp):
    """Box-plot de lead time real por categoria de falha."""
    valid = tp[tp['lead_time_minutes'].notna()].copy()
    top_cats = valid['failure_category'].value_counts().head(12).index.tolist()
    filtered = valid[valid['failure_category'].isin(top_cats)].copy()

    fig, ax = plt.subplots(figsize=(14, 8))
    order = filtered.groupby('failure_category')['lead_time_minutes'].median().sort_values(ascending=False).index
    sns.boxplot(data=filtered, y='failure_category', x='lead_time_minutes', orient='h',
                order=order, hue='failure_category', palette='RdYlGn', legend=False, ax=ax)
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax.set_title("Lead Time Real por Categoria de Falha HDFS", fontsize=16)
    ax.set_xlabel("Lead Time (minutos) — Positivo = ANTECIPOU", fontsize=13)
    ax.set_ylabel("")
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_time_bands(tp):
    """Pie chart das faixas de antecipação em TEMPO REAL."""
    valid = tp[tp['lead_time_minutes'].notna()].copy()

    def classify(val):
        if val > 480:      return "> 8 horas"
        elif val > 60:     return "1h — 8h"
        elif val > 10:     return "10 min — 1h"
        elif val > 1:      return "1 — 10 min"
        elif val > 0:      return "0 — 1 min"
        elif val == 0:     return "Simultâneo (0 min)"
        else:              return "Reativo (< 0)"

    valid['band'] = valid['lead_time_minutes'].apply(classify)
    order = ["> 8 horas", "1h — 8h", "10 min — 1h", "1 — 10 min", "0 — 1 min", "Simultâneo (0 min)", "Reativo (< 0)"]
    counts = valid['band'].value_counts()
    ordered_counts = [counts.get(b, 0) for b in order]
    colors_palette = ['#1abc9c', '#2ecc71', '#27ae60', '#f1c40f', '#e67e22', '#95a5a6', '#e74c3c']

    fig, ax = plt.subplots(figsize=(8, 8))
    non_zero = [(o, c, col) for o, c, col in zip(order, ordered_counts, colors_palette) if c > 0]
    if non_zero:
        labels, values, cols = zip(*non_zero)
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%',
            colors=cols, startangle=140, pctdistance=0.85,
            wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
        )
        plt.setp(autotexts, size=11, weight="bold")
    ax.set_title("Faixas de Antecipação (HDFS — Tempo Real)", fontsize=16, pad=20)
    return fig_to_base64(fig)


def plot_leadtime_distribution(tp):
    """Histogram/KDE do lead time em minutos."""
    valid = tp[tp['lead_time_minutes'].notna()].copy()
    # Clip para visualização (remover outliers extremos do gráfico)
    q_lo = valid['lead_time_minutes'].quantile(0.01)
    q_hi = valid['lead_time_minutes'].quantile(0.99)
    clipped = valid[(valid['lead_time_minutes'] >= q_lo) & (valid['lead_time_minutes'] <= q_hi)]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(data=clipped, x='lead_time_minutes', bins=50, kde=True, color='#3498db', ax=ax)
    ax.axvline(0, color='#e74c3c', linewidth=2, linestyle='--', label='Limiar (0 min)')
    median_val = valid['lead_time_minutes'].median()
    ax.axvline(median_val, color='#27ae60', linewidth=2, linestyle='--', label=f'Mediana: {fmt_time(median_val)}')
    ax.set_title("Distribuição do Lead Time Real (HDFS)", fontsize=16)
    ax.set_xlabel("Lead Time (minutos) — Positivo = Antecipação", fontsize=13)
    ax.set_ylabel("Quantidade de Sessões", fontsize=14)
    ax.legend(fontsize=12)
    return fig_to_base64(fig)


def plot_loss_vs_leadtime(tp):
    """Scatter: loss de alerta vs lead time."""
    valid = tp[(tp['lead_time_minutes'].notna()) & (tp['alert_loss'].notna())].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#27ae60' if a else '#e74c3c' for a in valid['anticipated']]
    ax.scatter(valid['alert_loss'], valid['lead_time_minutes'], c=colors, alpha=0.4, s=20, edgecolors='none')
    ax.set_xlabel("Alert Loss (Top-K)", fontsize=13)
    ax.set_ylabel("Lead Time (minutos)", fontsize=13)
    ax.set_title("Loss do Alerta vs Lead Time Real", fontsize=15)
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60', label='Antecipou', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', label='Simultâneo/Reativo', markersize=8),
    ]
    ax.legend(handles=legend_elements, fontsize=11)
    return fig_to_base64(fig)


# ============================================================
# HTML REPORT
# ============================================================
def generate_report():
    data, tp, all_anom_df, session_info = load_data()

    n_tp = len(tp)
    n_total_anom = len(all_anom_df)
    n_detected = int(all_anom_df['is_detected'].sum()) if 'is_detected' in all_anom_df.columns else n_tp
    n_not_detected = n_total_anom - n_detected

    valid_tp = tp[tp['lead_time_minutes'].notna()]
    n_anticipated = int(valid_tp['anticipated'].sum())
    n_simultaneous = int((valid_tp['lead_time_minutes'] == 0).sum())
    n_reactive = n_tp - n_anticipated - n_simultaneous
    pct_anticipated = n_anticipated / n_tp * 100 if n_tp > 0 else 0

    mean_lt = valid_tp['lead_time_minutes'].mean()
    median_lt = valid_tp['lead_time_minutes'].median()
    max_lt = valid_tp['lead_time_minutes'].max()
    min_lt = valid_tp['lead_time_minutes'].min()

    ant_only = valid_tp[valid_tp['anticipated']]
    mean_ant = ant_only['lead_time_minutes'].mean() if len(ant_only) > 0 else 0
    median_ant = ant_only['lead_time_minutes'].median() if len(ant_only) > 0 else 0

    # Pre-computed from JSON
    json_metrics = data.get('lead_time_metrics', {})

    # ── Gráficos ──
    print("📊 Gerando gráficos...")
    b64_hist = plot_anticipation_histogram(tp)
    b64_box = plot_category_boxplot(tp)
    b64_bands = plot_time_bands(tp)
    b64_dist = plot_leadtime_distribution(tp)
    b64_loss = plot_loss_vs_leadtime(tp)

    # ── Tabela: Média por Categoria ──
    cat_stats = valid_tp.groupby('failure_category').agg(
        count=('lead_time_minutes', 'size'),
        mean_min=('lead_time_minutes', 'mean'),
        median_min=('lead_time_minutes', 'median'),
        anticipated=('anticipated', 'sum'),
        max_min=('lead_time_minutes', 'max'),
        min_min=('lead_time_minutes', 'min'),
    ).reset_index()
    cat_stats['pct_ant'] = (cat_stats['anticipated'] / cat_stats['count'] * 100).round(1)
    cat_stats = cat_stats.sort_values('mean_min', ascending=False)

    cat_rows = ""
    for _, r in cat_stats.iterrows():
        color = '#27ae60' if r['mean_min'] > 0 else '#e74c3c'
        cat_rows += f"""<tr>
            <td style="max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{r['failure_category']}">{r['failure_category'][:90]}</td>
            <td>{int(r['count'])}</td>
            <td style="color:{color};font-weight:bold">{fmt_time(r['mean_min'])}</td>
            <td>{fmt_time(r['median_min'])}</td>
            <td>{fmt_time(r['max_min'])}</td>
            <td>{fmt_time(r['min_min'])}</td>
            <td>{r['pct_ant']:.0f}%</td>
        </tr>"""

    # ── Top 10 MAIOR ──
    top10_best = valid_tp.sort_values('lead_time_minutes', ascending=False).head(10)
    top10_worst = valid_tp.sort_values('lead_time_minutes', ascending=True).head(10)

    def make_top_rows(df_subset):
        rows = ""
        for _, r in df_subset.iterrows():
            cat = simplify_template(r.get('first_error_template', r.get('final_log', '')))[:60]
            lt = r['lead_time_minutes']
            color = '#27ae60' if lt > 0 else ('#95a5a6' if lt == 0 else '#e74c3c')
            sign = '+' if lt > 0 else ''
            loss = f"{r['alert_loss']:.3f}" if pd.notna(r.get('alert_loss')) else '—'
            n_ev = int(r['n_events']) if pd.notna(r.get('n_events')) else '—'
            rows += f"""<tr>
                <td><code style="font-size:10px;">{str(r['session_id'])[:30]}</code></td>
                <td style="color:{color};font-weight:bold">{sign}{fmt_time(lt)}</td>
                <td>{loss}</td>
                <td>{n_ev}</td>
                <td title="{cat}">{cat[:50]}{"..." if len(cat) > 50 else ""}</td>
            </tr>"""
        return rows

    best_rows = make_top_rows(top10_best)
    worst_rows = make_top_rows(top10_worst)

    # ── HTML ──
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LogGPT — Lead Time Real por Falha (HDFS)</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{ --primary:#1a1a2e; --accent:#16213e; --emerald:#27ae60; --ruby:#e74c3c; --gold:#f39c12; --bg:#0f0f23; --surface:#1e1e3f; --text:#e0e0e0; }}
        body {{ font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); margin:0; line-height:1.7; }}
        .hero {{ background:linear-gradient(135deg,#1a2a6c,#b21f1f,#fdbb2d); padding:60px 20px; text-align:center; border-bottom:4px solid var(--emerald); }}
        .hero h1 {{ font-size:2.8rem; margin:0; font-weight:800; color:#fff; text-shadow:0 2px 8px rgba(0,0,0,0.4); }}
        .hero p {{ font-size:1.1rem; color:rgba(255,255,255,0.85); margin-top:10px; }}
        .hero .subtitle {{ font-size:0.9rem; color:rgba(255,255,255,0.5); margin-top:5px; }}
        .container {{ max-width:1350px; margin:40px auto; padding:0 20px; }}
        section {{ background:var(--surface); border-radius:16px; padding:40px; margin-bottom:40px; box-shadow:0 8px 32px rgba(0,0,0,0.3); border:1px solid rgba(255,255,255,0.05); }}
        h2 {{ color:var(--gold); font-weight:800; border-bottom:2px solid rgba(243,156,18,0.3); padding-bottom:12px; margin-top:0; }}
        h3 {{ color:#3498db; font-weight:600; margin-top:30px; }}
        .kpi-row {{ display:flex; gap:20px; flex-wrap:wrap; margin:30px 0; }}
        .kpi {{ flex:1; min-width:150px; background:linear-gradient(145deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02)); padding:22px; border-radius:12px; text-align:center; border:1px solid rgba(255,255,255,0.08); }}
        .kpi .value {{ font-size:2rem; font-weight:800; margin:6px 0; }}
        .kpi .label {{ font-size:0.72rem; text-transform:uppercase; letter-spacing:1.5px; color:#7f8c8d; font-weight:600; }}
        .kpi.green .value {{ color:var(--emerald); }}
        .kpi.red .value {{ color:var(--ruby); }}
        .kpi.gold .value {{ color:var(--gold); }}
        .kpi.blue .value {{ color:#3498db; }}
        .kpi.purple .value {{ color:#9b59b6; }}
        table {{ width:100%; border-collapse:collapse; margin:20px 0; font-size:13px; }}
        th {{ background:rgba(243,156,18,0.15); color:var(--gold); padding:12px 10px; text-align:left; font-weight:600; border-bottom:2px solid rgba(243,156,18,0.3); }}
        td {{ padding:10px; border-bottom:1px solid rgba(255,255,255,0.05); }}
        tr:hover {{ background:rgba(255,255,255,0.03); }}
        .img-container {{ text-align:center; margin:30px 0; }}
        .img-container img {{ max-width:100%; border-radius:12px; border:1px solid rgba(255,255,255,0.1); }}
        .flex-grid {{ display:flex; gap:30px; flex-wrap:wrap; }}
        .col {{ flex:1; min-width:300px; }}
        .badge {{ display:inline-block; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:700; }}
        .badge-green {{ background:rgba(39,174,96,0.2); color:#27ae60; }}
        .badge-red {{ background:rgba(231,76,60,0.2); color:#e74c3c; }}
        .badge-gray {{ background:rgba(149,165,166,0.2); color:#95a5a6; }}
        code {{ background:rgba(255,255,255,0.1); padding:2px 6px; border-radius:4px; font-size:11px; }}
        .note {{ background:rgba(52,152,219,0.1); border-left:4px solid #3498db; padding:15px 20px; border-radius:0 8px 8px 0; margin:20px 0; font-size:14px; }}
        @media(max-width:800px) {{ .flex-grid {{ flex-direction:column; }} .kpi-row {{ flex-direction:column; }} }}
    </style>
</head>
<body>

<div class="hero">
    <h1>⏱ Lead Time Real por Falha — HDFS</h1>
    <p>LogGPT Threshold-Based · Hadoop Distributed File System</p>
    <p class="subtitle">Dataset: {n_total_anom:,} sessões anômalas · {n_detected:,} detectadas · Timestamps 2008-11</p>
</div>

<div class="container">

    <!-- SEC 1: RESUMO -->
    <section>
        <h2>1. Resumo Geral de Antecipação</h2>
        <p><strong>lead_time = timestamp_erro − timestamp_detecção</strong>. <span class="badge badge-green">Positivo</span> = modelo detectou ANTES do erro. <span class="badge badge-gray">Zero</span> = simultâneo. <span class="badge badge-red">Negativo</span> = reativo.</p>

        <div class="kpi-row">
            <div class="kpi green">
                <div class="label">Sessões Antecipadas</div>
                <div class="value">{n_anticipated:,}</div>
                <div class="label">({pct_anticipated:.1f}% dos {n_tp:,} TPs)</div>
            </div>
            <div class="kpi green">
                <div class="label">Média (Antecipados)</div>
                <div class="value">{fmt_time(mean_ant)}</div>
                <div class="label">antes do erro</div>
            </div>
            <div class="kpi blue">
                <div class="label">Mediana (Antecipados)</div>
                <div class="value">{fmt_time(median_ant)}</div>
                <div class="label">antes do erro</div>
            </div>
            <div class="kpi gold">
                <div class="label">Maior Antecipação</div>
                <div class="value">{fmt_time(max_lt)}</div>
                <div class="label">Lead time máximo</div>
            </div>
            <div class="kpi red">
                <div class="label">Não Detectadas</div>
                <div class="value">{n_not_detected:,}</div>
                <div class="label">Sessões perdidas</div>
            </div>
            <div class="kpi purple">
                <div class="label">Simultâneas (0 min)</div>
                <div class="value">{n_simultaneous:,}</div>
                <div class="label">Detecção no instante do erro</div>
            </div>
        </div>

        <div class="note">
            📐 <strong>Geral (todos TPs):</strong> Média = {fmt_time(mean_lt)} · Mediana = {fmt_time(median_lt)} · Max = {fmt_time(max_lt)} · Min = {fmt_time(min_lt)}
        </div>

        <div class="img-container">
            <img src="data:image/png;base64,{b64_hist}" alt="Lead Time por Sessão">
        </div>

        <div class="img-container">
            <img src="data:image/png;base64,{b64_dist}" alt="Distribuição Lead Time">
        </div>
    </section>

    <!-- SEC 2: POR CATEGORIA -->
    <section>
        <h2>2. Lead Time Médio por Categoria de Falha</h2>
        <p>Categorias baseadas no <code>EventTemplate</code> do primeiro log com <code>anom_label=1</code> de cada sessão.</p>

        <table>
            <tr>
                <th>Categoria de Falha</th>
                <th>Sessões</th>
                <th>Média (tempo)</th>
                <th>Mediana</th>
                <th>Máx</th>
                <th>Mín</th>
                <th>% Antecipadas</th>
            </tr>
            {cat_rows}
        </table>

        <div class="img-container">
            <img src="data:image/png;base64,{b64_box}" alt="Box-plot por Categoria">
        </div>
    </section>

    <!-- SEC 3: TOP 10 MELHOR -->
    <section>
        <h2>3. Top 10 — Maior Antecipação</h2>
        <p>Sessões com <strong>maior lead time real</strong> — o modelo alertou com mais antecedência.</p>
        <table>
            <tr>
                <th>Session ID</th>
                <th>Lead Time</th>
                <th>Alert Loss</th>
                <th>Eventos</th>
                <th>Tipo de Falha</th>
            </tr>
            {best_rows}
        </table>
    </section>

    <!-- SEC 4: TOP 10 PIOR -->
    <section>
        <h2>4. Top 10 — Menor Antecipação</h2>
        <p>Sessões com <strong>menor ou zero</strong> lead time — detecção simultânea ou muito próxima do erro.</p>
        <table>
            <tr>
                <th>Session ID</th>
                <th>Lead Time</th>
                <th>Alert Loss</th>
                <th>Eventos</th>
                <th>Tipo de Falha</th>
            </tr>
            {worst_rows}
        </table>
    </section>

    <!-- SEC 5: VISUAL -->
    <section>
        <h2>5. Análise Visual</h2>
        <div class="flex-grid">
            <div class="col img-container">
                <img src="data:image/png;base64,{b64_bands}" alt="Faixas">
                <p style="color:#7f8c8d;font-size:13px;"><b>Distribuição das faixas</b> de antecipação em tempo real.</p>
            </div>
            <div class="col img-container">
                <img src="data:image/png;base64,{b64_loss}" alt="Loss vs Lead Time">
                <p style="color:#7f8c8d;font-size:13px;"><b>Loss do alerta vs Lead Time:</b> Alta correlação indica que a intensidade do sinal anômalo está ligada à distância temporal do erro.</p>
            </div>
        </div>
    </section>

</div>
</body>
</html>"""

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ Relatório exportado: {REPORT_PATH}")
    print(f"\n📈 RESUMO TEMPORAL HDFS:")
    print(f"   Total Anômalas: {n_total_anom:,}")
    print(f"   Detectadas (TPs): {n_tp:,}")
    print(f"   Antecipadas: {n_anticipated:,} ({pct_anticipated:.1f}%)")
    print(f"   Simultâneas (0 min): {n_simultaneous:,}")
    print(f"   Média lead time (antecipados): {fmt_time(mean_ant)}")
    print(f"   Mediana (antecipados): {fmt_time(median_ant)}")
    print(f"   Maior antecipação: {fmt_time(max_lt)}")
    print(f"   Menor lead time: {fmt_time(min_lt)}")


if __name__ == "__main__":
    generate_report()
