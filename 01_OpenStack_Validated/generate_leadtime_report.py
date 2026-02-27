# -*- coding: utf-8 -*-
"""
Relatório de Antecipação por Categoria de Falha — OpenStack LogGPT
===================================================================
Gera um HTML report com lead times REAIS (minutos/horas) calculados
a partir da coluna 'time_hour' do CSV original (resolução de microssegundos).

Seções:
  1. Resumo geral (KPIs com tempo real + eventos)
  2. Média de antecipação por categoria de falha
  3. Top 10 sessões com MAIOR antecipação
  4. Top 10 sessões com MENOR antecipação (reativa)
  5. Análise posicional (scatter + faixas)
  6. Por EventId de erro
"""

import json
import re
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
WORKSPACE = Path("d:/ProLog/01_OpenStack_Validated")
DATA_DIR = Path("d:/ProLog/data")
CSV_PATH = DATA_DIR / "OpenStack_data_original.csv"
RESULTS_JSON = WORKSPACE / "results_metrics_detailed.json"
DOCS_DIR = WORKSPACE / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = DOCS_DIR / "relatorio_leadtime_por_falha.html"

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
    """Cria um rótulo curto a partir do error_template cru."""
    if raw is None:
        return "Desconhecido"
    s = str(raw).strip().strip('"')
    if s.startswith("Failure"):
        return "Failure!!!"
    for separator in ["Traceback (most recent call last):", "None File", "None None"]:
        idx = s.find(separator)
        if idx > 0:
            s = s[:idx].strip()
            break
    first_line = s.split('\n')[0].strip()
    if len(first_line) > 120:
        colon = first_line.find(':')
        if 0 < colon < 100:
            first_line = first_line[:colon + min(60, len(first_line) - colon)]
        else:
            first_line = first_line[:120]
    return first_line.strip(' "') + ("..." if len(s) > 120 else "")


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
# CARREGAMENTO COM TIMESTAMPS REAIS
# ============================================================
def load_data():
    print(f"📦 Loading CSV from {CSV_PATH}...")
    csv_df = pd.read_csv(CSV_PATH, usecols=['test_id', 'time_hour', 'anom_label', 'EventId', 'EventTemplate'])
    csv_df['time_hour'] = pd.to_datetime(csv_df['time_hour'], errors='coerce')
    csv_df = csv_df.sort_values(['test_id', 'time_hour']).reset_index(drop=True)

    # Construir per-session timestamp list (ordenado por time_hour)
    print("⏰ Building per-session timestamp index...")
    session_timestamps = {}
    session_first_error_ts = {}
    for tid, grp in csv_df.groupby('test_id'):
        timestamps = grp['time_hour'].tolist()
        session_timestamps[tid] = timestamps
        # Primeiro timestamp com anom_label=1
        anom_rows = grp[grp['anom_label'] == 1]
        if len(anom_rows) > 0:
            session_first_error_ts[tid] = anom_rows['time_hour'].iloc[0]

    # Carregar resultados do detector
    print(f"📦 Loading JSON from {RESULTS_JSON}...")
    with open(RESULTS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tp_list = [r for r in data['results'] if r['label'] == 1 and r['predicted'] == 1]
    tp = pd.DataFrame(tp_list)

    # ── Calcular lead time temporal REAL ──
    lead_times_min = []
    alert_timestamps = []
    error_timestamps = []

    for _, row in tp.iterrows():
        tid = row['test_id']
        step = row['first_anomaly_step']
        err_idx = row.get('first_error_index')
        ts_list = session_timestamps.get(tid, [])
        err_ts = session_first_error_ts.get(tid)

        alert_ts = None
        lt_min = None

        if step is not None and step >= 0 and step < len(ts_list) and err_ts is not None:
            alert_ts = ts_list[int(step)]
            if pd.notna(alert_ts) and pd.notna(err_ts):
                lt_min = (err_ts - alert_ts).total_seconds() / 60.0

        lead_times_min.append(lt_min)
        alert_timestamps.append(alert_ts)
        error_timestamps.append(err_ts)

    tp['lead_time_minutes'] = lead_times_min
    tp['alert_timestamp_real'] = alert_timestamps
    tp['error_timestamp_real'] = error_timestamps

    # alert_step_before_error (em eventos)
    tp['alert_step_before_error'] = tp.apply(
        lambda r: (r['first_error_index'] - r['first_anomaly_step'])
        if r.get('first_error_index') is not None and r.get('first_anomaly_step') is not None and r['first_anomaly_step'] >= 0
        else None, axis=1
    )

    # Flags
    tp['anticipated'] = tp['lead_time_minutes'].apply(lambda x: x > 0 if x is not None and not pd.isna(x) else False)
    tp['failure_category'] = tp['error_template'].apply(simplify_template)
    tp['detection_pct'] = (tp['first_anomaly_step'] / tp['n_events'] * 100).clip(upper=100)
    tp['error_position_pct'] = tp.apply(
        lambda r: (r['first_error_index'] / r['n_events'] * 100) if r.get('first_error_index') is not None and r['n_events'] > 0 else None,
        axis=1
    )

    return data, tp


# ============================================================
# GRÁFICOS
# ============================================================
def plot_anticipation_histogram(tp):
    """Histograma de lead time real (minutos)."""
    valid = tp[tp['lead_time_minutes'].notna()].copy()
    vals = valid['lead_time_minutes'].sort_values(ascending=False).values

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#27ae60' if v > 0 else '#e74c3c' for v in vals]
    ax.bar(range(len(vals)), vals, color=colors, edgecolor='none', alpha=0.8)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title("Antecipação Real por Sessão (Tempo antes do 1º Erro)", fontsize=16)
    ax.set_xlabel("Sessões (ordenadas por antecipação)", fontsize=13)
    ax.set_ylabel("Lead Time (minutos)", fontsize=13)

    mean_val = valid['lead_time_minutes'].mean()
    ax.axhline(mean_val, color='#2980b9', linestyle='dashed', linewidth=2)
    ax.text(len(vals) * 0.02, mean_val + (ax.get_ylim()[1] * 0.02),
            f'Média: {fmt_time(mean_val)}', color='#2980b9', fontweight='bold', fontsize=12)
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
    ax.set_title("Lead Time Real por Categoria de Falha (Top 12)", fontsize=16)
    ax.set_xlabel("Lead Time (minutos) — Positivo = ANTECIPOU", fontsize=13)
    ax.set_ylabel("")
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_detection_vs_error_scatter(tp):
    """Scatter: posição da detecção vs posição do erro na sessão."""
    valid = tp[(tp['detection_pct'].notna()) & (tp['error_position_pct'].notna())].copy()

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['#27ae60' if a else '#e74c3c' for a in valid['anticipated']]
    ax.scatter(valid['error_position_pct'], valid['detection_pct'], c=colors, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel("Posição do 1º Erro Real (% da sessão)", fontsize=13)
    ax.set_ylabel("Posição da Detecção pelo Modelo (% da sessão)", fontsize=13)
    ax.set_title("Detecção vs Erro Real (abaixo da diagonal = antecipou)", fontsize=15)
    ax.set_xlim(0, 105); ax.set_ylim(0, 105)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60', label='Antecipou', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', label='Reativo', markersize=10),
        Line2D([0], [0], color='black', linestyle='--', label='Linha de igualdade')
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper left')
    return fig_to_base64(fig)


def plot_time_bands(tp):
    """Pie chart das faixas de antecipação em TEMPO REAL."""
    valid = tp[tp['lead_time_minutes'].notna()].copy()

    def classify(val):
        if val > 1440:     return "> 1 dia"
        elif val > 60:     return "1h — 1 dia"
        elif val > 10:     return "10 min — 1h"
        elif val > 1:      return "1 — 10 min"
        elif val > 0:      return "0 — 1 min"
        elif val == 0:     return "Simultâneo"
        else:              return "Reativo (< 0)"

    valid['band'] = valid['lead_time_minutes'].apply(classify)
    order = ["> 1 dia", "1h — 1 dia", "10 min — 1h", "1 — 10 min", "0 — 1 min", "Simultâneo", "Reativo (< 0)"]
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
    ax.set_title("Faixas de Antecipação (Tempo Real)", fontsize=16, pad=20)
    return fig_to_base64(fig)


def plot_leadtime_distribution(tp):
    """Histogram/KDE do lead time em minutos."""
    valid = tp[tp['lead_time_minutes'].notna()].copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(data=valid, x='lead_time_minutes', bins=30, kde=True, color='#3498db', ax=ax)
    ax.axvline(0, color='#e74c3c', linewidth=2, linestyle='--', label='Limiar (0 min)')
    median_val = valid['lead_time_minutes'].median()
    ax.axvline(median_val, color='#27ae60', linewidth=2, linestyle='--', label=f'Mediana: {fmt_time(median_val)}')
    ax.set_title("Distribuição do Lead Time Real", fontsize=16)
    ax.set_xlabel("Lead Time (minutos) — Positivo = Antecipação", fontsize=13)
    ax.set_ylabel("Quantidade de Sessões", fontsize=14)
    ax.legend(fontsize=12)
    return fig_to_base64(fig)


# ============================================================
# HTML REPORT
# ============================================================
def generate_report():
    data, tp = load_data()

    n_tp = len(tp)
    valid_tp = tp[tp['lead_time_minutes'].notna()]
    n_anticipated = int(valid_tp['anticipated'].sum())
    n_reactive = n_tp - n_anticipated
    pct_anticipated = n_anticipated / n_tp * 100 if n_tp > 0 else 0

    mean_lt = valid_tp['lead_time_minutes'].mean()
    median_lt = valid_tp['lead_time_minutes'].median()
    max_lt = valid_tp['lead_time_minutes'].max()
    min_lt = valid_tp['lead_time_minutes'].min()

    ant_only = valid_tp[valid_tp['anticipated']]
    mean_ant = ant_only['lead_time_minutes'].mean() if len(ant_only) > 0 else 0
    median_ant = ant_only['lead_time_minutes'].median() if len(ant_only) > 0 else 0

    # Step-based stats too
    valid_steps = tp[tp['alert_step_before_error'].notna()]
    mean_steps = valid_steps['alert_step_before_error'].mean()

    # ── Gráficos ──
    print("📊 Gerando gráficos...")
    b64_hist = plot_anticipation_histogram(tp)
    b64_box = plot_category_boxplot(tp)
    b64_scatter = plot_detection_vs_error_scatter(tp)
    b64_bands = plot_time_bands(tp)
    b64_dist = plot_leadtime_distribution(tp)

    # ── Tabela: Média por Categoria ──
    cat_stats = valid_tp.groupby('failure_category').agg(
        count=('lead_time_minutes', 'size'),
        mean_min=('lead_time_minutes', 'mean'),
        median_min=('lead_time_minutes', 'median'),
        anticipated=('anticipated', 'sum'),
        max_min=('lead_time_minutes', 'max'),
        min_min=('lead_time_minutes', 'min'),
        mean_steps=('alert_step_before_error', 'mean'),
    ).reset_index()
    cat_stats['pct_ant'] = (cat_stats['anticipated'] / cat_stats['count'] * 100).round(1)
    cat_stats = cat_stats.sort_values('mean_min', ascending=False)

    cat_rows = ""
    for _, r in cat_stats.iterrows():
        color = '#27ae60' if r['mean_min'] > 0 else '#e74c3c'
        cat_rows += f"""<tr>
            <td style="max-width:350px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{r['failure_category']}">{r['failure_category'][:80]}</td>
            <td>{int(r['count'])}</td>
            <td style="color:{color};font-weight:bold">{fmt_time(r['mean_min'])}</td>
            <td>{fmt_time(r['median_min'])}</td>
            <td>{fmt_time(r['max_min'])}</td>
            <td>{fmt_time(r['min_min'])}</td>
            <td>{r['mean_steps']:.0f} eventos</td>
            <td>{r['pct_ant']:.0f}%</td>
        </tr>"""

    # ── Top 10 MAIOR, MENOR ──
    top10_best = valid_tp.sort_values('lead_time_minutes', ascending=False).head(10)
    top10_worst = valid_tp.sort_values('lead_time_minutes', ascending=True).head(10)

    def make_top_rows(df_subset, is_best=True):
        rows = ""
        for _, r in df_subset.iterrows():
            cat = simplify_template(r['error_template'])[:60]
            lt = r['lead_time_minutes']
            color = '#27ae60' if lt > 0 else '#e74c3c'
            sign = '+' if lt > 0 else ''
            steps = int(r['alert_step_before_error']) if pd.notna(r.get('alert_step_before_error')) else '—'
            alert_ts = str(r['alert_timestamp_real'])[:19] if pd.notna(r.get('alert_timestamp_real')) else '—'
            error_ts = str(r['error_timestamp_real'])[:19] if pd.notna(r.get('error_timestamp_real')) else '—'
            rows += f"""<tr>
                <td>{r['test_id']}</td>
                <td style="color:{color};font-weight:bold">{sign}{fmt_time(lt)}</td>
                <td>{steps} eventos</td>
                <td><small>{alert_ts}</small></td>
                <td><small>{error_ts}</small></td>
                <td title="{cat}">{cat[:45]}{"..." if len(cat) > 45 else ""}</td>
            </tr>"""
        return rows

    best_rows = make_top_rows(top10_best, is_best=True)
    worst_rows = make_top_rows(top10_worst, is_best=False)

    # ── Por EventId ──
    eid_stats = valid_tp.groupby('first_error_eventid').agg(
        count=('lead_time_minutes', 'size'),
        mean_min=('lead_time_minutes', 'mean'),
        median_min=('lead_time_minutes', 'median'),
        anticipated=('anticipated', 'sum'),
        sample_template=('failure_category', 'first')
    ).reset_index().sort_values('count', ascending=False).head(15)

    eid_rows = ""
    for _, r in eid_stats.iterrows():
        color = '#27ae60' if r['mean_min'] > 0 else '#e74c3c'
        pct = r['anticipated'] / r['count'] * 100
        eid_rows += f"""<tr>
            <td><code>{r['first_error_eventid']}</code></td>
            <td>{int(r['count'])}</td>
            <td style="color:{color};font-weight:bold">{fmt_time(r['mean_min'])}</td>
            <td>{fmt_time(r['median_min'])}</td>
            <td>{pct:.0f}%</td>
            <td title="{r['sample_template']}">{str(r['sample_template'])[:60]}</td>
        </tr>"""

    # ── HTML ──
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LogGPT — Relatório de Lead Time Real por Falha</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{ --primary:#1a1a2e; --accent:#16213e; --emerald:#27ae60; --ruby:#e74c3c; --gold:#f39c12; --bg:#0f0f23; --surface:#1e1e3f; --text:#e0e0e0; }}
        body {{ font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); margin:0; line-height:1.7; }}
        .hero {{ background:linear-gradient(135deg,#0f2027,#203a43,#2c5364); padding:60px 20px; text-align:center; border-bottom:4px solid var(--emerald); }}
        .hero h1 {{ font-size:2.8rem; margin:0; font-weight:800; background:linear-gradient(90deg,#27ae60,#2ecc71,#1abc9c); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
        .hero p {{ font-size:1.1rem; opacity:0.8; margin-top:10px; }}
        .hero .subtitle {{ font-size:0.9rem; opacity:0.5; margin-top:5px; }}
        .container {{ max-width:1350px; margin:40px auto; padding:0 20px; }}
        section {{ background:var(--surface); border-radius:16px; padding:40px; margin-bottom:40px; box-shadow:0 8px 32px rgba(0,0,0,0.3); border:1px solid rgba(255,255,255,0.05); }}
        h2 {{ color:var(--emerald); font-weight:800; border-bottom:2px solid rgba(39,174,96,0.3); padding-bottom:12px; margin-top:0; }}
        h3 {{ color:#3498db; font-weight:600; margin-top:30px; }}
        .kpi-row {{ display:flex; gap:20px; flex-wrap:wrap; margin:30px 0; }}
        .kpi {{ flex:1; min-width:160px; background:linear-gradient(145deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02)); padding:25px; border-radius:12px; text-align:center; border:1px solid rgba(255,255,255,0.08); }}
        .kpi .value {{ font-size:2.2rem; font-weight:800; margin:8px 0; }}
        .kpi .label {{ font-size:0.75rem; text-transform:uppercase; letter-spacing:1.5px; color:#7f8c8d; font-weight:600; }}
        .kpi.green .value {{ color:var(--emerald); }}
        .kpi.red .value {{ color:var(--ruby); }}
        .kpi.gold .value {{ color:var(--gold); }}
        .kpi.blue .value {{ color:#3498db; }}
        table {{ width:100%; border-collapse:collapse; margin:20px 0; font-size:13px; }}
        th {{ background:rgba(39,174,96,0.15); color:var(--emerald); padding:12px 10px; text-align:left; font-weight:600; border-bottom:2px solid rgba(39,174,96,0.3); }}
        td {{ padding:10px; border-bottom:1px solid rgba(255,255,255,0.05); }}
        tr:hover {{ background:rgba(255,255,255,0.03); }}
        .img-container {{ text-align:center; margin:30px 0; }}
        .img-container img {{ max-width:100%; border-radius:12px; border:1px solid rgba(255,255,255,0.1); }}
        .flex-grid {{ display:flex; gap:30px; flex-wrap:wrap; }}
        .col {{ flex:1; min-width:300px; }}
        .badge {{ display:inline-block; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:700; }}
        .badge-green {{ background:rgba(39,174,96,0.2); color:#27ae60; }}
        .badge-red {{ background:rgba(231,76,60,0.2); color:#e74c3c; }}
        code {{ background:rgba(255,255,255,0.1); padding:2px 6px; border-radius:4px; font-size:12px; }}
        small {{ color:#7f8c8d; }}
        .note {{ background:rgba(52,152,219,0.1); border-left:4px solid #3498db; padding:15px 20px; border-radius:0 8px 8px 0; margin:20px 0; font-size:14px; }}
        @media(max-width:800px) {{ .flex-grid {{ flex-direction:column; }} .kpi-row {{ flex-direction:column; }} }}
    </style>
</head>
<body>

<div class="hero">
    <h1>⏱ Relatório de Lead Time Real por Falha</h1>
    <p>LogGPT Top-5 — OpenStack · Antecipação Temporal com Timestamps Reais</p>
    <p class="subtitle">Timestamps extraídos da coluna <code>time_hour</code> do CSV original (resolução de microssegundos)</p>
</div>

<div class="container">

    <!-- SEC 1: RESUMO -->
    <section>
        <h2>1. Resumo Geral de Antecipação</h2>
        <p>Métrica principal: <strong>lead_time = timestamp_erro_real − timestamp_detecção_modelo</strong>. Valores <span class="badge badge-green">positivos</span> = modelo detectou ANTES do erro; <span class="badge badge-red">negativos</span> = detecção reativa (após o erro).</p>

        <div class="kpi-row">
            <div class="kpi green">
                <div class="label">Sessões Antecipadas</div>
                <div class="value">{n_anticipated}/{n_tp}</div>
                <div class="label">({pct_anticipated:.1f}%)</div>
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
                <div class="label">Tempo real</div>
            </div>
            <div class="kpi red">
                <div class="label">Pior (Reativo)</div>
                <div class="value">{fmt_time(min_lt)}</div>
                <div class="label">Tempo real</div>
            </div>
        </div>

        <div class="note">
            📐 <strong>Média geral (todos TPs):</strong> {fmt_time(mean_lt)} · <strong>Mediana geral:</strong> {fmt_time(median_lt)} · <strong>Média em eventos:</strong> {mean_steps:.0f} eventos
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
        <p>Cada sessão categorizada pelo <code>error_template</code> do primeiro log com <code>anom_label=1</code>. O tempo de antecipação é calculado com timestamps reais da coluna <code>time_hour</code>.</p>

        <table>
            <tr>
                <th>Categoria de Falha</th>
                <th>Sessões</th>
                <th>Média (tempo)</th>
                <th>Mediana</th>
                <th>Máx</th>
                <th>Mín</th>
                <th>Média (eventos)</th>
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
        <p>Sessões com <strong>maior lead time real</strong> — o modelo alertou com mais antecedência temporal.</p>
        <table>
            <tr>
                <th>Sessão</th>
                <th>Lead Time</th>
                <th>Em Eventos</th>
                <th>Timestamp Detecção</th>
                <th>Timestamp 1º Erro</th>
                <th>Tipo de Falha</th>
            </tr>
            {best_rows}
        </table>
    </section>

    <!-- SEC 4: TOP 10 PIOR -->
    <section>
        <h2>4. Top 10 — Detecção Mais Tardia (Reativa)</h2>
        <p>Sessões onde o modelo disparou o alerta <strong>depois</strong> do 1º erro real. Lead time negativo = detecção reativa.</p>
        <table>
            <tr>
                <th>Sessão</th>
                <th>Lead Time</th>
                <th>Em Eventos</th>
                <th>Timestamp Detecção</th>
                <th>Timestamp 1º Erro</th>
                <th>Tipo de Falha</th>
            </tr>
            {worst_rows}
        </table>
    </section>

    <!-- SEC 5: VISUAL -->
    <section>
        <h2>5. Análise Visual Detecção vs Erro</h2>
        <div class="flex-grid">
            <div class="col img-container">
                <img src="data:image/png;base64,{b64_scatter}" alt="Scatter">
                <p style="color:#7f8c8d;font-size:13px;"><b>Pontos abaixo da diagonal</b> = modelo detectou antes do erro. Quanto mais abaixo-esquerda, melhor.</p>
            </div>
            <div class="col img-container">
                <img src="data:image/png;base64,{b64_bands}" alt="Faixas">
                <p style="color:#7f8c8d;font-size:13px;"><b>Distribuição das faixas</b> de antecipação em tempo real.</p>
            </div>
        </div>
    </section>

    <!-- SEC 6: POR EVENT ID -->
    <section>
        <h2>6. Lead Time por EventId de Erro</h2>
        <p>Cada <code>first_error_eventid</code> representa um template único de falha. Top 15 mais frequentes:</p>
        <table>
            <tr>
                <th>EventId</th>
                <th>Ocorrências</th>
                <th>Média (tempo)</th>
                <th>Mediana</th>
                <th>% Antecipadas</th>
                <th>Exemplo</th>
            </tr>
            {eid_rows}
        </table>
    </section>

</div>
</body>
</html>"""

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ Relatório exportado: {REPORT_PATH}")
    print(f"\n📈 RESUMO TEMPORAL:")
    print(f"   Total TPs: {n_tp}")
    print(f"   Antecipadas: {n_anticipated} ({pct_anticipated:.1f}%)")
    print(f"   Média lead time (antecipados): {fmt_time(mean_ant)}")
    print(f"   Mediana (antecipados): {fmt_time(median_ant)}")
    print(f"   Maior antecipação: {fmt_time(max_lt)}")
    print(f"   Pior (reativo): {fmt_time(min_lt)}")


if __name__ == "__main__":
    generate_report()
