# -*- coding: utf-8 -*-
"""
Relatório Completo TCC — LogGPT: Detecção de Anomalias em Logs
Gera HTML com análise comparativa de OpenStack, HDFS e BGL.
"""
import json, io, base64, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path("d:/ProLog")
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def fmt(m):
    if m is None or pd.isna(m): return "—"
    if abs(m)<1: return f"{m*60:.1f}s"
    if abs(m)<60: return f"{m:.1f}min"
    if abs(m)<1440: return f"{m/60:.1f}h"
    return f"{m/1440:.1f}d"

# ─── METRICS ───
OS_M = {"name":"OpenStack","precision":0.972,"recall":0.960,"f1":0.966,"accuracy":0.977,
        "tp":169,"fn":7,"fp":5,"tn":239,"total_sessions":420,"threshold":"Top-5",
        "lt_mean_min":3.8,"lt_median_min":3.5,"lt_max_min":12.3,"lt_pct_ant":62.7,
        "n_templates":30,"dataset_period":"Jun-Jul 2018","domain":"Cloud Computing (IaaS)",
        "source":"Loghub/OpenStack","logs_total":"~424K","sessions_total":420,
        "train_sessions":250,"test_sessions":420}

HDFS_M = {"name":"HDFS","precision":0.950,"recall":0.823,"f1":0.882,"accuracy":0.949,
          "tp":13855,"fn":2983,"fp":724,"tn":55099,"total_sessions":72661,"threshold":0.286,
          "lt_mean_min":161.2,"lt_median_min":16.1,"lt_max_min":898.0,"lt_pct_ant":53.0,
          "n_templates":29,"dataset_period":"Nov 2008","domain":"Distributed Storage",
          "source":"Loghub/HDFS","logs_total":"~11M","sessions_total":"575K blocks",
          "train_sessions":"~460K","test_sessions":"~72K"}

BGL_M = {"name":"BGL","precision":0.489,"recall":1.0,"f1":0.657,"accuracy":0.489,
         "tp":489,"fn":0,"fp":511,"tn":0,"total_sessions":1000,"threshold":"Top-5",
         "lt_mean_min":None,"lt_median_min":None,"lt_max_min":None,"lt_pct_ant":None,
         "n_templates":242,"dataset_period":"Jun 2005 - Jan 2006","domain":"HPC Supercomputer",
         "source":"Loghub/BGL","logs_total":"~4.7M","sessions_total":"~370K windows",
         "train_sessions":"OpenStack model","test_sessions":"1000 windows"}

ALL = [OS_M, HDFS_M, BGL_M]

# ─── CHARTS ───
def chart_metrics_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['precision','recall','f1']
    titles = ['Precision','Recall','F1-Score']
    colors = ['#27ae60','#3498db','#e74c3c']
    for i,(m,t) in enumerate(zip(metrics,titles)):
        vals = [d[m]*100 for d in ALL]
        bars = axes[i].bar(['OpenStack','HDFS','BGL'], vals, color=colors, edgecolor='white', linewidth=2)
        axes[i].set_title(t, fontsize=16, fontweight='bold')
        axes[i].set_ylim(0, 110)
        axes[i].axhline(90, color='gray', linestyle='--', alpha=0.5)
        for b,v in zip(bars,vals):
            axes[i].text(b.get_x()+b.get_width()/2, v+2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)
    plt.tight_layout()
    return b64(fig)

def chart_confusion_matrices():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i,d in enumerate(ALL):
        cm = np.array([[d['tn'],d['fp']],[d['fn'],d['tp']]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if i<2 else 'Reds',
                    xticklabels=['Normal','Anômalo'], yticklabels=['Normal','Anômalo'],
                    ax=axes[i], cbar=False, annot_kws={"size":14})
        axes[i].set_title(d['name'], fontsize=15, fontweight='bold')
        axes[i].set_ylabel('Real'); axes[i].set_xlabel('Predito')
    plt.tight_layout()
    return b64(fig)

def chart_f1_radar():
    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    cats = ['Precision','Recall','F1','Accuracy']
    for d,c,ls in zip(ALL,['#27ae60','#3498db','#e74c3c'],['-','--',':']):
        vals = [d['precision'],d['recall'],d['f1'],d['accuracy']]
        vals += vals[:1]
        angles = np.linspace(0,2*np.pi,len(cats),endpoint=False).tolist()+[0]
        ax.plot(angles, vals, c, linewidth=2, linestyle=ls, label=d['name'])
        ax.fill(angles, vals, c, alpha=0.1)
    ax.set_xticks(np.linspace(0,2*np.pi,len(cats),endpoint=False))
    ax.set_xticklabels(cats, fontsize=12)
    ax.set_ylim(0,1.1)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_title("Comparativo de Métricas", fontsize=15, fontweight='bold', pad=20)
    return b64(fig)

def chart_leadtime():
    fig, ax = plt.subplots(figsize=(10, 5))
    names = ['OpenStack','HDFS']
    means = [OS_M['lt_mean_min'], HDFS_M['lt_mean_min']]
    medians = [OS_M['lt_median_min'], HDFS_M['lt_median_min']]
    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x-w/2, means, w, label='Média', color='#27ae60', edgecolor='white')
    b2 = ax.bar(x+w/2, medians, w, label='Mediana', color='#3498db', edgecolor='white')
    ax.set_ylabel('Lead Time (minutos)', fontsize=13)
    ax.set_title('Lead Time Real de Antecipação', fontsize=16, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=13)
    for b in [b1,b2]:
        for bar in b:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, fmt(bar.get_height()),
                    ha='center', fontweight='bold', fontsize=11)
    ax.legend(fontsize=12)
    return b64(fig)

def chart_template_diversity():
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [d['name'] for d in ALL]
    vals = [d['n_templates'] for d in ALL]
    colors = ['#27ae60','#3498db','#e74c3c']
    bars = ax.bar(names, vals, color=colors, edgecolor='white', linewidth=2)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2, v+3, str(v), ha='center', fontweight='bold', fontsize=14)
    ax.set_title('Diversidade de Templates por Dataset', fontsize=16, fontweight='bold')
    ax.set_ylabel('Nº de Templates Únicos', fontsize=13)
    return b64(fig)

def chart_pipeline():
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 1)
    steps = [
        (0.5, "📋 Coleta\nde Logs", "#1abc9c"),
        (2.0, "🔧 Parsing\n(Drain)", "#2ecc71"),
        (3.5, "📦 Sessões\n(Agrupamento)", "#27ae60"),
        (5.0, "🧠 Treino\n(Causal LM)", "#3498db"),
        (6.5, "🔍 Detecção\n(Top-K)", "#9b59b6"),
        (8.0, "⏱ Lead\nTime", "#e67e22"),
        (9.5, "📊 Relatório\nFinal", "#e74c3c"),
    ]
    for x,label,c in steps:
        ax.add_patch(plt.Circle((x,0.5),0.35,facecolor=c,edgecolor='white',linewidth=3,zorder=2))
        ax.text(x,0.5,label,ha='center',va='center',fontsize=8,fontweight='bold',color='white',zorder=3)
    for i in range(len(steps)-1):
        ax.annotate('',xy=(steps[i+1][0]-0.4,0.5),xytext=(steps[i][0]+0.4,0.5),
                    arrowprops=dict(arrowstyle='->',color='gray',lw=2))
    ax.axis('off')
    ax.set_title('Pipeline LogGPT — Visão Geral', fontsize=16, fontweight='bold', pad=10)
    return b64(fig)

# ─── HTML GENERATION ───
def generate():
    print("📊 Generating charts...")
    c_metrics = chart_metrics_comparison()
    c_cm = chart_confusion_matrices()
    c_radar = chart_f1_radar()
    c_lt = chart_leadtime()
    c_tmpl = chart_template_diversity()
    c_pipe = chart_pipeline()

    print("📝 Building HTML...")
    # Load HTML template
    from tcc_report_template import build_html
    html = build_html(c_metrics, c_cm, c_radar, c_lt, c_tmpl, c_pipe, OS_M, HDFS_M, BGL_M)

    out = DOCS / "relatorio_completo_tcc.html"
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✅ Report saved: {out}")

if __name__ == "__main__":
    generate()
