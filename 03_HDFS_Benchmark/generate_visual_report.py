# -*- coding: utf-8 -*-
"""
HDFS Visual Report Generator
Creates comprehensive HTML report with interactive charts and detailed analysis
"""
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("=" * 70)
print("  HDFS VISUAL REPORT GENERATOR")
print("=" * 70)

# Load data
print("\n[1/5] Loading data...")
with open("detection_results_partial.pkl", "rb") as f:
    results = pickle.load(f)

with open("mega_analysis_results.json", "r") as f:
    analysis = json.load(f)

print(f"  Loaded {len(results)} session results")
print(f"  Loaded analysis with {len(analysis['categories'])} categories")

# Prepare data
tp_results = [r for r in results if r['label'] == 1 and r['is_detected']]
fn_results = [r for r in results if r['label'] == 1 and not r['is_detected']]
fp_results = [r for r in results if r['label'] == 0 and r['is_detected']]
tn_results = [r for r in results if r['label'] == 0 and not r['is_detected']]

# Create output directory
output_dir = Path("visual_report")
output_dir.mkdir(exist_ok=True)

# =========================================================================
# CHART 1: Confusion Matrix
# =========================================================================
print("\n[2/5] Creating confusion matrix...")

fig = go.Figure(data=go.Heatmap(
    z=[[len(tp_results), len(fn_results)],
       [len(fp_results), len(tn_results)]],
    x=['Predicted Anomaly', 'Predicted Normal'],
    y=['Actual Anomaly', 'Actual Normal'],
    text=[[f'TP<br>{len(tp_results):,}', f'FN<br>{len(fn_results):,}'],
          [f'FP<br>{len(fp_results):,}', f'TN<br>{len(tn_results):,}']],
    texttemplate='%{text}',
    textfont={"size": 20},
    colorscale='RdYlGn',
    showscale=False
))

fig.update_layout(
    title='Confusion Matrix - HDFS Anomaly Detection',
    xaxis_title='Predicted',
    yaxis_title='Actual',
    height=500,
    font=dict(size=14)
)

fig.write_html(str(output_dir / "confusion_matrix.html"))
print("  ✓ Confusion matrix saved")

# =========================================================================
# CHART 2: Lead Time Distribution
# =========================================================================
print("\n[3/5] Creating lead time distribution...")

lead_times = [r['lead_time'] for r in tp_results if r['lead_time'] > 0]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Lead Time Distribution (All)', 'Lead Time Distribution (0-60 min)',
                    'Lead Time Box Plot', 'Lead Time Cumulative Distribution'),
    specs=[[{"type": "histogram"}, {"type": "histogram"}],
           [{"type": "box"}, {"type": "scatter"}]]
)

# Histogram - All
fig.add_trace(
    go.Histogram(x=lead_times, nbinsx=50, name='All Lead Times',
                 marker_color='steelblue'),
    row=1, col=1
)

# Histogram - Zoomed (0-60 min)
lead_times_short = [lt for lt in lead_times if lt <= 60]
fig.add_trace(
    go.Histogram(x=lead_times_short, nbinsx=30, name='0-60 min',
                 marker_color='coral'),
    row=1, col=2
)

# Box plot
fig.add_trace(
    go.Box(y=lead_times, name='Lead Time', marker_color='lightseagreen'),
    row=2, col=1
)

# Cumulative distribution
sorted_leads = np.sort(lead_times)
cumulative = np.arange(1, len(sorted_leads) + 1) / len(sorted_leads) * 100
fig.add_trace(
    go.Scatter(x=sorted_leads, y=cumulative, mode='lines',
               name='Cumulative %', line=dict(color='darkviolet', width=2)),
    row=2, col=2
)

fig.update_xaxes(title_text="Lead Time (minutes)", row=1, col=1)
fig.update_xaxes(title_text="Lead Time (minutes)", row=1, col=2)
fig.update_xaxes(title_text="Lead Time (minutes)", row=2, col=2)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=2)
fig.update_yaxes(title_text="Lead Time (minutes)", row=2, col=1)
fig.update_yaxes(title_text="Cumulative %", row=2, col=2)

fig.update_layout(
    title_text="Lead Time Analysis - HDFS Anomaly Detection",
    showlegend=False,
    height=800
)

fig.write_html(str(output_dir / "lead_time_distribution.html"))
print("  ✓ Lead time distribution saved")

# =========================================================================
# CHART 3: Category Performance
# =========================================================================
print("\n[4/5] Creating category performance charts...")

categories_df = pd.DataFrame(analysis['categories'])
categories_df = categories_df.sort_values('total_sessions', ascending=False)

# Bar chart - Recall by category
fig = go.Figure()

fig.add_trace(go.Bar(
    x=categories_df['category'],
    y=categories_df['recall'] * 100,
    text=[f"{r:.1f}%" for r in categories_df['recall'] * 100],
    textposition='outside',
    marker_color='mediumseagreen',
    name='Recall'
))

fig.update_layout(
    title='Recall by Error Category',
    xaxis_title='Category',
    yaxis_title='Recall (%)',
    yaxis_range=[0, 105],
    height=500,
    font=dict(size=12)
)

fig.write_html(str(output_dir / "category_recall.html"))

# Stacked bar - Detected vs Missed
fig = go.Figure()

fig.add_trace(go.Bar(
    name='Detected',
    x=categories_df['category'],
    y=categories_df['detected'],
    marker_color='lightgreen',
    text=categories_df['detected'],
    textposition='inside'
))

fig.add_trace(go.Bar(
    name='Missed',
    x=categories_df['category'],
    y=categories_df['missed'],
    marker_color='lightcoral',
    text=categories_df['missed'],
    textposition='inside'
))

fig.update_layout(
    title='Detection Performance by Category',
    xaxis_title='Category',
    yaxis_title='Sessions',
    barmode='stack',
    height=500,
    font=dict(size=12)
)

fig.write_html(str(output_dir / "category_performance.html"))

# Lead time by category
fig = go.Figure()

categories_with_lead = categories_df[categories_df['avg_lead_time_min'] > 0]

fig.add_trace(go.Bar(
    x=categories_with_lead['category'],
    y=categories_with_lead['avg_lead_time_min'],
    text=[f"{lt:.1f}m" for lt in categories_with_lead['avg_lead_time_min']],
    textposition='outside',
    marker_color='steelblue',
    name='Avg Lead Time'
))

fig.update_layout(
    title='Average Lead Time by Category',
    xaxis_title='Category',
    yaxis_title='Lead Time (minutes)',
    height=500,
    font=dict(size=12)
)

fig.write_html(str(output_dir / "category_lead_time.html"))

print("  ✓ Category performance charts saved")

# =========================================================================
# CHART 4: Alert Loss Distribution
# =========================================================================
print("\n[5/5] Creating alert loss distribution...")

tp_losses = [r['alert_loss'] for r in tp_results]
fp_losses = [r['alert_loss'] for r in fp_results]

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('True Positives (TP)', 'False Positives (FP)')
)

# TP losses
fig.add_trace(
    go.Histogram(x=tp_losses, nbinsx=50, name='TP Loss',
                 marker_color='green', opacity=0.7),
    row=1, col=1
)

# FP losses
fig.add_trace(
    go.Histogram(x=fp_losses, nbinsx=30, name='FP Loss',
                 marker_color='red', opacity=0.7),
    row=1, col=2
)

# Add threshold line
threshold = 0.2863
fig.add_vline(x=threshold, line_dash="dash", line_color="black",
              annotation_text=f"Threshold={threshold:.4f}",
              row=1, col=1)
fig.add_vline(x=threshold, line_dash="dash", line_color="black",
              annotation_text=f"Threshold={threshold:.4f}",
              row=1, col=2)

fig.update_xaxes(title_text="Alert Loss", row=1, col=1)
fig.update_xaxes(title_text="Alert Loss", row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=2)

fig.update_layout(
    title_text="Alert Loss Distribution - TP vs FP",
    showlegend=False,
    height=500
)

fig.write_html(str(output_dir / "loss_distribution.html"))
print("  ✓ Alert loss distribution saved")

# =========================================================================
# GENERATE HTML REPORT
# =========================================================================
print("\n[6/6] Generating HTML report...")

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HDFS Anomaly Detection - Visual Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #3498db;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        iframe {{
            width: 100%;
            border: none;
        }}
        .info-box {{
            background: #ecf0f1;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .warning-box {{
            background: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .success-box {{
            background: #d4edda;
            padding: 15px;
            border-left: 4px solid #28a745;
            margin: 20px 0;
            border-radius: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <h1>📊 HDFS Anomaly Detection - Visual Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Dataset:</strong> HDFS (11.17M log lines, 72,661 sessions)</p>
    <p><strong>Model:</strong> LogGPT-Small (28.98M parameters)</p>
    <p><strong>Threshold:</strong> 0.2863 (8.0σ adaptive)</p>

    <h2>📈 Key Metrics</h2>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{analysis['summary']['precision']:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Recall</div>
            <div class="metric-value">{analysis['summary']['recall']:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">F1 Score</div>
            <div class="metric-value">{analysis['summary']['f1']:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">True Positives</div>
            <div class="metric-value">{analysis['summary']['tp']:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">True Negatives</div>
            <div class="metric-value">{analysis['summary']['tn']:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Lead Time</div>
            <div class="metric-value">{analysis['lead_time_stats']['mean']:.0f}m</div>
        </div>
    </div>

    <div class="success-box">
        <strong>✅ Excellent Performance:</strong> The model achieves 95% precision with 82% recall, 
        correctly identifying 13,855 anomalies while maintaining only 733 false alarms out of 55,823 normal sessions.
    </div>

    <h2>🎯 Confusion Matrix</h2>
    <div class="chart-container">
        <iframe src="confusion_matrix.html" height="550"></iframe>
    </div>

    <h2>⏱️ Lead Time Analysis</h2>
    <div class="info-box">
        <strong>Lead Time</strong> = Time between first detection and actual failure (last log).
        <ul>
            <li><strong>Mean:</strong> {analysis['lead_time_stats']['mean']:.1f} minutes (≈{analysis['lead_time_stats']['mean']/60:.1f} hours)</li>
            <li><strong>Median:</strong> {analysis['lead_time_stats']['median']:.1f} minutes</li>
            <li><strong>Max:</strong> {analysis['lead_time_stats']['max']:.1f} minutes (≈{analysis['lead_time_stats']['max']/60:.1f} hours)</li>
        </ul>
    </div>
    <div class="chart-container">
        <iframe src="lead_time_distribution.html" height="850"></iframe>
    </div>

    <h2>📂 Performance by Error Category</h2>
    <table>
        <thead>
            <tr>
                <th>Category</th>
                <th>Total</th>
                <th>Detected</th>
                <th>Missed</th>
                <th>Recall</th>
                <th>Avg Lead Time</th>
            </tr>
        </thead>
        <tbody>
"""

for cat in categories_df.itertuples():
    html_content += f"""
            <tr>
                <td><strong>{cat.category}</strong></td>
                <td>{cat.total_sessions:,}</td>
                <td>{cat.detected:,}</td>
                <td>{cat.missed:,}</td>
                <td>{cat.recall:.1%}</td>
                <td>{cat.avg_lead_time_min:.1f} min</td>
            </tr>
"""

html_content += """
        </tbody>
    </table>

    <div class="chart-container">
        <iframe src="category_recall.html" height="550"></iframe>
    </div>

    <div class="chart-container">
        <iframe src="category_performance.html" height="550"></iframe>
    </div>

    <div class="chart-container">
        <iframe src="category_lead_time.html" height="550"></iframe>
    </div>

    <div class="warning-box">
        <strong>⚠️ Key Finding:</strong> NameSystem/BlockMap category (1,307 sessions) has 0% recall. 
        These sessions have only ~2 logs and patterns indistinguishable from normal behavior.
    </div>

    <h2>🔍 Alert Loss Distribution</h2>
    <div class="info-box">
        <strong>Alert Loss</strong> measures how anomalous a pattern is. Higher loss = more unusual pattern.
        <ul>
            <li><strong>Threshold:</strong> 0.2863 (sessions above this are flagged as anomalies)</li>
            <li><strong>TP Median Loss:</strong> 0.4858 (clearly above threshold)</li>
            <li><strong>FP Median Loss:</strong> 0.3103 (borderline cases)</li>
        </ul>
    </div>
    <div class="chart-container">
        <iframe src="loss_distribution.html" height="550"></iframe>
    </div>

    <h2>🎓 Key Insights</h2>
    <div class="success-box">
        <h3>✅ Strengths</h3>
        <ul>
            <li><strong>High Precision (95%):</strong> When the model alerts, it's correct 95% of the time</li>
            <li><strong>Excellent Specificity (98.7%):</strong> Very few false alarms on normal sessions</li>
            <li><strong>Long Lead Times:</strong> Up to 15 hours of advance warning for some anomalies</li>
            <li><strong>Category Excellence:</strong> "Other Exception" category achieves 99.8% recall</li>
        </ul>
    </div>

    <div class="warning-box">
        <h3>⚠️ Limitations</h3>
        <ul>
            <li><strong>Short Sessions:</strong> 97.9% of false negatives are sessions with ≤2 logs (insufficient context)</li>
            <li><strong>NameSystem/BlockMap:</strong> 1,307 sessions (7.8% of anomalies) completely undetectable</li>
            <li><strong>InterruptedIOException:</strong> 66.5% recall due to many short sessions</li>
        </ul>
    </div>

    <div class="info-box">
        <h3>💡 Recommendations</h3>
        <ul>
            <li><strong>Hybrid Approach:</strong> Use rule-based classification for sessions with ≤2 logs</li>
            <li><strong>Session Aggregation:</strong> Combine related sessions within time windows to enrich context</li>
            <li><strong>Threshold Tuning:</strong> Consider lower threshold (~0.20) to capture borderline anomalies</li>
        </ul>
    </div>

    <hr style="margin: 40px 0;">
    <p style="text-align: center; color: #7f8c8d;">
        <em>Report generated by HDFS Visual Report Generator</em>
    </p>
</body>
</html>
"""

# Save HTML report
with open(str(output_dir / "index.html"), "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\n{'='*70}")
print(f"  ✅ VISUAL REPORT COMPLETE!")
print(f"{'='*70}")
print(f"\n  📁 Report location: {output_dir.absolute()}")
print(f"  🌐 Open: {(output_dir / 'index.html').absolute()}")
print(f"\n  Generated files:")
print(f"    - index.html (main report)")
print(f"    - confusion_matrix.html")
print(f"    - lead_time_distribution.html")
print(f"    - category_recall.html")
print(f"    - category_performance.html")
print(f"    - category_lead_time.html")
print(f"    - loss_distribution.html")
print(f"\n{'='*70}")
