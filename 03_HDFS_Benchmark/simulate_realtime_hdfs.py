"""
LogGPT Real-Time Simulator — HDFS Dataset
==========================================
Replays the HDFS test set in real-time, grouped by exact timestamp (batch),
and serves two web dashboards:
  - Port 5557: Metrics dashboard (KPIs, alerts table)
  - Port 5558: Internal process dashboard (4 terminals)

Detection method: cross-entropy loss vs adaptive threshold
(same as detect.py / HDFS benchmark pipeline).
"""

import os
import sys
import time
import json
import argparse
import threading
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
import polars as pl
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# ── Add parent dir for shared model import ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LogGPT, GPTConfig
from config import (
    MODEL_NAME, MODEL_DIR, DEVICE, DATA_FILE,
    SKIP_START_LOGS, THRESHOLD as DEFAULT_THRESHOLD,
    INFER_SCHEMA_LENGTH, TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT,
    SESSION_ID_COL, TIMESTAMP_COL, TEMPLATE_COL, LABEL_COL,
    set_seeds
)

# ── Constants ─────────────────────────────────────────────────────────────────
PORT_METRICS  = 5557
PORT_PROCESS  = 5558
K             = 5      # For display only in process panel (Top-K probabilities)

# ── Shared state ──────────────────────────────────────────────────────────────
STATE = {
    "status": "loading",
    "current_timestamp": "...",
    "speed_mode": "realtime",
    "logs_processed": 0,
    "total_logs": 0,
    "sessions_active": 0,
    "sessions_completed": 0,
    "total_alerts": 0,
    "tp": 0, "tn": 0, "fp": 0, "fn": 0,
    "precision": None, "recall": None, "f1": None, "accuracy": None,
    "lead_time_mean": None,
    "alerts": [],
    "raw_feed": [],
    "template_feed": [],
    "process_feed": [],
    "session_context": None,
    "finished": False,
}

def safe_float(v):
    if v is None: return None
    try:
        return float(v)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD 1: Metrics (Port 5557)
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_METRICS_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LogGPT HDFS — Dashboard de Métricas</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#040a10;color:#e0e0e0;padding:20px;min-height:100vh}
  h1{text-align:center;font-size:1.7rem;margin-bottom:4px;background:linear-gradient(90deg,#00bcd4,#4caf50);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .subtitle{text-align:center;color:#888;font-size:.85rem;margin-bottom:16px}
  .progress-outer{background:#0d1f0d;border-radius:20px;height:28px;overflow:hidden;position:relative;margin-bottom:20px;border:1px solid #1a3a1a}
  .progress-inner{height:100%;background:linear-gradient(90deg,#00695c,#4caf50);border-radius:20px;transition:width .5s ease}
  .progress-text{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:.78rem;font-weight:600;color:#fff}
  .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
  .card{background:linear-gradient(145deg,#0d1a0d,#111);border-radius:12px;padding:18px 14px;text-align:center;border:1px solid transparent}
  .card.cyan{border-color:#00bcd444}.card.green{border-color:#4caf5044}.card.blue{border-color:#1976d244}.card.red{border-color:#ef535044}.card.yellow{border-color:#ffca2844}
  .label{font-size:.7rem;text-transform:uppercase;letter-spacing:1px;color:#888;margin-bottom:6px}
  .value{font-size:1.9rem;font-weight:700}
  .card.cyan .value{color:#00bcd4}.card.green .value{color:#4caf50}.card.blue .value{color:#42a5f5}.card.red .value{color:#ef5350}.card.yellow .value{color:#ffca28}
  .section{background:#0a0f0a;border-radius:12px;padding:18px;margin-bottom:16px;border:1px solid #1a3a1a}
  h2{font-size:1rem;margin-bottom:14px;color:#4caf50;border-bottom:1px solid #1a3a1a;padding-bottom:8px}
  table{width:100%;border-collapse:collapse;font-size:.82rem}
  th{text-align:left;padding:8px 10px;color:#00bcd4;border-bottom:1px solid #1a3a1a;font-weight:600}
  td{padding:7px 10px;border-bottom:1px solid #0d1a0d}
  tr:hover{background:rgba(255,255,255,.02)}
  .tp{color:#4caf50} .fp{color:#ffca28}
  .lt-before{color:#4caf50} .lt-after{color:#ffca28}
  .badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:.7rem;font-weight:700}
  .badge-wait{background:rgba(66,165,245,.2);color:#42a5f5}
  .finished-banner{background:linear-gradient(135deg,#1b5e20,#2e7d32);border-radius:10px;padding:20px;text-align:center;margin-bottom:20px;border:2px solid #4caf50}
  .timestamp{color:#00bcd4;font-size:.9rem;text-align:center;margin-bottom:14px}
</style>
</head>
<body>
<h1>🧠 LogGPT HDFS — Dashboard de Métricas</h1>
<p class="subtitle">Visão Gerencial (Porta 5557) · Detecção por Cross-Entropy Loss</p>

<div id="finished-banner" style="display:none" class="finished-banner">
  <h2>🏁 Simulação HDFS Concluída!</h2>
  <p style="color:rgba(255,255,255,.8)">Veja o relatório no console.</p>
</div>

<p class="timestamp" id="ts">⏱ Aguardando início...</p>

<div class="progress-outer">
  <div class="progress-inner" id="pbar" style="width:0%"></div>
  <div class="progress-text" id="ptext">0 / 0 logs (0%)</div>
</div>

<div class="grid">
  <div class="card cyan"><div class="label">Logs Processados</div><div class="value" id="logs">0</div></div>
  <div class="card blue"><div class="label">Sessões Ativas</div><div class="value" id="active">0</div></div>
  <div class="card green"><div class="label">Concluídas</div><div class="value" id="done">0</div></div>
  <div class="card red"><div class="label">Alertas</div><div class="value" id="alerts">0</div></div>
</div>

<div class="grid">
  <div class="card green"><div class="label">True Positives</div><div class="value" id="tp">0</div></div>
  <div class="card blue"><div class="label">True Negatives</div><div class="value" id="tn">0</div></div>
  <div class="card yellow"><div class="label">False Positives</div><div class="value" id="fp">0</div></div>
  <div class="card red"><div class="label">False Negatives</div><div class="value" id="fn">0</div></div>
</div>

<div class="grid">
  <div class="card green"><div class="label">Precision</div><div class="value" id="prec">—</div></div>
  <div class="card blue"><div class="label">Recall</div><div class="value" id="rec">—</div></div>
  <div class="card yellow"><div class="label">F1-Score</div><div class="value" id="f1">—</div></div>
  <div class="card cyan"><div class="label">Lead Time Médio</div><div class="value" id="lt">—</div></div>
</div>

<div class="section">
  <h2>🚨 Alertas Emitidos (<span id="alert-count">0</span>)</h2>
  <table>
    <thead><tr><th>#</th><th>Sessão (BlockId)</th><th>Timestamp do Alerta</th><th>Loss</th><th>Lead Time</th><th>Resultado</th></tr></thead>
    <tbody id="alert-table"><tr><td colspan="6" style="color:#888;text-align:center">Nenhum alerta ainda</td></tr></tbody>
  </table>
</div>

<script>
function fmtTime(m){
  if(m==null)return '—';
  if(Math.abs(m)<1)return (m*60).toFixed(1)+'s';
  if(Math.abs(m)<60)return m.toFixed(1)+'min';
  return (m/60).toFixed(1)+'h';
}

function update(){
  fetch('/api/state').then(r=>r.json()).then(d=>{
    document.getElementById('ts').textContent='⏱ '+d.current_timestamp;
    let pct=d.total_logs?((d.logs_processed/d.total_logs)*100):0;
    document.getElementById('pbar').style.width=pct+'%';
    document.getElementById('ptext').textContent=d.logs_processed.toLocaleString()+' / '+d.total_logs.toLocaleString()+' logs ('+pct.toFixed(1)+'%)';
    document.getElementById('logs').textContent=d.logs_processed.toLocaleString();
    document.getElementById('active').textContent=d.sessions_active;
    document.getElementById('done').textContent=d.sessions_completed;
    document.getElementById('alerts').textContent=d.total_alerts;
    document.getElementById('alert-count').textContent=d.total_alerts;
    document.getElementById('tp').textContent=d.tp;
    document.getElementById('tn').textContent=d.tn;
    document.getElementById('fp').textContent=d.fp;
    document.getElementById('fn').textContent=d.fn;
    document.getElementById('prec').textContent=d.precision?(d.precision*100).toFixed(1)+'%':'—';
    document.getElementById('rec').textContent=d.recall?(d.recall*100).toFixed(1)+'%':'—';
    document.getElementById('f1').textContent=d.f1?(d.f1*100).toFixed(1)+'%':'—';
    document.getElementById('lt').textContent=fmtTime(d.lead_time_mean);

    if(d.alerts&&d.alerts.length>0){
      var html='';
      d.alerts.slice().reverse().forEach(function(a,i){
        var lt=a.lead_time;
        var ltStr='—';var ltCls='';
        if(typeof lt==='number'&&lt>0){ltStr=fmtTime(lt)+' antes';ltCls='lt-before';}
        else if(typeof lt==='number'){ltStr=fmtTime(Math.abs(lt))+' depois';ltCls='lt-after';}
        var lossStr=a.loss!=null?a.loss.toFixed(3):'—';
        var res='<span class="badge badge-wait">…</span>';
        if(a.is_tp===true)res='<span class="tp">✅ TP</span>';
        else if(a.is_tp===false)res='<span class="fp">⚠️ FP</span>';
        html+='<tr><td>'+(d.alerts.length-i)+'</td><td>'+String(a.session_id||'').substring(0,20)+'</td><td>'+
          String(a.alert_time||'').substring(0,25)+'</td><td style="color:#ff7043">'+lossStr+'</td><td class="'+ltCls+'">'+ltStr+'</td><td>'+res+'</td></tr>';
      });
      document.getElementById('alert-table').innerHTML=html;
    }

    if(d.finished){
      document.getElementById('finished-banner').style.display='block';
      return;
    }
    setTimeout(update,400);
  }).catch(function(){setTimeout(update,1000)});
}
update();
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD 2: Internal Process (Port 5558)
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_PROCESS_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LogGPT HDFS — Painel de Processo</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#040a10;color:#e0e0e0;padding:20px;height:100vh;display:flex;flex-direction:column}
  h1{text-align:center;font-size:1.6rem;margin-bottom:6px;color:#00bcd4}
  .subtitle{text-align:center;color:#888;font-size:.85rem;margin-bottom:20px}
  .grid-2x2{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:20px;flex:1;min-height:0}
  .terminal-box{background:#04100a;border:1px solid #1a3a3a;border-radius:10px;display:flex;flex-direction:column;overflow:hidden;box-shadow:0 4px 6px rgba(0,0,0,0.3)}
  .terminal-header{background:#061a14;padding:10px 16px;font-weight:600;font-size:.9rem;border-bottom:1px solid #1a3a3a;display:flex;justify-content:space-between;align-items:center}
  .header-raw{border-top:3px solid #00bcd4;color:#00bcd4}
  .header-template{border-top:3px solid #ffca28;color:#ffca28}
  .header-detect{border-top:3px solid #4caf50;color:#4caf50}
  .header-session{border-top:3px solid #80cbc4;color:#80cbc4}
  .badge{color:#fff;padding:2px 8px;border-radius:12px;font-size:.75rem}
  .terminal-content{flex:1;overflow-y:auto;padding:16px;font-family:'Consolas','Courier New',monospace;font-size:.8rem;line-height:1.6;color:#ccc}
  .terminal-content::-webkit-scrollbar{width:8px}
  .terminal-content::-webkit-scrollbar-thumb{background:#1a4a3a;border-radius:4px}
  .log-line{border-bottom:1px dashed #0d2a1a;padding-bottom:6px;margin-bottom:6px}
  .ts{color:#4dd0e1} .sid{color:#607d8b} .msg{color:#e0f2f1}
  .tmplt{color:#ffe082} .wild{color:#ef5350;font-weight:bold} .eid{color:#80cbc4}
  .pipe-title{color:#4caf50;font-weight:bold;margin-top:10px}
  .pipe-info{color:#aaa} .pipe-val{color:#fff}
  .loss-bar-wrap{display:flex;align-items:center;margin:4px 0;gap:8px}
  .loss-label{width:60px;font-size:.75rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .loss-bar{flex:1;height:14px;background:#0d2a1a;border-radius:4px;overflow:hidden}
  .loss-fill{height:100%;background:linear-gradient(90deg,#4caf50,#ffca28,#ef5350);border-radius:4px;transition:width .3s}
  .loss-val{width:60px;text-align:right;font-size:.75rem}
  .status-ok{color:#4caf50;font-weight:bold} .status-err{color:#ef5350;font-weight:bold}
  .sess-title{color:#80cbc4;font-weight:bold;margin-bottom:6px}
  .sess-event{display:flex;gap:8px;align-items:baseline;font-size:.78rem;padding:2px 0}
  .sess-num{color:#607d8b;width:36px;flex-shrink:0} .sess-id{color:#4dd0e1;flex-shrink:0}
</style>
</head>
<body>
<h1>🧠 LogGPT HDFS — Painel de Processo Interno</h1>
<p class="subtitle" id="ts">Aguardando dados...</p>

<div class="grid-2x2">
  <div class="terminal-box">
    <div class="terminal-header header-raw">
      1. Entrada Bruta (Content)
      <span class="badge" id="badge-raw" style="background:#00bcd444;color:#00bcd4">0</span>
    </div>
    <div class="terminal-content" id="term-raw"><span style="color:#607d8b">Aguardando...</span></div>
  </div>

  <div class="terminal-box">
    <div class="terminal-header header-template">
      2. Extração de Template (Drain)
      <span class="badge" id="badge-tmplt" style="background:#ffca2844;color:#ffca28">0 parsings/s</span>
    </div>
    <div class="terminal-content" id="term-tmplt"><span style="color:#607d8b">Aguardando...</span></div>
  </div>

  <div class="terminal-box">
    <div class="terminal-header header-detect">
      3. Detecção (Cross-Entropy Loss)
      <span class="badge" id="badge-model" style="background:#4caf5044;color:#4caf50">0 forward/s</span>
    </div>
    <div class="terminal-content" id="term-model"><span style="color:#607d8b">Aguardando...</span></div>
  </div>

  <div class="terminal-box">
    <div class="terminal-header header-session">
      4. Buffer de Sessões (BlockId)
      <span class="badge" id="badge-sess" style="background:#80cbc444;color:#80cbc4">0 ativas</span>
    </div>
    <div class="terminal-content" id="term-sess"><span style="color:#607d8b">Aguardando...</span></div>
  </div>
</div>

<script>
function escapeHtml(s) {
    if(!s) return '';
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function update(){
  fetch('/api/process').then(r=>r.json()).then(d=>{
    document.getElementById('ts').textContent = 'Lote: ' + (d.current_timestamp||'...') + ' (' + (d.speed_mode||'').toUpperCase() + ')';

    // 1. Raw logs
    if(d.raw_feed && d.raw_feed.length) {
      try {
      document.getElementById('badge-raw').textContent = '+' + d.raw_feed.length;
      let html = '';
      d.raw_feed.forEach(l => {
        let ts = String(l.t||''); let sid = String(l.sid||'');
        let msg = escapeHtml(l.c);
        html += '<div class="log-line"><span class="ts">['+ts.substring(11,23)+']</span> <span class="sid">['+sid.substring(0,16)+']</span> <span class="msg">' + msg + '</span></div>';
      });
      let el = document.getElementById('term-raw');
      el.innerHTML = html; el.scrollTop = el.scrollHeight;
      } catch(e){ console.error('t1',e); }
    }

    // 2. Templates
    if(d.template_feed && d.template_feed.length) {
      try {
      document.getElementById('badge-tmplt').textContent = d.template_feed.length + ' parsed';
      let html = '';
      d.template_feed.forEach(l => {
        let ts = String(l.t||''); let sid = String(l.sid||''); let eid = String(l.e||'');
        let tmplt = escapeHtml(l.tmpl).replace(/&lt;\*&gt;/g, '<span class="wild">&lt;*&gt;</span>');
        html += '<div class="log-line"><span class="ts">['+ts.substring(11,23)+']</span> <span class="sid">['+sid.substring(0,16)+']</span> <span class="eid">['+eid+']</span> <span class="tmplt">' + tmplt + '</span></div>';
      });
      let el = document.getElementById('term-tmplt');
      el.innerHTML = html; el.scrollTop = el.scrollHeight;
      } catch(e){ console.error('t2',e); }
    }

    // 3. Detection (Cross-Entropy)
    if(d.process_feed && d.process_feed.length) {
      try {
      document.getElementById('badge-model').textContent = d.process_feed.length + ' forward(s)';
      let p = d.process_feed[d.process_feed.length-1];
      let tid3 = String(p.session_id||'?');
      let html = '<div class="pipe-title">━━━ Sessão: '+tid3.substring(0,20)+' | Evento #'+p.step+' ━━━</div>';
      html += '<div class="pipe-info">Token Atual: <span class="pipe-val">'+escapeHtml(p.event_id)+'</span></div>';
      html += '<div class="pipe-info">Context Size: <span class="pipe-val">'+p.seq_len+' tokens</span></div>';
      html += '<div class="pipe-info">Threshold: <span class="pipe-val">'+p.threshold.toFixed(3)+'</span></div>';
      html += '<div class="pipe-title">Cross-Entropy Loss (por token):</div>';
      let lossPct = Math.min((p.loss / (p.threshold * 2)) * 100, 100);
      let lossColor = p.loss > p.threshold ? '#ef5350' : (p.loss > p.threshold * 0.7 ? '#ffca28' : '#4caf50');
      html += '<div class="loss-bar-wrap"><div class="loss-label">Loss</div><div class="loss-bar"><div class="loss-fill" style="width:'+lossPct+'%;background:'+lossColor+'"></div></div><div class="loss-val" style="color:'+lossColor+'">'+p.loss.toFixed(4)+'</div></div>';
      html += '<div class="loss-bar-wrap"><div class="loss-label">Threshold</div><div class="loss-bar"><div class="loss-fill" style="width:50%;background:#607d8b"></div></div><div class="loss-val" style="color:#aaa">'+p.threshold.toFixed(4)+'</div></div>';
      let resIcon = p.is_anomaly ? '<span class="status-err">🚨 ANOMALIA (Loss > Threshold)</span>' : '<span class="status-ok">🟢 NORMAL (Loss ≤ Threshold)</span>';
      html += '<div style="margin-top:12px">Resultado: '+resIcon+'</div>';
      html += '<div style="margin-top:6px;color:#607d8b">P(anomaly) = softmax(loss vs threshold)</div>';
      document.getElementById('term-model').innerHTML = html;
      } catch(e){ console.error('t3',e); }
    }

    // 4. Session context
    if(d.session_context) {
      try {
      let sc = d.session_context;
      document.getElementById('badge-sess').textContent = d.sessions_active + ' ativas';
      let tid4 = String(sc.session_id||'?');
      let html = '<div class="sess-title">BlockId: ['+tid4.substring(0,30)+']</div>';
      html += '<div style="margin-bottom:8px;color:#aaa">Últimos eventos ('+sc.total_events+' total):</div>';
      let evts = sc.events || [];
      evts.slice(-20).forEach((e, i) => {
        let isLast = (i === evts.slice(-20).length - 1);
        let arrow = isLast ? '<span style="color:#4dd0e1">>></span>' : '  ';
        let hl = isLast ? 'style="color:#fff;font-weight:bold"' : '';
        let et = String(e.t||'');
        html += '<div class="sess-event" '+hl+'>';
        html += '<span class="sess-num">#'+(sc.total_events - evts.slice(-20).length + i + 1)+'</span>';
        html += '<span class="ts">['+et.substring(11,19)+']</span>';
        html += '<span class="sess-id">'+String(e.id||'')+'</span>';
        html += '<span>'+arrow+' '+escapeHtml(e.tmpl).substring(0,60)+'</span>';
        html += '</div>';
      });
      let el = document.getElementById('term-sess');
      el.innerHTML = html; el.scrollTop = el.scrollHeight;
      } catch(e){ console.error('t4',e); }
    }

    if(d.finished) return;
    setTimeout(update, Math.max(200, d.speed_mode==='turbo'?100:400));
  }).catch(function(err){console.error('fetch err',err);setTimeout(update,1000)});
}
update();
</script>
</body>
</html>
"""

# ─── Web Servers ──────────────────────────────────────────────────────────────
class MetricsDashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = {k: v for k, v in STATE.items() if k not in ("raw_feed", "template_feed", "process_feed", "session_context")}
            self.wfile.write(json.dumps(data, default=str).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_METRICS_HTML.encode())


class ProcessDashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/process":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = {
                "current_timestamp": STATE.get("current_timestamp"),
                "speed_mode": STATE.get("speed_mode"),
                "sessions_active": STATE.get("sessions_active", 0),
                "raw_feed": STATE.get("raw_feed", []),
                "template_feed": STATE.get("template_feed", []),
                "process_feed": STATE.get("process_feed", []),
                "session_context": STATE.get("session_context"),
                "finished": STATE.get("finished", False),
            }
            self.wfile.write(json.dumps(data, default=str).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_PROCESS_HTML.encode())


def start_web_servers(port_metrics=PORT_METRICS, port_process=PORT_PROCESS):
    server1 = HTTPServer(("0.0.0.0", port_metrics), MetricsDashboardHandler)
    t1 = threading.Thread(target=server1.serve_forever, daemon=True)
    t1.start()
    server2 = HTTPServer(("0.0.0.0", port_process), ProcessDashboardHandler)
    t2 = threading.Thread(target=server2.serve_forever, daemon=True)
    t2.start()
    return server1, server2


# ─── Session Buffer ───────────────────────────────────────────────────────────
class SessionBuffer:
    def __init__(self, session_id):
        self.session_id = session_id
        self.true_label = None
        self.events = []          # EventTemplate tokens
        self.timestamps = []
        self.full_events = []     # {id, t, tmpl, raw}
        self.context_ids = []     # Accumulated token IDs for streaming inference
        self.is_alerted = False
        self.alert_timestamp = None
        self.alert_step = None
        self.first_error_ts = None
        self.lead_time_min = None
        self.status = "monitoring"

    def add_event(self, event_template, timestamp, anom_label, raw_log=""):
        self.events.append(str(event_template or ""))
        self.timestamps.append(timestamp)
        self.full_events.append({
            "id": str(event_template or ""),
            "t": str(timestamp),
            "tmpl": str(event_template or ""),
            "raw": str(raw_log or ""),
        })
        if anom_label == 1:
            self.true_label = 1
            if self.first_error_ts is None:
                self.first_error_ts = timestamp
        elif self.true_label is None:
            self.true_label = 0

    def mark_alert(self, step_idx=None):
        if not self.is_alerted:
            self.is_alerted = True
            self.alert_step = step_idx if step_idx is not None else len(self.events) - 1
            idx = min(self.alert_step, len(self.timestamps) - 1)
            self.alert_timestamp = self.timestamps[idx]
            self.status = "alerted"
            if self.first_error_ts is not None:
                try:
                    alert_ts = pd.to_datetime(self.alert_timestamp)
                    error_ts = pd.to_datetime(self.first_error_ts)
                    self.lead_time_min = (error_ts - alert_ts).total_seconds() / 60.0
                except Exception:
                    pass

    def finalize(self):
        if self.is_alerted and self.first_error_ts and self.alert_timestamp:
            try:
                alert_ts = pd.to_datetime(self.alert_timestamp)
                error_ts = pd.to_datetime(self.first_error_ts)
                self.lead_time_min = (error_ts - alert_ts).total_seconds() / 60.0
            except Exception:
                pass
        if not self.is_alerted:
            self.status = "normal"


# ─── Simulation Engine ────────────────────────────────────────────────────────
def run_simulation(model, tokenizer, test_df, session_expected, speed_mode, threshold):
    global STATE

    sessions = {}
    completed = []
    alerts_list = []
    session_received = defaultdict(int)
    start_time = time.time()
    total_logs = len(test_df)

    MAX_CONTEXT = model.config.block_size

    STATE["status"] = "running"
    STATE["total_logs"] = total_logs
    STATE["speed_mode"] = speed_mode
    STATE["raw_feed"] = []
    STATE["template_feed"] = []
    STATE["process_feed"] = []

    # Group by exact timestamp (batch = all logs at same instant)
    test_df = test_df.with_columns(
        pl.col(TIMESTAMP_COL).cast(pl.Utf8).alias("exact_ts")
    )
    unique_timestamps = test_df["exact_ts"].unique(maintain_order=True).to_list()

    total_processed = 0
    prev_ts_dt = None

    for exact_ts in unique_timestamps:
        batch_df = test_df.filter(pl.col("exact_ts") == exact_ts)
        batch_rows = list(batch_df.iter_rows(named=True))

        # ── Speed control ────────────────────────────────────────────────────
        if speed_mode == "demo":
            time.sleep(0.05)
        elif speed_mode == "realtime" and prev_ts_dt is not None:
            try:
                curr_ts_dt = pd.to_datetime(exact_ts)
                delta = (curr_ts_dt - prev_ts_dt).total_seconds()
                if 0 < delta < 5:
                    time.sleep(delta)
                prev_ts_dt = curr_ts_dt
            except Exception:
                pass
        elif speed_mode == "realtime" and prev_ts_dt is None:
            try:
                prev_ts_dt = pd.to_datetime(exact_ts)
            except Exception:
                pass

        raw_feed_batch = []
        tmpl_feed_batch = []
        process_feed_batch = []

        for row in batch_rows:
            sid = row[SESSION_ID_COL]
            event_template = str(row.get(TEMPLATE_COL) or "")
            anom_label = row.get(LABEL_COL, 0)
            raw_log = str(row.get("Content") or event_template)  # HDFS may not have Content

            if sid not in sessions:
                sessions[sid] = SessionBuffer(sid)

            buf = sessions[sid]
            buf.add_event(event_template, exact_ts, anom_label, raw_log)

            raw_feed_batch.append({"t": str(exact_ts), "sid": str(sid), "c": raw_log})
            tmpl_feed_batch.append({"t": str(exact_ts), "sid": str(sid), "e": str(event_template), "tmpl": str(event_template)})

            # ── Streaming inference ──────────────────────────────────────────
            step = len(buf.events) - 1
            if step < SKIP_START_LOGS:
                # Warm-up: just accumulate tokens
                new_ids = tokenizer.encode(" \n " + event_template if step > 0 else event_template)
                buf.context_ids.extend(new_ids)
                if len(buf.context_ids) > MAX_CONTEXT:
                    buf.context_ids = buf.context_ids[-MAX_CONTEXT:]
            elif not buf.is_alerted:
                text = " \n " + event_template
                new_ids = tokenizer.encode(text)
                full_seq = buf.context_ids + new_ids
                if len(full_seq) > MAX_CONTEXT:
                    input_seq = full_seq[-MAX_CONTEXT:]
                    target_start = len(input_seq) - len(new_ids)
                else:
                    input_seq = full_seq
                    target_start = len(buf.context_ids)

                loss_val = 0.0
                if len(input_seq) >= 2 and target_start > 0:
                    x = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits, _ = model(x)
                    logit_indices = list(range(target_start - 1, len(input_seq) - 1))
                    target_ids = input_seq[target_start:]
                    if logit_indices and target_ids:
                        try:
                            rel_logits = logits[0, logit_indices, :]
                            rel_targets = torch.tensor(target_ids, dtype=torch.long).to(DEVICE)
                            if rel_logits.shape[0] == rel_targets.shape[0]:
                                loss_val = F.cross_entropy(rel_logits, rel_targets).item()
                        except Exception:
                            pass

                    is_anomaly = loss_val > threshold

                    # Top-K for display
                    last_pred = logits[0, -2, :]
                    probs = torch.softmax(last_pred, dim=-1)
                    top_probs, topk_idx = torch.topk(probs, K)
                    topk_info = [
                        {"id": tokenizer.decode([i]).strip(), "prob": float(p)}
                        for p, i in zip(top_probs.tolist(), topk_idx.tolist())
                    ]

                    process_feed_batch.append({
                        "session_id": str(sid),
                        "step": step,
                        "event_id": str(event_template)[:30],
                        "seq_len": len(input_seq),
                        "loss": float(loss_val),
                        "threshold": float(threshold),
                        "topk": topk_info,
                        "is_anomaly": bool(is_anomaly),
                    })

                    if is_anomaly:
                        buf.mark_alert(step)
                        alerts_list.append({
                            "session_id": str(sid),
                            "alert_time": str(exact_ts),
                            "step": int(step),
                            "loss": float(loss_val),
                            "lead_time": safe_float(buf.lead_time_min),
                            "is_tp": None,
                        })

                buf.context_ids.extend(new_ids)
                if len(buf.context_ids) > MAX_CONTEXT:
                    buf.context_ids = buf.context_ids[-MAX_CONTEXT:]

            # ── Track session completion ─────────────────────────────────────
            session_received[sid] += 1
            if session_received[sid] >= session_expected.get(sid, float("inf")):
                if sid not in [s.session_id for s in completed]:
                    buf.finalize()
                    true_label = buf.true_label or 0
                    pred_label = 1 if buf.is_alerted else 0

                    if true_label == 1 and pred_label == 1:   STATE["tp"] += 1
                    elif true_label == 0 and pred_label == 1: STATE["fp"] += 1
                    elif true_label == 1 and pred_label == 0: STATE["fn"] += 1
                    else:                                      STATE["tn"] += 1

                    for a in alerts_list:
                        if a["session_id"] == str(sid):
                            a["is_tp"] = (true_label == 1)
                            a["lead_time"] = buf.lead_time_min
                            break

                    completed.append(buf)

            total_processed += 1

        # ── Update shared state ───────────────────────────────────────────────
        tp, tn, fp, fn = STATE["tp"], STATE["tn"], STATE["fp"], STATE["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        rec  = tp / (tp + fn) if (tp + fn) > 0 else None
        f1v  = 2 * prec * rec / (prec + rec) if (prec and rec) else None

        # Find the most active session for session context panel
        sc_data = None
        active_sessions = [s for s in sessions.values() if not s in completed]
        if active_sessions:
            biggest = max(active_sessions, key=lambda s: len(s.events))
            evts_for_display = [{"id": e["id"], "t": e["t"], "tmpl": e["tmpl"]} for e in biggest.full_events]
            sc_data = {
                "session_id": str(biggest.session_id),
                "total_events": len(biggest.events),
                "events": evts_for_display[-20:],
            }

        STATE.update({
            "current_timestamp": str(exact_ts),
            "logs_processed": total_processed,
            "sessions_active": len(sessions) - len(completed),
            "sessions_completed": len(completed),
            "total_alerts": len(alerts_list),
            "precision": safe_float(prec),
            "recall": safe_float(rec),
            "f1": safe_float(f1v),
            "alerts": alerts_list[-50:],
            "raw_feed": raw_feed_batch,
            "template_feed": tmpl_feed_batch,
            "process_feed": process_feed_batch,
            "session_context": sc_data,
        })

    # ── Final metrics ─────────────────────────────────────────────────────────
    tp, tn, fp, fn = STATE["tp"], STATE["tn"], STATE["fp"], STATE["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1v  = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    lead_times = [a["lead_time"] for a in alerts_list if a.get("lead_time") and a["lead_time"] > 0]
    lt_mean = sum(lead_times) / len(lead_times) if lead_times else None

    STATE.update({
        "status": "finished",
        "finished": True,
        "precision": prec, "recall": rec, "f1": f1v,
        "accuracy": (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0,
        "sessions_active": 0,
        "sessions_completed": len(completed),
        "alerts": alerts_list,
        "lead_time_mean": lt_mean,
        "elapsed": time.time() - start_time,
    })

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hdfs_realtime_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(STATE, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ Simulação HDFS concluída! Resultados: {out_path}")
    print(f"   Dashboards ativos. Ctrl+C para encerrar.")
    print(f"\n📊 Métricas Finais:")
    print(f"   Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1v:.4f}")
    print(f"   TP={tp} | FP={fp} | FN={fn} | TN={tn}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LogGPT HDFS Real-Time Simulator")
    speed = parser.add_mutually_exclusive_group()
    speed.add_argument("--turbo",   action="store_true", help="Sem delay")
    speed.add_argument("--demo",    action="store_true", help="50ms entre lotes")
    speed.add_argument("--realtime",action="store_true", help="Proporcional ao timestamp real (default)")
    parser.add_argument("--port",   type=int, default=PORT_METRICS, help="Porta do dashboard de métricas")
    args = parser.parse_args()

    speed_mode = "turbo" if args.turbo else ("demo" if args.demo else "realtime")

    print("🧠 LogGPT HDFS Real-Time Simulator — Initializing...")
    set_seeds()

    # ── Load Model ─────────────────────────────────────────────────────────
    print("   Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    config_path = os.path.join(MODEL_DIR, "config.pt")
    model_path  = os.path.join(MODEL_DIR, "hdfs_loggpt.pt")

    cfg = torch.load(config_path, weights_only=False)
    if not hasattr(cfg, "dropout"):
        cfg.dropout = 0.0
    model = LogGPT(cfg)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.to(DEVICE)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   ✅ Modelo HDFS carregado ({params:.1f}M params) em {DEVICE}")

    # ── Adaptive Threshold ─────────────────────────────────────────────────
    threshold = DEFAULT_THRESHOLD
    thresh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threshold_config.json")
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            conf = json.load(f)
            threshold = conf.get("threshold", threshold)
        print(f"   ✅ Threshold adaptativo: {threshold:.4f}")
    else:
        print(f"   ⚠️  Threshold padrão: {threshold}")

    # ── Prepare Test Data ───────────────────────────────────────────────────
    print("   Preparing test data (same split as detect.py)...")
    df = pl.read_csv(str(DATA_FILE), infer_schema_length=INFER_SCHEMA_LENGTH)
    print(f"📦 Loaded {len(df)} rows | Columns: {df.columns}")

    normal_ids = df.filter(pl.col(LABEL_COL) == 0)[SESSION_ID_COL].unique().to_list()
    anom_ids   = df.filter(pl.col(LABEL_COL) == 1)[SESSION_ID_COL].unique().to_list()

    _, test_val_ids  = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    _, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)

    test_ids_set = set(test_norm_ids + anom_ids)
    test_df = df.filter(pl.col(SESSION_ID_COL).is_in(list(test_ids_set))).sort([TIMESTAMP_COL])

    print(f"   ✅ Test set: {len(test_df)} logs | {len(test_norm_ids)} sessões normais + {len(anom_ids)} anômalas")
    print(f"   Speed mode: {speed_mode.upper()}")

    session_expected = {}
    for grp in test_df.group_by(SESSION_ID_COL):
        sid_val = grp[0][0]
        cnt = len(grp[1])
        session_expected[sid_val] = cnt

    # ── Start Servers ───────────────────────────────────────────────────────
    port_m = args.port
    port_p = args.port + 1
    server1, server2 = start_web_servers(port_m, port_p)
    print(f"\n🌐 Dashboard de Métricas: http://localhost:{port_m}")
    print(f"🌐 Painel de Processo:    http://localhost:{port_p}")
    print("   Abra no navegador para acompanhar a simulação!\n")

    try:
        run_simulation(model, tokenizer, test_df, session_expected, speed_mode, threshold)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Simulação interrompida.")
        server1.shutdown()
        server2.shutdown()


if __name__ == "__main__":
    main()
