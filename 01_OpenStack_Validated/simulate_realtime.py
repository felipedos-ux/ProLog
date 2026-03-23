"""
LogGPT Real-Time Anomaly Detection Simulator — Web Dashboard
=============================================================
Simulates a production environment by replaying OpenStack test logs one-by-one
in chronological order. The trained LogGPT model processes each incoming log
and issues anomaly alerts in real-time, visible via a web dashboard.

Usage:
    python simulate_realtime.py                # Default: proportional to real timestamps
    python simulate_realtime.py --turbo        # No delay (validation)
    python simulate_realtime.py --demo         # 50ms delay
    python simulate_realtime.py --port 8080    # Custom port

Then open http://localhost:5555 in your browser.
"""

import argparse
import json
import os
import sys
import time
import threading
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import torch
import polars as pl
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from model import LogGPT, GPTConfig
from dataset import load_openstack_data
from config import (
    MODEL_NAME, MODEL_DIR, DEVICE,
    BLOCK_SIZE, SKIP_START_LOGS, LOG_COLUMN,
    TEST_SIZE_NORMAL, TEST_SIZE_VAL_SPLIT,
    set_seeds
)

# ─── Constants ───────────────────────────────────────────────────────────────
K = 5
PAD_TOKEN_ID = 50256

# ─── Shared State (thread-safe via GIL for reads) ───────────────────────────
STATE = {
    "status": "initializing",
    "current_timestamp": "",
    "logs_processed": 0,
    "total_logs": 0,
    "sessions_active": 0,
    "sessions_completed": 0,
    "total_alerts": 0,
    "tp": 0, "fp": 0, "fn": 0, "tn": 0,
    "precision": 0, "recall": 0, "f1": 0, "accuracy": 0,
    "last_event": {},
    "alerts": [],
    "log_feed": [],
    "lead_time_mean": None,
    "lead_time_pct": 0,
    "elapsed": 0,
    "speed_mode": "realtime",
    "finished": False,
}
MAX_LOG_FEED = 25  # Keep last N log lines for the mini terminal

def safe_float(v):
    """Convert any numeric to plain Python float for JSON serialization."""
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

# ─── HTML Dashboards ────────────────────────────────────────────────────────
DASHBOARD_METRICS_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LogGPT — Dashboard de Métricas</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#111;color:#e0e0e0;padding:20px}
  h1{text-align:center;font-size:1.6rem;margin-bottom:6px;color:#4fc3f7}
  .subtitle{text-align:center;color:#888;font-size:.85rem;margin-bottom:20px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:20px}
  .card{background:#1a1a2e;border-radius:10px;padding:16px;text-align:center;border:1px solid #2a2a4a}
  .card .value{font-size:1.8rem;font-weight:800;margin:4px 0}
  .card .label{font-size:.7rem;text-transform:uppercase;letter-spacing:1px;color:#888}
  .green .value{color:#4caf50} .blue .value{color:#42a5f5} .red .value{color:#ef5350}
  .yellow .value{color:#ffca28} .cyan .value{color:#26c6da}
  .progress-outer{background:#1a1a2e;border-radius:8px;height:28px;margin-bottom:20px;overflow:hidden;border:1px solid #2a2a4a;position:relative}
  .progress-inner{height:100%;background:linear-gradient(90deg,#1b5e20,#4caf50);transition:width .3s;border-radius:8px}
  .progress-text{position:absolute;top:0;left:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;font-size:.8rem;font-weight:600;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.5)}
  .section{background:#1a1a2e;border-radius:10px;padding:18px;margin-bottom:16px;border:1px solid #2a2a4a}
  .section h2{font-size:1.1rem;margin-bottom:12px;color:#4fc3f7;border-bottom:1px solid #2a2a4a;padding-bottom:8px}
  table{width:100%;border-collapse:collapse;font-size:.82rem}
  th{text-align:left;padding:8px 10px;color:#4fc3f7;border-bottom:1px solid #2a2a4a;font-weight:600}
  td{padding:7px 10px;border-bottom:1px solid #1a1a2e}
  tr:hover{background:rgba(255,255,255,.03)}
  .tp{color:#4caf50} .fp{color:#ffca28}
  .lt-before{color:#4caf50} .lt-after{color:#ffca28}
  .badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:.7rem;font-weight:700}
  .badge-ok{background:rgba(76,175,80,.2);color:#4caf50}
  .badge-alert{background:rgba(239,83,80,.2);color:#ef5350}
  .badge-wait{background:rgba(66,165,245,.2);color:#42a5f5}
  .finished-banner{background:linear-gradient(135deg,#1b5e20,#2e7d32);border-radius:10px;padding:20px;text-align:center;margin-bottom:20px;border:2px solid #4caf50}
  .finished-banner h2{color:#fff;border:none;margin-bottom:4px}
  .timestamp{color:#ffca28;font-size:.9rem;text-align:center;margin-bottom:14px}
</style>
</head>
<body>
<h1>🧠 LogGPT — Dashboard de Métricas</h1>
<p class="subtitle">Visão Gerencial (Porta 5555)</p>

<div id="finished-banner" style="display:none" class="finished-banner">
  <h2>🏁 Simulação Concluída!</h2>
  <p style="color:rgba(255,255,255,.8)">Veja o relatório final no console.</p>
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
    <thead><tr><th>#</th><th>Sessão</th><th>Timestamp do Alerta</th><th>Lead Time</th><th>Resultado</th></tr></thead>
    <tbody id="alert-table"><tr><td colspan="5" style="color:#888;text-align:center">Nenhum alerta ainda</td></tr></tbody>
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

    // Alerts table
    if(d.alerts&&d.alerts.length>0){
      var html='';
      d.alerts.slice().reverse().forEach(function(a,i){
        var lt=a.lead_time;
        var ltStr='—';var ltCls='';
        if(typeof lt==='number'&&lt>0){ltStr=fmtTime(lt)+' antes';ltCls='lt-before';}
        else if(typeof lt==='number'){ltStr=fmtTime(Math.abs(lt))+' depois';ltCls='lt-after';}
        var res='<span class="badge badge-wait">…</span>';
        if(a.is_tp===true)res='<span class="tp">✅ TP</span>';
        else if(a.is_tp===false)res='<span class="fp">⚠️ FP</span>';
        html+='<tr><td>'+(d.alerts.length-i)+'</td><td>'+a.test_id.substring(0,40)+'</td><td>'+
          (a.alert_time||'').substring(0,25)+'</td><td class="'+ltCls+'">'+ltStr+'</td><td>'+res+'</td></tr>';
      });
      document.getElementById('alert-table').innerHTML=html;
    }

    if(d.finished){
      document.getElementById('finished-banner').style.display='block';
      return;// stop polling
    }
    setTimeout(update,400);
  }).catch(function(){setTimeout(update,1000)});
}
update();
</script>
</body>
</html>
"""

DASHBOARD_PROCESS_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LogGPT — Painel de Processo</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#050510;color:#e0e0e0;padding:20px;height:100vh;display:flex;flex-direction:column}
  h1{text-align:center;font-size:1.6rem;margin-bottom:6px;color:#b388ff}
  .subtitle{text-align:center;color:#888;font-size:.85rem;margin-bottom:20px}
  .grid-2x2{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:20px;flex:1;min-height:0}
  .terminal-box{background:#0a0a1a;border:1px solid #333;border-radius:10px;display:flex;flex-direction:column;overflow:hidden;box-shadow:0 4px 6px rgba(0,0,0,0.3)}
  .terminal-header{background:#1a1a2e;padding:10px 16px;font-weight:600;font-size:.9rem;border-bottom:1px solid #333;display:flex;justify-content:space-between;align-items:center}
  .header-raw{border-top:3px solid #4fc3f7;color:#4fc3f7}
  .header-template{border-top:3px solid #ffca28;color:#ffca28}
  .header-detect{border-top:3px solid #b388ff;color:#b388ff}
  .header-session{border-top:3px solid #4caf50;color:#4caf50}
  .badge{background:var(--bg);color:#fff;padding:2px 8px;border-radius:12px;font-size:.75rem}
  .terminal-content{flex:1;overflow-y:auto;padding:16px;font-family:'Consolas','Courier New',monospace;font-size:.8rem;line-height:1.6;color:#ccc}
  .terminal-content::-webkit-scrollbar{width:8px}
  .terminal-content::-webkit-scrollbar-thumb{background:#444;border-radius:4px}
  
  /* Log specific colors */
  .log-line{border-bottom:1px dashed #222;padding-bottom:6px;margin-bottom:6px}
  .ts{color:#64b5f6} .sid{color:#888} .msg{color:#fff}
  .pid{color:#e57373} .lvl{color:#81c784} .cmp{color:#ffb74d}
  .tmplt{color:#ffd54f} .wild{color:#ef5350;font-weight:bold}
  
  /* Pipeline colors */
  .pipe-title{color:#b388ff;font-weight:bold;margin-top:10px}
  .pipe-info{color:#aaa} .pipe-val{color:#fff}
  .pred-bar-wrap{display:flex;align-items:center;margin:2px 0;gap:8px}
  .pred-label{width:110px;text-align:right;color:#ccc}
  .pred-bar{height:6px;background:#333;flex:1;border-radius:3px;overflow:hidden}
  .pred-fill{height:100%;background:#b388ff}
  .pred-pct{width:50px;text-align:left;color:#888}
  .status-ok{color:#4caf50;font-weight:bold}
  .status-err{color:#ef5350;font-weight:bold}
  
  /* Session Context */
  .sess-title{color:#81c784;font-weight:bold;margin-bottom:8px}
  .sess-event{display:flex;gap:12px;margin:2px 0}
  .sess-num{color:#666;width:25px}
  .sess-id{color:#4fc3f7;width:70px}
</style>
</head>
<body>
<h1>🧠 LogGPT — Painel de Processo Interno</h1>
<p class="subtitle" id="ts">Aguardando lote de dados...</p>

<div class="grid-2x2">
  <!-- 1. Log Original -->
  <div class="terminal-box">
    <div class="terminal-header header-raw">1. Entrada Bruta (Content) <span class="badge" style="--bg:#0277bd" id="badge-raw">0 logs/s</span></div>
    <div class="terminal-content" id="term-raw">Aguardando...</div>
  </div>
  
  <!-- 2. Extração -->
  <div class="terminal-box">
    <div class="terminal-header header-template">2. Extração de Template (Drain) <span class="badge" style="--bg:#ff8f00" id="badge-tmplt">0 parsings/s</span></div>
    <div class="terminal-content" id="term-tmplt">Aguardando...</div>
  </div>
  
  <!-- 3. Modelo -->
  <div class="terminal-box">
    <div class="terminal-header header-detect">3. Inferência (Top-K) <span class="badge" style="--bg:#6200ea" id="badge-model">0 forward/s</span></div>
    <div class="terminal-content" id="term-model">Aguardando...</div>
  </div>
  
  <!-- 4. Contexto -->
  <div class="terminal-box">
    <div class="terminal-header header-session">4. Buffer de Sessões <span class="badge" style="--bg:#2e7d32" id="badge-sess">0 ativas</span></div>
    <div class="terminal-content" id="term-sess">Aguardando...</div>
  </div>
</div>

<script>
function escapeHtml(s) {
    if(!s) return '';
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function update(){
  fetch('/api/process').then(r=>r.json()).then(d=>{
    document.getElementById('ts').textContent = 'Lote Processado: ' + (d.current_timestamp||'...') + ' (' + (d.speed_mode||'').toUpperCase() + ')';
    
    // 1. Raw Logs
    if(d.raw_feed && d.raw_feed.length) {
      document.getElementById('badge-raw').textContent = '+' + d.raw_feed.length;
      let html = '';
      d.raw_feed.forEach(l => {
        let ts = String(l.t||'');
        let sid = String(l.sid||'');
        let msg = escapeHtml(l.c);
        html += '<div class="log-line"><span class="ts">['+ts.substring(11,23)+']</span> <span class="sid">['+sid.substring(0,8)+']</span> <span class="msg">' + msg + '</span></div>';
      });
      let el = document.getElementById('term-raw');
      el.innerHTML = html;
      el.scrollTop = el.scrollHeight;
    }

    // 2. Templates
    if(d.template_feed && d.template_feed.length) {
      document.getElementById('badge-tmplt').textContent = d.template_feed.length + ' parsed';
      let html = '';
      d.template_feed.forEach(l => {
        let ts = String(l.t||'');
        let sid = String(l.sid||'');
        let eid = String(l.e||'');
        let tmplt = escapeHtml(l.tmpl).replace(/&lt;\*&gt;/g, '<span class="wild">&lt;*&gt;</span>');
        html += '<div class="log-line"><span class="ts">['+ts.substring(11,23)+']</span> <span class="sid">['+sid.substring(0,8)+']</span> <span class="eid">['+eid+']</span> <span class="tmplt">' + tmplt + '</span></div>';
      });
      let el = document.getElementById('term-tmplt');
      el.innerHTML = html;
      el.scrollTop = el.scrollHeight;
    }

    // 3. Inferência Modelo
    if(d.process_feed && d.process_feed.length) {
      try {
      document.getElementById('badge-model').textContent = d.process_feed.length + ' forward(s)';
      let p = d.process_feed[d.process_feed.length-1];
      let tid3 = String(p.test_id||'?');
      let html = '<div class="pipe-title">━━━ Sessão: '+tid3.substring(0,8)+' | Evento #'+p.step+' ━━━</div>';
      html += '<div class="pipe-info">Token Atual: <span class="pipe-val">'+escapeHtml(p.event_id)+'</span></div>';
      html += '<div class="pipe-info">Context Size: <span class="pipe-val">'+p.seq_len+' tokens</span></div>';
      html += '<div class="pipe-title">Top-'+p.k+' Predições:</div>';
      
      if(p.topk) p.topk.forEach((pred, i) => {
        let pct = (pred.prob * 100).toFixed(1);
        let highlight = (String(pred.id) === String(p.event_id)) ? 'style="color:#4fc3f7;font-weight:bold"' : '';
        html += '<div class="pred-bar-wrap">';
        html += '<div class="pred-label" '+highlight+'>'+escapeHtml(pred.id)+'</div>';
        html += '<div class="pred-bar"><div class="pred-fill" style="width:'+pct+'%"></div></div>';
        html += '<div class="pred-pct">'+pct+'%</div>';
        html += '</div>';
      });
      
      let resIcon = p.is_anomaly ? '<span class="status-err">🚨 ANOMALIA (Fora do Top-K)</span>' : '<span class="status-ok">🟢 NORMAL (Acerto do Modelo)</span>';
      html += '<div style="margin-top:12px">Resultado: '+resIcon+'</div>';
      html += '<div style="margin-top:6px;color:#666">Probabilidades P(x_t | x_0...x_t-1)</div>';
      document.getElementById('term-model').innerHTML = html;
      } catch(e){ console.error('term-model err',e); }
    }

    // 4. Sessão
    if(d.session_context) {
      try {
      let sc = d.session_context;
      document.getElementById('badge-sess').textContent = d.sessions_active + ' ativas';
      let tid4 = String(sc.test_id||'?');
      let html = '<div class="sess-title">Sessão ativa: ['+tid4.substring(0,25)+']</div>';
      html += '<div style="margin-bottom:8px;color:#aaa">Últimos eventos ('+sc.total_events+' total):</div>';
      let evts = sc.events || [];
      evts.slice(-20).forEach((e, i) => {
        let isLast = (i === evts.slice(-20).length - 1);
        let arrow = isLast ? '<span style="color:#b388ff">>></span>' : '  ';
        let hl = isLast ? 'style="color:#fff;font-weight:bold"' : '';
        let et = String(e.t||'');
        html += '<div class="sess-event" '+hl+'>';
        html += '<span class="sess-num">#'+(sc.total_events - evts.slice(-20).length + i + 1)+'</span>';
        html += '<span class="ts">['+et.substring(11,19)+']</span>';
        html += '<span class="sess-id">'+String(e.id||'')+'</span>';
        html += '<span>'+arrow+' '+escapeHtml(e.tmpl).substring(0, 60)+'</span>';
        html += '</div>';
      });
      let el = document.getElementById('term-sess');
      el.innerHTML = html;
      el.scrollTop = el.scrollHeight;
      } catch(e){ console.error('term-sess err',e); }
    }

    if(d.finished) return;
    setTimeout(update,Math.max(200, d.speed_mode==='turbo'?100:400));
  }).catch(function(err){console.error('fetch err',err);setTimeout(update,1000)});
}
update();
</script>
</body>
</html>
"""


# ─── Web Servers ─────────────────────────────────────────────────────────────
class MetricsDashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(STATE, default=str).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_METRICS_HTML.encode())

class ProcessDashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/process":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            # Send specific process arrays instead of the whole STATE to avoid huge JSONs
            proc_state = {
                "current_timestamp": STATE["current_timestamp"],
                "speed_mode": STATE["speed_mode"],
                "raw_feed": STATE.get("raw_feed", []),
                "template_feed": STATE.get("template_feed", []),
                "process_feed": STATE.get("process_feed", []),
                "session_context": STATE.get("session_context", None),
                "sessions_active": STATE["sessions_active"],
                "finished": STATE["finished"]
            }
            self.wfile.write(json.dumps(proc_state, default=str).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_PROCESS_HTML.encode())


def start_web_servers(port_metrics, port_process):
    server1 = HTTPServer(("0.0.0.0", port_metrics), MetricsDashboardHandler)
    t1 = threading.Thread(target=server1.serve_forever, daemon=True)
    t1.start()
    
    server2 = HTTPServer(("0.0.0.0", port_process), ProcessDashboardHandler)
    t2 = threading.Thread(target=server2.serve_forever, daemon=True)
    t2.start()
    
    return server1, server2


# ─── Session Buffer ──────────────────────────────────────────────────────────
class SessionBuffer:
    def __init__(self, test_id):
        self.test_id = test_id
        self.true_label = None
        self.events = []
        self.timestamps = []
        self.is_alerted = False
        self.alert_timestamp = None
        self.alert_step = None
        self.first_error_ts = None
        self.lead_time_min = None
        self.status = "monitoring"

    def add_event(self, event_id, timestamp, anom_label, raw_log, template):
        self.events.append(event_id)
        self.timestamps.append(timestamp)
        # Store for process dashboard
        if not hasattr(self, 'full_events'):
            self.full_events = []
        self.full_events.append({"id": event_id, "t": timestamp, "tmpl": template, "raw": raw_log})
        
        if anom_label == 1:
            self.true_label = 1
            if self.first_error_ts is None:
                self.first_error_ts = timestamp
        elif self.true_label is None:
            self.true_label = 0

    def get_token_string(self):
        return " ".join(self.events)

    def mark_alert(self, step_idx):
        if not self.is_alerted:
            self.is_alerted = True
            self.alert_step = step_idx
            self.alert_timestamp = self.timestamps[step_idx] if step_idx < len(self.timestamps) else self.timestamps[-1]
            self.status = "alerted"
            if self.first_error_ts is not None:
                try:
                    alert_ts = pd.to_datetime(self.alert_timestamp)
                    error_ts = pd.to_datetime(self.first_error_ts)
                    self.lead_time_min = (error_ts - alert_ts).total_seconds() / 60.0
                except Exception:
                    pass

    def finalize(self):
        # Recalculate lead time with final info
        if self.is_alerted and self.first_error_ts and self.alert_timestamp:
            try:
                alert_ts = pd.to_datetime(self.alert_timestamp)
                error_ts = pd.to_datetime(self.first_error_ts)
                self.lead_time_min = (error_ts - alert_ts).total_seconds() / 60.0
            except Exception:
                pass
        if not self.is_alerted:
            self.status = "normal"


# ─── Simulation Engine ───────────────────────────────────────────────────────
def fmt_time(minutes):
    if minutes is None: return None
    if abs(minutes) < 1: return f"{minutes*60:.1f}s"
    if abs(minutes) < 60: return f"{minutes:.1f}min"
    return f"{minutes/60:.1f}h"


def run_simulation(model, tokenizer, test_df, session_expected, speed_mode):
    global STATE

    sessions = {}
    completed = []
    alerts_list = []
    session_received = defaultdict(int)
    start_time = time.time()
    total_logs = len(test_df)

    STATE["status"] = "running"
    STATE["total_logs"] = total_logs
    STATE["speed_mode"] = speed_mode
    STATE["raw_feed"] = []
    STATE["template_feed"] = []
    STATE["process_feed"] = []

    # Prepare batching by exact timestamp
    # Create a string column for grouping
    test_df = test_df.with_columns(
        pl.concat_str([pl.col("timestamp"), pl.lit(" "), pl.col("hour")]).str.strip_chars().alias("exact_ts")
    )
    
    # We iterate over unique timestamps in order
    unique_timestamps = test_df["exact_ts"].unique(maintain_order=True).to_list()
    
    total_processed = 0
    prev_ts_dt = None

    for exact_ts in unique_timestamps:
        # Get all logs for this exact microsecond/second
        batch_df = test_df.filter(pl.col("exact_ts") == exact_ts)
        batch_rows = list(batch_df.iter_rows(named=True))
        
        # ── Speed control (Delay applied ONCE per batch timestamp) ──────
        if speed_mode == "demo":
            time.sleep(0.05)
        elif speed_mode == "realtime" and prev_ts_dt is not None:
            try:
                curr_ts_dt = pd.to_datetime(exact_ts)
                delta = (curr_ts_dt - prev_ts_dt).total_seconds()
                if 0 < delta < 10:
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
        last_ev = {}

        # ── Process all logs in the batch ───────────────────────────────
        for row in batch_rows:
            tid = row["test_id"]
            event_id = row[LOG_COLUMN]
            anom_label = row["anom_label"]
            raw_log = str(row.get("Content") or "N/A")
            tmpl_log = str(row.get("EventTemplate") or "N/A")

            if tid not in sessions:
                sessions[tid] = SessionBuffer(tid)

            buf = sessions[tid]
            buf.add_event(event_id, exact_ts, anom_label, raw_log, tmpl_log)

            raw_feed_batch.append({"t": str(exact_ts), "sid": str(tid), "c": raw_log})
            tmpl_feed_batch.append({"t": str(exact_ts), "sid": str(tid), "e": str(event_id), "tmpl": tmpl_log})

            last_ev = {
                "test_id": tid,
                "event_id": event_id,
                "timestamp": exact_ts,
                "n_events": len(buf.events),
                "status": "normal",
            }

            # Inference
            if len(buf.events) <= SKIP_START_LOGS + 1:
                last_ev["status"] = "warmup"
            elif not buf.is_alerted:
                token_str = buf.get_token_string()
                tokens = tokenizer.encode(token_str, truncation=True, max_length=1024)
                if len(tokens) >= 2:
                    input_ids = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
                    with torch.no_grad():
                        logits, _ = model(input_ids)
                    last_pred = logits[0, -2, :]
                    last_actual = input_ids[0, -1].item()
                    probs = torch.softmax(last_pred, dim=-1)
                    top_probs, topk_idx = torch.topk(probs, K)
                    is_in_topk = (topk_idx == last_actual).any().item()

                    # Record details for process dashboard
                    topk_info = []
                    for prob, idx in zip(top_probs.tolist(), topk_idx.tolist()):
                        dec_token = tokenizer.decode([idx]).strip()
                        topk_info.append({"id": dec_token, "prob": prob})
                    
                    actual_token_str = tokenizer.decode([last_actual]).strip()

                    process_feed_batch.append({
                        "test_id": tid,
                        "step": len(buf.events) - 1,
                        "event_id": actual_token_str,
                        "seq_len": len(tokens),
                        "k": K,
                        "topk": topk_info,
                        "is_anomaly": not is_in_topk
                    })

                    if not is_in_topk:
                        buf.mark_alert(len(buf.events) - 1)
                        last_ev["status"] = "ANOMALY"
                        alerts_list.append({
                            "test_id": str(tid),
                            "alert_time": str(exact_ts),
                            "step": int(len(buf.events) - 1),
                            "lead_time": safe_float(buf.lead_time_min),
                            "is_tp": None,
                        })
            else:
                last_ev["status"] = "anomaly_cont"

            # ── Track session completion ─────────────────────────────────
            session_received[tid] += 1
            if session_received[tid] >= session_expected.get(tid, float("inf")):
                if tid not in [s.test_id for s in completed]:
                    buf.finalize()
                    true_label = buf.true_label or 0
                    pred_label = 1 if buf.is_alerted else 0

                    if true_label == 1 and pred_label == 1:   STATE["tp"] += 1
                    elif true_label == 0 and pred_label == 1: STATE["fp"] += 1
                    elif true_label == 1 and pred_label == 0: STATE["fn"] += 1
                    else:                                     STATE["tn"] += 1

                    for a in alerts_list:
                        if a["test_id"] == tid:
                            a["is_tp"] = (true_label == 1)
                            a["lead_time"] = buf.lead_time_min
                            break

                    completed.append(buf)

            total_processed += 1

        # ── Update shared state at the end of the batch ──────────────────
        elapsed = time.time() - start_time

        tp, fp = STATE["tp"], STATE["fp"]
        fn, tn = STATE["fn"], STATE["tn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

        lead_times = [a["lead_time"] for a in alerts_list if isinstance(a.get("lead_time"), (int, float)) and a["lead_time"] > 0]
        lt_mean = safe_float(sum(lead_times) / len(lead_times)) if lead_times else None

        clean_alerts = []
        for a in alerts_list[-50:]:
            clean_alerts.append({
                "test_id": str(a["test_id"]),
                "alert_time": str(a["alert_time"]),
                "step": int(a["step"]),
                "lead_time": safe_float(a.get("lead_time")),
                "is_tp": a.get("is_tp"), 
            })

        STATE["raw_feed"].extend(raw_feed_batch)
        if len(STATE["raw_feed"]) > 40: STATE["raw_feed"] = STATE["raw_feed"][-40:]
        
        STATE["template_feed"].extend(tmpl_feed_batch)
        if len(STATE["template_feed"]) > 40: STATE["template_feed"] = STATE["template_feed"][-40:]
        
        STATE["process_feed"].extend(process_feed_batch)
        if len(STATE["process_feed"]) > 10: STATE["process_feed"] = STATE["process_feed"][-10:]

        # Select the last modified session for context display
        sess_ctx = None
        if batch_rows:
            last_tid = batch_rows[-1]["test_id"]
            if last_tid in sessions:
                b = sessions[last_tid]
                sess_ctx = {
                    "test_id": b.test_id,
                    "total_events": len(b.full_events),
                    "events": b.full_events[-25:]  # push last 25 for rendering
                }

        STATE.update({
            "logs_processed": total_processed,
            "current_timestamp": str(exact_ts),
            "sessions_active": len(sessions) - len(completed),
            "sessions_completed": len(completed),
            "total_alerts": len(alerts_list),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1_val),
            "accuracy": float(acc),
            "last_event": last_ev if last_ev else STATE.get("last_event", {}),
            "alerts": clean_alerts,
            "session_context": sess_ctx,
            "lead_time_mean": lt_mean,
            "elapsed": float(elapsed),
        })

    # ── Finalize remaining sessions ──────────────────────────────────────
    for tid, buf in sessions.items():
        if tid not in [s.test_id for s in completed]:
            buf.finalize()
            true_label = buf.true_label or 0
            pred_label = 1 if buf.is_alerted else 0
            if true_label == 1 and pred_label == 1:   STATE["tp"] += 1
            elif true_label == 0 and pred_label == 1:  STATE["fp"] += 1
            elif true_label == 1 and pred_label == 0:  STATE["fn"] += 1
            else:                                       STATE["tn"] += 1
            for a in alerts_list:
                if a["test_id"] == tid:
                    a["is_tp"] = (true_label == 1)
                    a["lead_time"] = buf.lead_time_min
            completed.append(buf)

    # Final metrics
    tp, fp, fn, tn = STATE["tp"], STATE["fp"], STATE["fn"], STATE["tn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    lead_times = [a["lead_time"] for a in alerts_list if a.get("lead_time") is not None and a["lead_time"] > 0]
    lt_mean = sum(lead_times) / len(lead_times) if lead_times else None

    STATE.update({
        "status": "finished",
        "finished": True,
        "precision": prec,
        "recall": rec,
        "f1": f1_val,
        "accuracy": (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0,
        "sessions_active": 0,
        "sessions_completed": len(completed),
        "alerts": alerts_list,
        "lead_time_mean": lt_mean,
        "lead_time_pct": (len(lead_times) / tp * 100) if tp > 0 else 0,
        "elapsed": time.time() - start_time,
    })

    # Save JSON results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realtime_simulation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(STATE, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ Simulação concluída! Resultados salvos em: {out_path}")
    print(f"   O dashboard web continua ativo. Pressione Ctrl+C para encerrar.")


def main():
    parser = argparse.ArgumentParser(description="LogGPT Real-Time Simulator")
    speed = parser.add_mutually_exclusive_group()
    speed.add_argument("--turbo", action="store_true", help="No delay (fastest)")
    speed.add_argument("--demo", action="store_true", help="50ms delay per log")
    speed.add_argument("--realtime", action="store_true", help="Proportional to real timestamps (default)")
    parser.add_argument("--port", type=int, default=5555, help="Web dashboard port (default: 5555)")
    args = parser.parse_args()

    if args.turbo:
        speed_mode = "turbo"
    elif args.demo:
        speed_mode = "demo"
    else:
        speed_mode = "realtime"  # Default

    print("🧠 LogGPT Real-Time Simulator — Initializing...")
    set_seeds()

    # ── Load Model ───────────────────────────────────────────────────────
    print("   Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    config = torch.load(f"{MODEL_DIR}/config.pt", weights_only=False)
    if not hasattr(config, 'dropout'):
        config.dropout = 0.0

    model = LogGPT(config)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/loggpt_weights.pt", weights_only=False))
    model.to(DEVICE)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   ✅ Model loaded ({params:.1f}M params) on {DEVICE}")

    # ── Prepare Test Data ────────────────────────────────────────────────
    print("   Preparing test data (same split as detect_custom.py)...")
    df = load_openstack_data()  # Already reads full CSV with Content, EventTemplate, etc.

    normal_ids = df.filter(pl.col("anom_label") == 0)["test_id"].unique().to_list()
    anom_ids = df.filter(pl.col("anom_label") == 1)["test_id"].unique().to_list()

    _, test_val_ids = train_test_split(normal_ids, test_size=TEST_SIZE_NORMAL, random_state=42)
    _, test_norm_ids = train_test_split(test_val_ids, test_size=TEST_SIZE_VAL_SPLIT, random_state=42)

    test_ids_set = set(test_norm_ids + anom_ids)
    test_df = df.filter(pl.col("test_id").is_in(list(test_ids_set))).sort(["timestamp", "hour"])

    session_expected = {}
    for tid, cnt in zip(*test_df.group_by("test_id").len().get_columns()):
        session_expected[tid] = cnt

    total_logs = len(test_df)
    print(f"   ✅ Test set: {total_logs:,} logs | {len(test_norm_ids)} normal + {len(anom_ids)} anomalous sessions")
    print(f"   Speed mode: {speed_mode.upper()}")

    # ── Start Web Servers ────────────────────────────────────────────────
    server_metrics, server_process = start_web_servers(args.port, args.port + 1)
    print(f"\n🌐 Dashboard de Métricas: http://localhost:{args.port}")
    print(f"🌐 Painel de Processo:    http://localhost:{args.port + 1}")
    print(f"   Abra no navegador para acompanhar a simulação!\n")

    STATE["total_logs"] = total_logs
    STATE["speed_mode"] = speed_mode
    STATE["status"] = "ready"

    # ── Run Simulation in Main Thread ────────────────────────────────────
    run_simulation(model, tokenizer, test_df, session_expected, speed_mode)

    # Keep server alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Encerrando...")
        server_metrics.shutdown()
        server_process.shutdown()


if __name__ == "__main__":
    main()
