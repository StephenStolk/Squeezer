"""
ContextSqueezer Analytics Dashboard

FastAPI application that exposes aggregated telemetry from the local SQLite
store at http://127.0.0.1:8788.

Endpoints
---------
GET  /api/summary      – overall aggregated stats
GET  /api/timeline     – per-request timeline for charting
GET  /api/algo         – per-algorithm token savings breakdown
GET  /api/ccr          – CCR store statistics
GET  /api/pii          – PII interception audit log
GET  /                 – Serve the dashboard SPA (index.html)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Inline HTML dashboard (single file, no external build step)
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ContextSqueezer Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0f1117; --surface: #1a1d2e; --border: #2d3148;
      --accent: #7c6aff; --green: #22d3a6; --red: #f55; --text: #e2e8f0;
      --muted: #64748b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', system-ui, sans-serif; background: var(--bg);
           color: var(--text); min-height: 100vh; }
    header { padding: 18px 32px; border-bottom: 1px solid var(--border);
             display: flex; align-items: center; gap: 12px; }
    header .logo { font-size: 1.3rem; font-weight: 700; color: var(--accent); }
    header .sub  { color: var(--muted); font-size: .85rem; }
    .badge { background: var(--green); color: #000; padding: 2px 10px;
             border-radius: 12px; font-size: .75rem; font-weight: 600; }
    main { padding: 28px 32px; max-width: 1400px; margin: 0 auto; }
    .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
    .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 24px; }
    .card { background: var(--surface); border: 1px solid var(--border);
            border-radius: 12px; padding: 20px; }
    .card h3 { font-size: .75rem; color: var(--muted); text-transform: uppercase;
               letter-spacing: .05em; margin-bottom: 8px; }
    .stat-val { font-size: 2rem; font-weight: 700; color: var(--accent); }
    .stat-sub { font-size: .8rem; color: var(--muted); margin-top: 4px; }
    .chart-wrap { position: relative; height: 260px; }
    .section-title { font-size: .95rem; font-weight: 600; margin-bottom: 14px;
                     color: var(--text); }
    table { width: 100%; border-collapse: collapse; font-size: .82rem; }
    th, td { padding: 8px 12px; border-bottom: 1px solid var(--border); text-align: left; }
    th { color: var(--muted); font-weight: 500; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 6px;
            font-size: .72rem; font-weight: 600; background: var(--border); }
    .pill.green { background: rgba(34,211,166,.15); color: var(--green); }
    .pill.red   { background: rgba(255,85,85,.15);  color: var(--red); }
    footer { text-align: center; padding: 24px; color: var(--muted); font-size: .75rem;
             border-top: 1px solid var(--border); margin-top: 24px; }
    @media (max-width: 900px) { .grid-4 { grid-template-columns: repeat(2,1fr); }
                                 .grid-2 { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<header>
  <span class="logo">⚡ ContextSqueezer</span>
  <span class="sub">Local Analytics Dashboard</span>
  <span class="badge" id="status-badge">Loading…</span>
</header>
<main>
  <!-- KPI row -->
  <div class="grid-4">
    <div class="card">
      <h3>Requests Processed</h3>
      <div class="stat-val" id="kpi-requests">—</div>
    </div>
    <div class="card">
      <h3>Tokens Saved</h3>
      <div class="stat-val" id="kpi-saved">—</div>
      <div class="stat-sub" id="kpi-pct">—</div>
    </div>
    <div class="card">
      <h3>Avg Proxy Latency</h3>
      <div class="stat-val" id="kpi-latency">—</div>
      <div class="stat-sub">ms overhead</div>
    </div>
    <div class="card">
      <h3>Cache Hits</h3>
      <div class="stat-val" id="kpi-cache">—</div>
      <div class="stat-sub" id="kpi-ccr">—</div>
    </div>
  </div>

  <!-- Timeline chart -->
  <div class="grid-2">
    <div class="card">
      <div class="section-title">Token Timeline</div>
      <div class="chart-wrap"><canvas id="timeline-chart"></canvas></div>
    </div>
    <div class="card">
      <div class="section-title">Algorithm Breakdown</div>
      <div class="chart-wrap"><canvas id="algo-chart"></canvas></div>
    </div>
  </div>

  <!-- Tables -->
  <div class="grid-2">
    <div class="card">
      <div class="section-title">Algorithm Savings Detail</div>
      <table id="algo-table">
        <thead><tr><th>Algorithm</th><th>Tokens Saved</th><th>Share</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card">
      <div class="section-title">PII Interception Log</div>
      <table id="pii-table">
        <thead><tr><th>Pattern</th><th>Occurrences</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <!-- Per-component breakdown (multi-agent / component-proxy mode) -->
  <div class="card" id="component-card" style="margin-bottom:24px; display:none;">
    <div class="section-title">Per-Component Breakdown <span style="color:var(--muted); font-weight:400;">(multi-agent / component-proxy mode)</span></div>
    <table id="component-table">
      <thead><tr><th>Component</th><th>Requests</th><th>Raw tok</th><th>Compressed tok</th><th>Saved</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</main>
<footer>ContextSqueezer · All data local · Zero external telemetry</footer>

<script>
const API = '';
const PALETTE = ['#7c6aff','#22d3a6','#f59e0b','#f55','#38bdf8','#a78bfa','#fb7185','#34d399'];

async function load() {
  try {
    const s = await fetch(`${API}/api/summary`).then(r => r.json());
    document.getElementById('status-badge').textContent = 'Live';
    document.getElementById('kpi-requests').textContent = s.total_requests.toLocaleString();
    document.getElementById('kpi-saved').textContent = fmtK(s.total_tokens_saved);
    document.getElementById('kpi-pct').textContent = `${s.compression_ratio_pct}% compression`;
    document.getElementById('kpi-latency').textContent = s.avg_proxy_latency_ms.toFixed(1);
    document.getElementById('kpi-cache').textContent = s.cache_hits.toLocaleString();
    document.getElementById('kpi-ccr').textContent = `${s.ccr_fetches} CCR fetches`;

    renderTimeline(s.timeline || []);
    renderAlgo(s.algo_breakdown || {});
    renderAlgoTable(s.algo_breakdown || {});
    renderComponentTable(s.component_breakdown || []);
  } catch(e) {
    document.getElementById('status-badge').textContent = 'Offline';
  }

  try {
    const pii = await fetch(`${API}/api/pii`).then(r => r.json());
    renderPiiTable(pii.items || []);
  } catch(_) {}
}

function fmtK(n) {
  if (n >= 1e6) return (n/1e6).toFixed(1)+'M';
  if (n >= 1e3) return (n/1e3).toFixed(1)+'K';
  return n.toLocaleString();
}

let timelineChart, algoChart;

function renderTimeline(data) {
  const labels = data.map((_, i) => `#${i+1}`);
  const raw = data.map(d => d.raw);
  const comp = data.map(d => d.compressed);
  const ctx = document.getElementById('timeline-chart').getContext('2d');
  if (timelineChart) timelineChart.destroy();
  timelineChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Raw tokens', data: raw, borderColor: '#f55', backgroundColor: 'rgba(255,85,85,.1)', tension: .3, fill: true },
        { label: 'Compressed', data: comp, borderColor: '#22d3a6', backgroundColor: 'rgba(34,211,166,.1)', tension: .3, fill: true }
      ]
    },
    options: { responsive: true, maintainAspectRatio: false,
               plugins: { legend: { labels: { color: '#e2e8f0' } } },
               scales: { x: { ticks: { color: '#64748b' }, grid: { color: '#2d3148' } },
                         y: { ticks: { color: '#64748b' }, grid: { color: '#2d3148' } } } }
  });
}

function renderAlgo(breakdown) {
  const labels = Object.keys(breakdown);
  const values = Object.values(breakdown);
  const ctx = document.getElementById('algo-chart').getContext('2d');
  if (algoChart) algoChart.destroy();
  if (!labels.length) return;
  algoChart = new Chart(ctx, {
    type: 'doughnut',
    data: { labels, datasets: [{ data: values, backgroundColor: PALETTE, borderWidth: 0 }] },
    options: { responsive: true, maintainAspectRatio: false,
               plugins: { legend: { position: 'right', labels: { color: '#e2e8f0', boxWidth: 12 } } } }
  });
}

function renderAlgoTable(breakdown) {
  const total = Object.values(breakdown).reduce((a,b)=>a+b, 0) || 1;
  const rows = Object.entries(breakdown).sort((a,b)=>b[1]-a[1]);
  const tbody = document.querySelector('#algo-table tbody');
  tbody.innerHTML = rows.map(([k,v]) => `
    <tr>
      <td>${k.replace(/_/g,' ')}</td>
      <td>${fmtK(v)}</td>
      <td><span class="pill green">${(v/total*100).toFixed(1)}%</span></td>
    </tr>`).join('');
}

function renderPiiTable(items) {
  const tbody = document.querySelector('#pii-table tbody');
  if (!items.length) { tbody.innerHTML = '<tr><td colspan="2" style="color:#64748b">No PII detected</td></tr>'; return; }
  tbody.innerHTML = items.map(i => `
    <tr>
      <td><span class="pill red">${i.pattern}</span></td>
      <td>${i.count}</td>
    </tr>`).join('');
}

function renderComponentTable(items) {
  const card = document.getElementById('component-card');
  if (!items.length) { card.style.display = 'none'; return; }
  card.style.display = 'block';
  const tbody = document.querySelector('#component-table tbody');
  tbody.innerHTML = items.map(c => `
    <tr>
      <td><span class="pill green">${c.component_id}</span></td>
      <td>${c.requests}</td>
      <td>${fmtK(c.raw_tokens)}</td>
      <td>${fmtK(c.compressed_tokens)}</td>
      <td>${fmtK(c.tokens_saved)}</td>
    </tr>`).join('');
}

load();
setInterval(load, 8000);
</script>
</body>
</html>
"""


async def run_dashboard(settings: "Settings") -> None:  # type: ignore[name-defined]
    """Start the FastAPI analytics dashboard."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
        import uvicorn
        from contextsqueezer.storage.sqlite_store import Store
        import sqlite3
    except ImportError as e:
        log.warning("Dashboard dependencies missing (%s) – dashboard disabled.", e)
        return

    from contextsqueezer.config import Settings as _Settings

    app = FastAPI(title="ContextSqueezer Dashboard", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return _DASHBOARD_HTML

    @app.get("/api/summary")
    async def summary():
        async with Store(settings.db_path) as store:
            data = await store.dashboard_summary()
        return JSONResponse(data)

    @app.get("/api/timeline")
    async def timeline():
        async with Store(settings.db_path) as store:
            data = await store.dashboard_summary()
        return JSONResponse({"timeline": data.get("timeline", [])})

    @app.get("/api/algo")
    async def algo():
        async with Store(settings.db_path) as store:
            data = await store.dashboard_summary()
        return JSONResponse({"breakdown": data.get("algo_breakdown", {})})

    @app.get("/api/pii")
    async def pii():
        import aiosqlite
        items: list[dict] = []
        try:
            async with aiosqlite.connect(settings.db_path) as db:
                db.row_factory = aiosqlite.Row
                rows = await (
                    await db.execute(
                        "SELECT pattern, SUM(count) as count FROM pii_log GROUP BY pattern ORDER BY count DESC"
                    )
                ).fetchall()
                items = [{"pattern": r["pattern"], "count": r["count"]} for r in rows]
        except Exception:
            pass
        return JSONResponse({"items": items})

    @app.get("/api/ccr")
    async def ccr():
        async with Store(settings.db_path) as store:
            count = await store.ccr_count()
        return JSONResponse({"total_entries": count})

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=settings.dashboard_port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    log.info("Dashboard at http://127.0.0.1:%d", settings.dashboard_port)
    await server.serve()
