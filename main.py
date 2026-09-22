import os
import tempfile
import uuid
import json
import threading
import queue
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from backend.main import _build_pipeline_result, _to_serializable

app = FastAPI(title="AutoML Studio")
MODEL_STATE: dict = {}
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AutoML Studio</title>
  <style>
    :root {
      --bg: #edf2ff;
      --card: rgba(255,255,255,0.82);
      --soft: #f5f7ff;
      --line: #dfe7ff;
      --text: #1f2a44;
      --muted: #6e7aa6;
      --primary: #6957f5;
      --primary-2: #8e7bff;
      --success: #1dbf73;
      --shadow: 0 20px 45px rgba(108, 92, 231, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: Inter, Arial, sans-serif; background: linear-gradient(180deg, #edf2ff, #f8f9ff);
      color: var(--text);
    }
    .wrap { max-width: 1180px; margin: 18px auto; padding: 0 20px 40px; }
    .topbar {
      display: flex; align-items: center; justify-content: space-between;
      background: rgba(255,255,255,0.5); border: 1px solid var(--line); border-radius: 18px;
      box-shadow: var(--shadow); padding: 15px 22px; margin-bottom: 20px;
      backdrop-filter: blur(10px);
    }
    .brand { font-size: 2rem; font-weight: 800; letter-spacing: -0.06em; color: var(--primary); }
    nav { display: flex; gap: 14px; }
    .nav-btn {
      border: none; background: transparent; color: var(--muted); padding: 8px 12px; border-radius: 10px;
      font-weight: 700; cursor: pointer; transition: .2s ease; 
    }
    .nav-btn.active, .nav-btn:hover { background: rgba(105,87,245,0.08); color: var(--primary); }
    .hero {
      display: grid; grid-template-columns: 1.3fr 0.9fr; gap: 26px; background: rgba(255,255,255,0.7);
      border: 1px solid var(--line); border-radius: 28px; padding: 34px 30px; box-shadow: var(--shadow);
    }
    .tag {
      display: inline-flex; align-items: center; border-radius: 999px; padding: 7px 14px; font-size: 0.8rem;
      background: rgba(105,87,245,0.08); color: var(--primary); font-weight: 700; border: 1px solid rgba(105,87,245,0.15);
    }
    h1 { font-size: clamp(3rem, 5vw, 5rem); line-height: 1; letter-spacing: -0.07em; margin: 20px 0 18px; }
    .lead { font-size: 2rem; color: var(--muted); line-height: 1.4; font-weight: 500; }
    .pills { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }
    .pill {
      background: rgba(105,87,245,0.08); color: var(--primary); padding: 9px 15px; border-radius: 999px;
      border: 1px solid rgba(105,87,245,0.14); font-weight: 700; font-size: .82rem; transition: transform .2s ease, box-shadow .2s ease;
    }
    .pill:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(105,87,245,0.12); }
    .panel {
      background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(245,247,255,0.82));
      border: 1px solid var(--line); border-radius: 24px; padding: 20px; box-shadow: var(--shadow);
    }
    .panel h3 { margin: 0 0 14px; font-size: 1.15rem; }
    .field { margin-top: 14px; }
    .label { display:block; font-size: .8rem; font-weight: 700; color: var(--muted); margin-bottom: 8px; }
    .file-wrap { position: relative; }
    .file-wrap input[type=file] {
      position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; z-index: 10;
    }
    .fake-file {
      display: flex; align-items: center; justify-content: space-between; width: 100%; padding: 14px 16px; border-radius: 12px;
      border: 1px solid var(--line); background: rgba(255,255,255,0.7); color: var(--text); font-weight: 700;
      transition: .2s ease;
    }
    .fake-file:hover { border-color: var(--primary); box-shadow: 0 10px 24px rgba(105,87,245,0.08); }
    .choose-btn {
      background: linear-gradient(135deg, var(--primary), var(--primary-2)); border: none; color: white;
      border-radius: 10px; padding: 8px 14px; font-weight: 700; cursor: pointer;
    }
    select, input[type=text] {
      width: 100%; padding: 14px 16px; border-radius: 12px; border: 1px solid var(--line); background: rgba(255,255,255,0.7);
      color: var(--text); font-size: 1rem; outline: none; transition: .2s ease;
    }
    select:focus, input[type=text]:focus { border-color: var(--primary); box-shadow: 0 0 0 4px rgba(105,87,245,0.1); }
    .tiny { color: var(--muted); font-size: .82rem; margin-top: 10px; }
    .primary-btn {
      width: 100%; margin-top: 16px; border: none; border-radius: 14px; padding: 16px 18px; cursor: pointer;
      font-size: 1.1rem; font-weight: 800; color: white; background: linear-gradient(135deg, var(--primary), var(--primary-2));
      box-shadow: 0 16px 30px rgba(105,87,245,0.22); transition: transform .2s ease, box-shadow .2s ease;
    }
    .primary-btn:hover { transform: translateY(-2px); box-shadow: 0 18px 35px rgba(105,87,245,0.28); }
    .status {
      margin-top: 16px; background: rgba(29,191,115,0.08); color: #0f8d56; border: 1px solid rgba(29,191,115,0.25);
      border-radius: 12px; padding: 12px 14px; font-weight: 700; display:none;
    }
    .status.show { display:block; }
    .progress { height: 12px; border-radius: 999px; overflow: hidden; background: rgba(105,87,245,0.08); margin-top: 8px; }
    .progress > span {
      display: block; height: 100%; width: 0; background: linear-gradient(90deg, var(--primary), var(--primary-2));
      border-radius: 999px; transition: width .2s ease;
    }
    .metrics { display: grid; grid-template-columns: repeat(6, minmax(140px, 1fr)); gap: 18px; margin-top: 32px; }
    .metric { background: rgba(255,255,255,0.7); border: 1px solid var(--line); border-radius: 18px; padding: 18px; min-height: 132px; display:flex; flex-direction:column; justify-content:space-between; }
    .metric .k { font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800; line-height: 1.2; }
    .metric .v { font-size: clamp(1.35rem, 1.8vw, 2rem); font-weight: 800; letter-spacing: -0.05em; line-height: 1.15; margin-top: auto; padding-top: 10px; }
    .results { display:none; margin-top: 34px; gap: 24px; }
    .results.show { display:grid; }
    .results-grid { display:grid; grid-template-columns: 1.4fr .9fr; gap: 24px; }
    .section {
      background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(245,247,255,0.72));
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px 18px 16px;
      box-shadow: 0 12px 30px rgba(108, 92, 231, 0.08);
    }
    .section h2 { margin: 0 0 12px; font-size: 1.1rem; letter-spacing: -0.03em; }
    .section h4 { margin: 0; color: var(--muted); font-size: .9rem; font-weight: 600; }
    .preview-wrap { margin-top: 14px; overflow:auto; border: 1px solid var(--line); border-radius: 14px; }
    .scroll-panel { padding-right: 8px; }
    .preview-wrap { margin-top: 14px; height: 260px; overflow-x: auto; overflow-y: auto; border: 1px solid var(--line); border-radius: 16px; }
    .correlation-wrap { margin-top: 14px; height: 170px; overflow:auto; border: 1px solid var(--line); border-radius: 16px; }
    .correlation-list { display:grid; gap: 8px; padding: 12px; min-width: 0; }
    .corr-row { display:grid; grid-template-columns: minmax(90px, 1fr) minmax(100px, 1fr) 56px; gap: 8px; align-items:center; }
    .corr-row > div:first-child, .corr-row > div:nth-child(2) { overflow:hidden; text-overflow: ellipsis; white-space: nowrap; }
    .corr-bar { height: 14px; border-radius: 999px; background: rgba(105,87,245,0.10); overflow:hidden; }
    .corr-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--primary), var(--primary-2)); }
    .accordion details {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(255,255,255,0.64);
      margin-top: 10px;
      overflow: hidden;
      transition: border-color .2s ease, box-shadow .2s ease;
    }
    .accordion details[open] {
      border-color: rgba(105,87,245,0.35);
      box-shadow: 0 10px 24px rgba(105,87,245,0.06);
    }
    .accordion summary {
      cursor: pointer;
      list-style: none;
      padding: 15px 18px;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--text);
    }
    .accordion summary::-webkit-details-marker { display:none; }
    .accordion summary::after {
      content: '▾';
      color: var(--primary);
      font-size: 1rem;
      transition: transform .2s ease;
      transform: rotate(0deg);
    }
    .accordion details[open] summary::after {
      transform: rotate(180deg);
    }
    .accordion .body { padding: 0 18px 16px; color: var(--muted); line-height: 1.6; }
    .bar-list { display:grid; gap: 12px; margin-top: 12px; }
    .bar-row { display:grid; grid-template-columns: minmax(120px, 170px) 1fr 72px; gap: 12px; align-items:center; }
    .bar-row > div:first-child { overflow:hidden; text-overflow: ellipsis; white-space: nowrap; }
    .bar-track { height: 16px; border-radius: 999px; background: rgba(105,87,245,0.10); overflow:hidden; }
    .bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--primary), var(--primary-2)); }
    .tune-list { display:grid; gap: 12px; margin-top: 12px; }
    .tune-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.7);
      padding: 14px 16px;
      margin-top: 10px;
    }
    .tune-card .title { font-weight: 800; margin-bottom: 8px; }
    .tune-card .meta { color: var(--muted); font-size: .88rem; line-height: 1.5; }
    .table-wrap { height: 260px; overflow:auto; border: 1px solid var(--line); border-radius: 16px; margin-top: 14px; max-width: 100%; }
    .compact-card {
      display: flex;
      flex-direction: column;
      gap: 14px;
      width: min(560px, 100%);
      margin: 18px auto 0;
      padding: 18px 18px 16px;
      background: rgba(255,255,255,0.76);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      align-self: center;
    }
    .compact-card .card-header {
      display: flex;
      flex-direction: column;
      gap: 4px;
      padding: 2px 4px 0;
    }
    .compact-card .card-header h4 {
      margin: 0;
      color: var(--muted);
      font-size: .9rem;
      font-weight: 600;
    }
    .compact-card .card-header h2 {
      margin: 0;
      font-size: 1.1rem;
      line-height: 1.35;
    }
    .compact-card .matrix-panel,
    .compact-card .report-panel {
      background: rgba(255,255,255,0.55);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }
    .compact-card .matrix-panel {
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 120px;
    }
    .compact-card .matrix-panel .table-wrap {
      height: auto;
      max-width: 100%;
      margin: 0;
      overflow: visible;
      border: none;
      border-radius: 0;
      background: transparent;
    }
    .compact-card table {
      min-width: 180px;
      width: auto;
      max-width: 100%;
      margin: 0 auto;
      background: transparent;
    }
    .compact-card td {
      padding: 10px 14px;
      border-bottom: 1px solid rgba(223,231,255,0.8);
      text-align: center;
    }
    .compact-card .report-panel {
      padding: 10px 12px;
    }
    .compact-card pre {
      margin: 0;
      font-size: .76rem;
      line-height: 1.35;
      white-space: pre-wrap;
      font-family: inherit;
      color: var(--muted);
    }
    table { width: 100%; min-width: 1100px; border-collapse: collapse; background: rgba(255,255,255,0.76); table-layout: auto; }
    th, td { padding: 12px 14px; border-bottom: 1px solid rgba(223,231,255,0.8); text-align: left; white-space: nowrap; }
    th, td { overflow: visible; }
    th { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .09em; }
    .best-card { border: 1px solid var(--line); border-radius: 18px; padding: 16px; background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(245,247,255,0.88)); }
    .best-card .name { font-size: 1.2rem; font-weight: 800; margin-top: 6px; line-height: 1.2; }
    .best-card .meta { color: var(--muted); margin-top: 8px; line-height: 1.4; }
    .badge { display:inline-flex; align-items:center; padding: 7px 12px; border-radius: 999px; background: rgba(105,87,245,0.10); color: var(--primary); font-size: .78rem; font-weight: 800; }
    #featureForm { display: none; margin-top: 28px; }
    #featureForm .grid { display: grid; grid-template-columns: repeat(2, minmax(240px, 1fr)); gap: 16px; }
    .input-wrap { background: rgba(255,255,255,0.7); border: 1px solid var(--line); padding: 14px; border-radius: 14px; }
    .input-wrap label { display:block; font-weight:700; margin-bottom:8px; color: var(--muted); }
    .input-wrap input { width: 100%; padding: 12px; border-radius: 10px; border: 1px solid var(--line); }
    .results * { min-width: 0; }
    #predictBtn { display:none; }
    @media (max-width: 980px) {
      .hero { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
      #featureForm .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="topbar">
      <div class="brand">AutoML Studio</div>
      <nav id="topNav" style="display:none;">
        <button class="nav-btn active">Upload</button>
        <button class="nav-btn">Results</button>
        <button class="nav-btn">Predict</button>
        <button class="nav-btn">Details</button>
      </nav>
    </header>

    <section class="hero">
      <div>
        <span class="tag">Python AutoML Pipeline</span>
        <h1>Train smarter. Predict faster.</h1>
        <div class="lead">Upload a dataset, auto-train multiple models, then predict instantly using the saved artifact.</div>
        <div class="pills">
          <div class="pill">Progress tracking</div>
          <div class="pill">Report downloads</div>
          <div class="pill">Saved model</div>
          <div class="pill">Prediction page</div>
        </div>
      </div>

      <div class="panel">
        <h3>Dataset file</h3>
        <div class="field">
          <input id="fileInput" type="file" accept=".csv,.xlsx,.xls,.json,text/csv,text/plain,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/json" onchange="onFileSelected()" style="width:100%; padding:12px; border:1px solid var(--line); border-radius:12px; background:rgba(255,255,255,0.85); font-size:1rem; cursor:pointer;" />
          <div id="fileNotice" style="margin-top:8px; padding:10px 14px; border-radius:10px; font-size:.9rem; font-weight:700; background:rgba(105,87,245,0.08); display:none;"></div>
        </div>
        <div class="tiny" id="fileMeta">Maximum file size: 20 MB · CSV, Excel or JSON</div>

        <div class="field">
          <label class="label" for="target">Target column</label>
          <select id="target"></select>
        </div>

        <button class="primary-btn" type="button" id="trainBtn">Run AutoML Pipeline</button>
        <div class="status" id="statusBox">Pipeline completed successfully. <span id="statusPct">100%</span></div>
        <div class="progress"><span id="progressBar"></span></div>
      </div>
    </section>

    <section class="metrics" id="metrics"></section>

    <section class="results" id="results">
      <div class="section">
        <h4>Dataset preview</h4>
        <h2>First five rows</h2>
        <div class="preview-wrap"><table id="previewTable"></table></div>
      </div>

      <div class="results-grid">
        <div class="section">
          <h4>Pipeline steps</h4>
          <h2>Data preprocessing summary</h2>
          <div class="accordion" id="prepAccordion"></div>
        </div>
        <div class="section">
          <h4>Hyperparameter tuning</h4>
          <h2>Best tuned models</h2>
          <div id="tuningBox"></div>
        </div>
      </div>

      <div class="results-grid" style="margin-top:24px;">
        <div class="section">
          <h4>Model comparison</h4>
          <h2>Top 3 model comparison</h2>
          <div id="comparisonBars"></div>
          <div class="table-wrap"><table id="comparisonTable"></table></div>
        </div>
        <div class="section">
          <h4>Model interpretability</h4>
          <h2>Feature importance highlights</h2>
          <div id="importanceBox"></div>
        </div>
      </div>
    </section>

    <form id="featureForm">
      <div class="grid" id="featureGrid"></div>
      <button class="primary-btn" id="predictBtn" type="submit">Predict</button>
    </form>
  </div>

  <script>
    const state = { result: null, file: null };

    const fileInput = document.getElementById('fileInput');
    const fileNotice = document.getElementById('fileNotice');
    const targetInput = document.getElementById('target');
    const metrics = document.getElementById('metrics');
    const results = document.getElementById('results');
    const previewTable = document.getElementById('previewTable');
    const correlationList = document.getElementById('correlationList');
    const prepAccordion = document.getElementById('prepAccordion');
    const bestCard = document.getElementById('bestCard');
    const comparisonBars = document.getElementById('comparisonBars');
    const comparisonTable = document.getElementById('comparisonTable');
    const importanceBox = document.getElementById('importanceBox');
    const tuningBox = document.getElementById('tuningBox');
    const featureForm = document.getElementById('featureForm');
    const featureGrid = document.getElementById('featureGrid');
    const statusBox = document.getElementById('statusBox');
    const progressBar = document.getElementById('progressBar');
    const trainBtn = document.getElementById('trainBtn');
    const predictBtn = document.getElementById('predictBtn');

    async function parseColumnsLocally(file) {
      const ext = '.' + file.name.split('.').pop().toLowerCase();
      if (ext === '.csv') {
        try {
          const slice = file.slice(0, 8192);
          const text = await slice.text();
          const firstLine = text.split(String.fromCharCode(10))[0].replace(String.fromCharCode(13), '');
          if (firstLine && firstLine.trim().length > 0) {
            const delim = firstLine.includes('\t') ? '\t' : (firstLine.includes(';') && !firstLine.includes(',') ? ';' : ',');
            const cols = [];
            let cur = '';
            let inQuotes = false;
            for (let i = 0; i < firstLine.length; i++) {
              const ch = firstLine[i];
              if (ch === '"') {
                inQuotes = !inQuotes;
              } else if (ch === delim && !inQuotes) {
                cols.push(cur.trim().replace(/^["']|["']$/g, ''));
                cur = '';
              } else {
                cur += ch;
              }
            }
            cols.push(cur.trim().replace(/^["']|["']$/g, ''));
            const clean = cols.filter(c => c.length > 0);
            if (clean.length > 0) return clean;
          }
        } catch (e) {
          console.warn('Local CSV parse error', e);
        }
      } else if (ext === '.json') {
        try {
          const slice = file.slice(0, 16384);
          const text = await slice.text();
          const trimmed = text.trim();
          if (trimmed.startsWith('[')) {
            const endIdx = trimmed.indexOf('}');
            if (endIdx > 0) {
              const obj = JSON.parse(trimmed.slice(1, endIdx + 1).trim());
              return Object.keys(obj);
            }
          }
        } catch (e) {
          console.warn('Local JSON parse error', e);
        }
      }
      return null;
    }

    async function onFileSelected() {
      const file = fileInput.files && fileInput.files.length ? fileInput.files[0] : null;
      if (!file) {
        if (fileNotice) { fileNotice.style.display = 'none'; fileNotice.textContent = ''; }
        return;
      }

      const allowedExts = ['.csv', '.xlsx', '.xls', '.json'];
      const fileExt = '.' + file.name.split('.').pop().toLowerCase();
      if (!allowedExts.includes(fileExt)) {
        if (fileNotice) {
          fileNotice.style.display = 'block';
          fileNotice.style.background = 'rgba(239,68,68,0.1)';
          fileNotice.style.color = '#dc2626';
          fileNotice.innerHTML = '❌ <strong>Invalid file format (' + (fileExt || 'none') + '):</strong> Only .csv, .xlsx, .xls, and .json files are supported.';
        }
        fileInput.value = '';
        state.file = null;
        state.file_id = null;
        targetInput.innerHTML = '';
        return;
      }

      if (file.size > 20 * 1024 * 1024) {
        if (fileNotice) {
          fileNotice.style.display = 'block';
          fileNotice.style.background = 'rgba(239,68,68,0.1)';
          fileNotice.style.color = '#dc2626';
          fileNotice.innerHTML = '❌ <strong>File too large:</strong> Maximum allowed size is 20 MB.';
        }
        fileInput.value = '';
        state.file = null;
        state.file_id = null;
        targetInput.innerHTML = '';
        return;
      }

      state.file = file;

      // Instant client-side column extraction (0 ms)
      const localCols = await parseColumnsLocally(file);
      if (localCols && localCols.length > 0) {
        targetInput.innerHTML = localCols.map((c, index) => `<option value="${c}"${index === localCols.length - 1 ? ' selected' : ''}>${c}</option>`).join('');
        targetInput.selectedIndex = localCols.length - 1;
        if (fileNotice) {
          fileNotice.style.display = 'block';
          fileNotice.style.background = 'rgba(29,191,115,0.08)';
          fileNotice.style.color = '#0f8d56';
          fileNotice.innerHTML = '⚡ <strong>' + file.name + ' ready!</strong> (' + localCols.length + ' columns detected instantly). Syncing with server...';
        }
      } else if (fileNotice) {
        fileNotice.style.display = 'block';
        fileNotice.style.background = 'rgba(105,87,245,0.08)';
        fileNotice.style.color = 'var(--primary)';
        fileNotice.innerHTML = '⏳ <strong>Analyzing ' + file.name + '...</strong>';
      }

      const fd = new FormData();
      fd.append('file', file);

      state.uploadPromise = fetch('/api/upload', { method: 'POST', body: fd }).then(async (res) => {
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || res.statusText || 'Upload failed');
        }
        const data = await res.json();
        state.file_id = data.file_id;
        const serverCols = data.columns || [];
        if (!localCols && serverCols.length > 0) {
          targetInput.innerHTML = serverCols.map((c, index) => `<option value="${c}"${index === serverCols.length - 1 ? ' selected' : ''}>${c}</option>`).join('');
          if (serverCols.length) targetInput.selectedIndex = serverCols.length - 1;
        }
        if (fileNotice) {
          fileNotice.style.display = 'block';
          fileNotice.style.background = 'rgba(29,191,115,0.1)';
          fileNotice.style.color = '#0f8d56';
          fileNotice.innerHTML = '✓ <strong>' + (data.filename || file.name) + ' uploaded!</strong> (' + (serverCols.length || (localCols ? localCols.length : 0)) + ' columns detected). Ready to run pipeline.';
        }
        return data.file_id;
      }).catch((e) => {
        if (fileNotice) {
          fileNotice.style.display = 'block';
          fileNotice.style.background = 'rgba(239,68,68,0.1)';
          fileNotice.style.color = '#dc2626';
          fileNotice.innerHTML = '❌ <strong>Upload error:</strong> ' + e.message;
        }
        state.file_id = null;
        state.uploadPromise = null;
      });
    }

    fileInput.addEventListener('change', onFileSelected);
    fileInput.addEventListener('input', onFileSelected);

    trainBtn.addEventListener('click', async () => {
      let fileId = state.file_id;
      if (!fileId && state.uploadPromise) {
        statusBox.classList.add('show');
        statusBox.innerHTML = 'Syncing dataset with server...';
        fileId = await state.uploadPromise;
      }
      if (!fileId) {
        const file = state.file || (fileInput.files && fileInput.files[0]);
        if (!file) return alert('Please choose a dataset first.');
        const uploadFd = new FormData();
        uploadFd.append('file', file);
        const up = await fetch('/api/upload', { method: 'POST', body: uploadFd });
        if (!up.ok) {
          const err = await up.json().catch(() => ({}));
          return alert('Upload failed: ' + (err.detail || up.statusText));
        }
        const upData = await up.json();
        fileId = upData.file_id;
        state.file_id = fileId;
      }

      statusBox.classList.add('show');
      statusBox.innerHTML = 'Starting pipeline...';
      progressBar.style.width = '4%';

      const es = new EventSource(`/api/train-stream?file_id=${encodeURIComponent(fileId)}&target=${encodeURIComponent(targetInput.value || '')}`);
      es.addEventListener('progress', (e) => {
        try {
          const payload = JSON.parse(e.data);
          statusBox.innerHTML = payload.message || 'Processing...';
          progressBar.style.width = (payload.progress || 0) + '%';
        } catch (err) {
          console.error('Bad progress event', err);
        }
      });
      es.addEventListener('result', (e) => {
        try {
          const payload = JSON.parse(e.data);
          state.result = payload;
          renderPreview(payload.dataset?.preview || []);
          renderMetrics(payload.summary || payload.dashboard_summary || {});
          renderResults(payload);
          renderFeatureFields(payload.feature_names || payload.selected_feature_names || []);
          statusBox.innerHTML = 'Pipeline completed successfully. <span id="statusPct">100%</span>';
          progressBar.style.width = '100%';
          // show navigation bar only after we have results
          document.getElementById('topNav').style.display = 'flex';
        } catch (err) {
          console.error('Bad result event', err);
        }
      });
      es.addEventListener('error', (e) => {
        let msg = 'Pipeline failed or disconnected.';
        try {
          if (e.data) {
            const parsed = JSON.parse(e.data);
            if (parsed.message) msg = parsed.message;
          }
        } catch (_) {}
        statusBox.innerHTML = '❌ <strong>Error:</strong> ' + msg;
        statusBox.style.background = 'rgba(239,68,68,0.1)';
        statusBox.style.color = '#dc2626';
        console.error('SSE error', e);
        es.close();
      });
    });

    featureForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const payload = {};
      for (const input of document.querySelectorAll('#featureGrid input')) {
        payload[input.name] = input.value;
      }
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json().catch(() => ({}));

      if (!resp.ok) {
        const detail = data?.detail || 'Prediction request failed.';
        alert('Prediction error: ' + detail + ' - Please train the model first or re-run the pipeline.');
        return;
      }

      const prediction = data?.prediction;
      const prob0 = Number(data?.probabilities?.['0'] ?? 0) * 100;
      const prob1 = Number(data?.probabilities?.['1'] ?? 0) * 100;

      if (prediction === undefined || prediction === null) {
        alert('Prediction failed: the backend did not return a valid prediction value.');
        return;
      }

      alert('Prediction: ' + prediction + ' | Class 0: ' + prob0.toFixed(2) + '% | Class 1: ' + prob1.toFixed(2) + '%');
    });

    function renderMetrics(summary) {
      const missingValueLabel = Number(summary.missing_values || 0) === 0 ? 'No missing values' : (summary.missing_values || 0);
      const metricValue = formatSummaryMetric(summary.best_metric_label, summary.best_metric_value ?? summary.best_metric ?? '—');
      const cards = [
        ['Problem type', summary.problem_type || 'Classification'],
        ['Rows', summary.rows || 0],
        ['Columns', summary.columns || 0],
        ['Missing values', missingValueLabel],
        ['Best model', summary.best_model || '—'],
        [summary.best_metric_label || 'Accuracy', metricValue],
      ];
      metrics.innerHTML = cards.map(([k,v]) => '<div class="metric"><div class="k">' + k + '</div><div class="v">' + v + '</div></div>').join('');
    }

    function renderResults(payload) {
      results.classList.add('show');
      const preprocessing = payload.preprocessing || {};
      const droppedReasons = payload.dataset?.dropped_column_reasons || {};
      prepAccordion.innerHTML = [
        ['Dropped columns', payload.dataset?.dropped_columns || []],
        ['Dropped column reasons', droppedReasons],
        ['Missing values filled', Object.keys(preprocessing.missing_value_report || {}).length ? preprocessing.missing_value_report : 'No missing values'],
        ['Encoding applied', preprocessing.encoding_report || {}],
      ].map(([title, data]) => `<details><summary>${title}</summary><div class="body">${formatBlock(data)}</div></details>`).join('');

      const comparison = payload.comparison || [];
      const metric = payload.primary_metric || (payload.problem_type === 'Regression' ? 'R2' : 'Accuracy');
      const topComparison = comparison.slice(0, 3);
      const maxValue = Math.max(...topComparison.map((row) => Number(row[metric]) || 0), 1);
      comparisonBars.innerHTML = topComparison.map((row) => `<div class="bar-row"><div>${row.Model}</div><div class="bar-track"><div class="bar-fill" style="width:${Math.max(((Number(row[metric]) || 0) / maxValue) * 100, 4)}%"></div></div><div>${formatMetric(row[metric])}</div></div>`).join('');
      comparisonTable.innerHTML = `<thead><tr>${Object.keys(comparison[0] || { Model:'', Accuracy:'', Precision:'', Recall:'', 'F1 Score':'', 'ROC-AUC':'' }).map((k) => `<th>${k}</th>`).join('')}</tr></thead><tbody>${topComparison.map((row) => `<tr>${Object.values(row).map((value) => `<td>${formatMetric(value)}</td>`).join('')}</tr>`).join('')}</tbody>`;

      const importance = payload.feature_importance || {};
      const correlationPairs = payload.dataset?.correlation_pairs || [];
      const options = [
        ...Object.keys(importance),
        'Correlation'
      ];
      const firstModel = options[0] || 'Correlation';
      importanceBox.innerHTML = `<div class="field"><label class="label">Model</label><select id="importanceModel">${options.map((name) => `<option value="${name}">${name}</option>`).join('')}</select></div><div id="importanceList"></div>`;
      const drawImportance = (name) => {
        if (name === 'Correlation') {
          if (!correlationPairs.length) {
            document.getElementById('importanceList').innerHTML = '<div class="tiny">No numeric correlation data available.</div>';
            return;
          }
          const maxValue = Math.max(...correlationPairs.map((row) => Number(row.correlation) || 0), 1);
          document.getElementById('importanceList').innerHTML = correlationPairs.map((row) => `<div class="corr-row"><div>${row.feature_a}</div><div class="corr-bar"><div class="corr-fill" style="width:${Math.max((Number(row.correlation) / maxValue) * 100, 4)}%"></div></div><div>${(Number(row.correlation) || 0).toFixed(2)}</div><div style="grid-column:1 / -1; color:var(--muted); font-size:.82rem; margin-top:-4px;">${row.feature_b}</div></div>`).join('');
          return;
        }

        const rows = (importance[name] || []).slice(0, 10);
        document.getElementById('importanceList').innerHTML = rows.length ? `<div class="bar-list">${rows.map((row) => `<div class="bar-row"><div>${row.feature}</div><div class="bar-track"><div class="bar-fill" style="width:${Math.max((Number(row.importance) || 0) * 100, 4)}%"></div></div><div>${(Number(row.importance) * 100 || 0).toFixed(1)}%</div></div>`).join('')}</div>` : '<div class="tiny">No feature-importance data for this model.</div>';
      };
      drawImportance(firstModel);
      document.getElementById('importanceModel').onchange = (e) => drawImportance(e.target.value);

      const tuningSummary = payload.tuning_summary || {};
      const tunedModels = Object.entries(tuningSummary)
        .filter(([, info]) => info && info.was_tuned)
        .slice(0, 3);
      tuningBox.innerHTML = tunedModels.length
        ? `<div class="tune-list">${tunedModels.map(([name, info]) => `<div class="tune-card"><div class="title">${name}</div><div class="meta">Baseline CV: ${formatMetric(info.baseline_cv_score)}<br/>Tuned CV: ${formatMetric(info.tuned_cv_score)}<br/>Best params: ${formatBlock(info.best_params || 'No tuned params')}</div></div>`).join('')}</div>`
        : '<div class="tiny">No tuned models were retained.</div>';
    }

    function formatBlock(value) {
      if (!value || (Array.isArray(value) && !value.length)) return '<div class="tiny">None</div>';
      if (Array.isArray(value)) return value.map((item) => `<div>${item}</div>`).join('');
      if (typeof value === 'object') return Object.entries(value).map(([k, v]) => `<div><strong>${k}</strong>: ${v}</div>`).join('');
      return `<div>${value}</div>`;
    }

    function formatSummaryMetric(label, value) {
      if (value === null || value === undefined || value === '—') return '—';
      const numeric = Number(value);
      if (Number.isNaN(numeric)) return value;
      const metricLabel = String(label || '').toLowerCase();
      if (metricLabel.includes('accuracy') || metricLabel.includes('precision') || metricLabel.includes('recall') || metricLabel.includes('f1') || metricLabel.includes('roc')) {
        return (numeric * 100).toFixed(2) + '%';
      }
      return Number.isInteger(numeric) ? String(numeric) : numeric.toFixed(2);
    }

    function renderPreview(rows) {
      if (!rows.length) {
        previewTable.innerHTML = '<tbody><tr><td class="tiny">No preview available.</td></tr></tbody>';
        return;
      }
      const columns = Object.keys(rows[0]);
      previewTable.innerHTML = `<thead><tr>${columns.map((col) => `<th>${col}</th>`).join('')}</tr></thead><tbody>${rows.map((row) => `<tr>${columns.map((col) => `<td>${row[col] ?? ''}</td>`).join('')}</tr>`).join('')}</tbody>`;
    }

    function metricLabel(payload) {
      return payload.best_model?.Model ? (payload.primary_metric || 'Accuracy') : 'Accuracy';
    }

    function formatMetric(value) {
      if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4);
      return value ?? '—';
    }

    function renderFeatureFields(names) {
      if (!names || !names.length) return;
      featureGrid.innerHTML = names.map((name) => `
        <div class="input-wrap">
          <label>${name}</label>
          <input name="${name}" type="number" value="0" step="any" />
        </div>
      `).join('');
      featureForm.style.display = 'block';
      predictBtn.style.display = 'block';
    }
  </script>
</body>
</html>
"""


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def home() -> str:
    return HTML


ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.json'}


@app.post('/api/columns')
async def detect_columns(file: UploadFile = File(...)):
    suffix = Path(file.filename or 'data.csv').suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            content={'detail': f"Unsupported file type '{suffix or 'none'}'. Supported formats: .csv, .xlsx, .xls, .json"},
            status_code=400,
        )
    raw = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or '.csv') as tmp:
        tmp.write(raw)
        path = tmp.name
    try:
        if suffix in {'.csv', ''}:
            df = pd.read_csv(path, nrows=0)
        elif suffix in {'.xlsx', '.xls'}:
            df = pd.read_excel(path, nrows=0)
        elif suffix == '.json':
            try:
                df = pd.read_json(path, nrows=1)
            except Exception:
                df = pd.read_json(path)
        columns = [str(c).strip() for c in df.columns]
        if not columns:
            return JSONResponse(content={'detail': 'No columns found in dataset'}, status_code=400)
        return JSONResponse(content=columns)
    except Exception as e:
        return JSONResponse(content={'detail': f'Error reading columns: {str(e)}'}, status_code=400)
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post('/api/upload')
async def upload_file(file: UploadFile = File(...)):
    suffix = Path(file.filename or 'data.csv').suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            {'detail': f"Unsupported file type '{suffix or 'none'}'. Supported formats: .csv, .xlsx, .xls, .json"},
            status_code=400,
        )
    file_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{file_id}{suffix}"
    raw = await file.read()
    with open(dest, 'wb') as f:
        f.write(raw)
    try:
        if suffix in {'.csv', ''}:
            df = pd.read_csv(dest, nrows=0)
        elif suffix in {'.xlsx', '.xls'}:
            df = pd.read_excel(dest, nrows=0)
        elif suffix == '.json':
            try:
                df = pd.read_json(dest, nrows=1)
            except Exception:
                df = pd.read_json(dest)
        columns = [str(c).strip() for c in df.columns]
        if not columns:
            if os.path.exists(dest):
                os.remove(dest)
            return JSONResponse({'detail': 'No columns detected in dataset'}, status_code=400)
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        return JSONResponse({'detail': f'Failed to parse dataset: {str(e)}'}, status_code=400)
    return JSONResponse({'file_id': file_id, 'filename': file.filename or 'data.csv', 'columns': columns})


@app.get('/api/train-stream')
async def train_stream(file_id: str, target: str | None = None):
    # Stream progress via Server-Sent Events
    from fastapi.responses import StreamingResponse

    # find uploaded file
    found = None
    for p in UPLOAD_DIR.iterdir():
        if p.stem == file_id:
            found = p
            break
    if found is None:
        return JSONResponse({'detail': 'Uploaded file not found'}, status_code=404)

    q = queue.Queue()

    def progress_cb(progress, message):
        q.put({'type': 'progress', 'progress': int(progress), 'message': message})

    def run_pipeline():
        try:
            result = _build_pipeline_result(job_id=str(uuid.uuid4()), file_path=str(found), target_column=target or '', progress_callback=progress_cb)
            best_model_object = result.get('best_model_object')
            serializable_result = _to_serializable(dict(result))
            if isinstance(serializable_result, dict):
                serializable_result.pop('best_model_object', None)
            MODEL_STATE.clear()
            if isinstance(serializable_result, dict):
                MODEL_STATE.update(serializable_result)
            if best_model_object is not None:
                MODEL_STATE['best_model_object'] = best_model_object
            q.put({'type': 'result', 'result': serializable_result})
        except Exception as exc:
            q.put({'type': 'error', 'message': str(exc)})

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    def event_stream():
      while True:
        item = q.get()
        if item['type'] == 'progress':
          payload = json.dumps({'progress': item['progress'], 'message': item['message']})
          yield f"event: progress\ndata: {payload}\n\n"
        elif item['type'] == 'result':
          yield f"event: result\ndata: {json.dumps(item['result'])}\n\n"
          break
        elif item['type'] == 'error':
          yield f"event: error\ndata: {json.dumps({'message': item['message']})}\n\n"
          break

    return StreamingResponse(event_stream(), media_type='text/event-stream')


def _detect_default_target(columns: list[str]) -> str | None:
    if not columns:
        return None
    norm = {str(c).strip().lower(): c for c in columns}
    for key in ["survived", "target", "label", "class", "y", "loan_status", "income", "outcome", "status"]:
        if key in norm:
            return norm[key]
    for cand in ["survived", "target", "label", "class", "y"]:
        for c in columns:
            if cand in str(c).lower():
                return c
    return columns[-1]


@app.post('/api/train')
async def train(file: UploadFile = File(...), target_column: str = Form(None)):
    raw = await file.read()
    suffix = Path(file.filename or 'data.csv').suffix.lower() or '.csv'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        path = tmp.name
    try:
        # try to read columns and pick a default if the provided target is missing
        try:
            df = pd.read_csv(path) if suffix in {'.csv', ''} else pd.read_excel(path)
            cols = list(df.columns)
        except Exception:
            cols = []
        if not target_column or (cols and target_column not in cols):
            detected = _detect_default_target(cols)
            if detected:
                target_column = detected

        result = _build_pipeline_result(
            job_id=str(uuid.uuid4()),
            file_path=path,
            target_column=target_column,
            progress_callback=None,
        )
        summary = result.get('dashboard_summary', {})
        MODEL_STATE.clear()
        MODEL_STATE.update(result)
        return {
            'summary': {
                'problem_type': summary.get('problem_type', result.get('problem_type', 'Classification')),
                'rows': summary.get('rows', result.get('dataset', {}).get('rows', 0)),
                'columns': summary.get('columns', result.get('dataset', {}).get('columns', 0)),
                'missing_values': summary.get('missing_values', result.get('dataset', {}).get('missing_values_total', 0)),
                'best_model': summary.get('best_model', result.get('best_model_name', '—')),
                'best_metric': summary.get('best_metric_value', '—'),
            },
            'feature_names': result.get('selected_feature_names') or result.get('feature_names', []),
            'detected_target': target_column,
        }
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post('/api/predict')
async def predict(payload: dict):
    model = MODEL_STATE.get('best_model_object')
    features = MODEL_STATE.get('selected_feature_names') or MODEL_STATE.get('feature_names') or []
    if model is None or not features:
        return JSONResponse({'detail': 'Train a model first.'}, status_code=400)
    row = {name: float(payload.get(name, 0)) for name in features}
    df = pd.DataFrame([row], columns=features)
    pred = int(model.predict(df)[0])
    probs = {}
    if hasattr(model, 'predict_proba'):
        classes = [str(c) for c in model.classes_.tolist()]
        vals = model.predict_proba(df)[0].tolist()
        probs = {cls: float(v) for cls, v in zip(classes, vals, strict=False)}
    return {'prediction': pred, 'probabilities': probs}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False)