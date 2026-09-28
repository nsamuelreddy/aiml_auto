import os
import shutil
import uuid
import json
import threading
import queue
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware

from backend.main import _build_pipeline_result, _to_serializable, detect_default_target

app = FastAPI(title="AutoML Studio")
app.add_middleware(GZipMiddleware, minimum_size=1000)

MODEL_STATE: dict = {}
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DIR = Path(__file__).resolve().parent / "demo_datasets"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

def _cleanup_old_uploads():
    cutoff = time.time() - 86400
    for p in UPLOAD_DIR.glob("*.csv"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
        except Exception:
            pass

threading.Thread(target=_cleanup_old_uploads, daemon=True).start()

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
      --bg-gradient: linear-gradient(180deg, #edf2ff, #f8f9ff);
      --card: rgba(255, 255, 255, 0.85);
      --card-solid: #ffffff;
      --panel-bg: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(245,247,255,0.85));
      --soft: #f5f7ff;
      --line: #dfe7ff;
      --text: #1f2a44;
      --muted: #6e7aa6;
      --primary: #6957f5;
      --primary-2: #8e7bff;
      --success: #1dbf73;
      --shadow: 0 20px 45px rgba(108, 92, 231, 0.12);
      --input-bg: rgba(255, 255, 255, 0.9);
      --table-bg: rgba(255, 255, 255, 0.85);
      --code-bg: rgba(245, 247, 255, 0.8);
      --pill-bg: rgba(105, 87, 245, 0.08);
      --pill-border: rgba(105, 87, 245, 0.14);
    }
    [data-theme="dark"] {
      --bg: #0b0f19;
      --bg-gradient: linear-gradient(180deg, #0b0f19, #131b2e);
      --card: rgba(21, 30, 48, 0.88);
      --card-solid: #151e30;
      --panel-bg: linear-gradient(180deg, rgba(21, 30, 48, 0.95), rgba(16, 23, 38, 0.9));
      --soft: #19243a;
      --line: #263554;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --primary: #818cf8;
      --primary-2: #a5b4fc;
      --success: #10b981;
      --shadow: 0 20px 45px rgba(0, 0, 0, 0.45);
      --input-bg: #19243a;
      --table-bg: rgba(21, 30, 48, 0.85);
      --code-bg: #101726;
      --pill-bg: rgba(129, 140, 248, 0.14);
      --pill-border: rgba(129, 140, 248, 0.25);
    }
    *, *::before, *::after { box-sizing: border-box; }
    html, body {
      margin: 0; padding: 0; width: 100%; max-width: 100%; overflow-x: hidden;
      font-family: Inter, Arial, sans-serif; background: var(--bg-gradient);
      color: var(--text);
      transition: background 0.25s ease, color 0.25s ease;
    }
    .wrap { max-width: 1180px; margin: 18px auto; padding: 0 20px 40px; }
    .topbar {
      display: flex; align-items: center; justify-content: space-between;
      background: var(--card); border: 1px solid var(--line); border-radius: 18px;
      box-shadow: var(--shadow); padding: 12px 20px; margin-bottom: 20px;
      backdrop-filter: blur(10px);
      position: sticky; top: 12px; z-index: 100;
      gap: 12px;
    }
    .hero, .results, #featureForm, #metrics, .section, #detailsSection {
      scroll-margin-top: 90px;
    }
    .brand { font-size: 1.55rem; font-weight: 800; letter-spacing: -0.05em; color: var(--primary); white-space: nowrap; order: 1; }
    nav { display: flex; gap: 6px; order: 2; }
    .nav-btn {
      border: none; background: transparent; color: var(--muted); padding: 7px 12px; border-radius: 10px;
      font-weight: 700; cursor: pointer; transition: .2s ease; font-size: .88rem;
    }
    .nav-btn.active, .nav-btn:hover { background: var(--pill-bg); color: var(--primary); }
    #themeToggle { order: 3; }
    .hero {
      display: grid; grid-template-columns: 1.3fr 0.9fr; gap: 26px; background: var(--card);
      border: 1px solid var(--line); border-radius: 28px; padding: 34px 30px; box-shadow: var(--shadow);
    }
    .tag {
      display: inline-flex; align-items: center; border-radius: 999px; padding: 7px 14px; font-size: 0.8rem;
      background: var(--pill-bg); color: var(--primary); font-weight: 700; border: 1px solid var(--pill-border);
    }
    h1 { font-size: clamp(3rem, 5vw, 5rem); line-height: 1; letter-spacing: -0.07em; margin: 20px 0 18px; }
    .lead { font-size: 2rem; color: var(--muted); line-height: 1.4; font-weight: 500; }
    .pills { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }
    .pill {
      background: var(--pill-bg); color: var(--primary); padding: 9px 15px; border-radius: 999px;
      border: 1px solid var(--pill-border); font-weight: 700; font-size: .82rem; transition: transform .2s ease, box-shadow .2s ease;
    }
    .pill:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(105,87,245,0.12); }
    .panel {
      background: var(--panel-bg);
      border: 1px solid var(--line); border-radius: 24px; padding: 20px; box-shadow: var(--shadow);
    }
    .panel h3 { margin: 0 0 14px; font-size: 1.15rem; }
    .field { margin-top: 14px; }
    .label { display:block; font-size: .8rem; font-weight: 700; color: var(--muted); margin-bottom: 8px; }
    select, input[type=text] {
      width: 100%; padding: 14px 16px; border-radius: 12px; border: 1px solid var(--line); background: var(--input-bg);
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
    .progress { height: 12px; border-radius: 999px; overflow: hidden; background: var(--pill-bg); margin-top: 8px; }
    .progress > span {
      display: block; height: 100%; width: 0; background: linear-gradient(90deg, var(--primary), var(--primary-2));
      border-radius: 999px; transition: width .2s ease;
    }
    .metrics { display: grid; grid-template-columns: repeat(6, minmax(140px, 1fr)); gap: 14px; margin-top: 20px; }
    .metric { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 14px 16px; min-height: 100px; display:flex; flex-direction:column; justify-content:space-between; }
    .metric .k { font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800; line-height: 1.2; }
    .metric .v { font-size: clamp(1.2rem, 1.6vw, 1.8rem); font-weight: 800; letter-spacing: -0.05em; line-height: 1.15; margin-top: auto; padding-top: 6px; }
    .results { display:none; margin-top: 18px; gap: 14px; }
    .results.show { display:grid; }
    .results-columns { display:grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }
    .results-col { display:flex; flex-direction:column; gap: 14px; }
    #importanceList { max-height: 330px; overflow-y: auto; padding-right: 4px; }
    .section {
      background: var(--panel-bg);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px 18px 14px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
      height: fit-content;
    }
    .section h2 { margin: 0 0 10px; font-size: 1.05rem; letter-spacing: -0.03em; }
    .section h4 { margin: 0 0 4px; color: var(--muted); font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .preview-wrap { margin-top: 10px; height: auto; max-height: 220px; overflow-x: auto; overflow-y: auto; border: 1px solid var(--line); border-radius: 14px; }
    .accordion details {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--card);
      margin-top: 8px;
      overflow: hidden;
      transition: border-color .2s ease, box-shadow .2s ease;
    }
    .accordion details[open] {
      border-color: var(--primary);
      box-shadow: 0 10px 24px rgba(105,87,245,0.06);
    }
    .accordion summary {
      cursor: pointer;
      list-style: none;
      padding: 12px 14px;
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
    .accordion .body { padding: 0 14px 12px; color: var(--muted); line-height: 1.6; }
    .bar-list, #comparisonBars { display:grid; gap: 10px; margin-top: 10px; }
    .bar-row { display:grid; grid-template-columns: minmax(130px, 1.2fr) 2fr 64px; gap: 10px; align-items:center; }
    .bar-row > div:first-child { overflow:hidden; text-overflow: ellipsis; white-space: nowrap; }
    .bar-track { height: 16px; border-radius: 999px; background: var(--pill-bg); overflow:hidden; }
    .bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--primary), var(--primary-2)); }
    .tune-list { display:grid; gap: 8px; margin-top: 8px; }
    .tune-card {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--card);
      padding: 10px 14px;
      margin-top: 8px;
    }
    .tune-card .title { font-weight: 800; margin-bottom: 4px; }
    .tune-card .meta { color: var(--muted); font-size: .85rem; line-height: 1.45; }
    .table-wrap, .preview-wrap {
      width: 100%;
      max-height: 380px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
      margin-top: 10px;
      background: var(--card-solid);
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03);
      -webkit-overflow-scrolling: touch;
    }
    table {
      width: 100%;
      min-width: 100%;
      border-collapse: collapse;
      background: var(--table-bg);
      color: var(--text);
      font-size: 0.88rem;
      text-align: left;
    }
    thead th {
      background: var(--soft);
      color: var(--text);
      font-weight: 700;
      font-size: 0.82rem;
      padding: 12px 16px;
      border-bottom: 2px solid var(--line);
      border-right: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 2;
      white-space: nowrap;
    }
    thead th:last-child {
      border-right: none;
    }
    tbody td {
      padding: 11px 16px;
      border-bottom: 1px solid var(--line);
      border-right: 1px solid var(--line);
      white-space: nowrap;
      font-size: 0.88rem;
    }
    tbody td:last-child {
      border-right: none;
    }
    tbody tr:last-child td {
      border-bottom: none;
    }
    tbody tr:nth-child(even) {
      background: rgba(105, 87, 245, 0.025);
    }
    [data-theme="dark"] tbody tr:nth-child(even) {
      background: rgba(255, 255, 255, 0.02);
    }
    tbody tr:hover {
      background: rgba(105, 87, 245, 0.06);
    }
    [data-theme="dark"] tbody tr:hover {
      background: rgba(129, 140, 248, 0.1);
    }
    #featureForm { display: none; margin-top: 28px; }
    #featureForm .grid { display: grid; grid-template-columns: repeat(2, minmax(240px, 1fr)); gap: 16px; }
    .input-wrap { background: var(--card); border: 1px solid var(--line); padding: 14px; border-radius: 14px; }
    .input-wrap label { display:block; font-weight:700; margin-bottom:8px; color: var(--muted); }
    .input-wrap input, .input-wrap select {
      width: 100%; padding: 12px 14px; border-radius: 10px; border: 1px solid var(--line);
      background: var(--input-bg); color: var(--text); font-size: 0.95rem; font-family: inherit; outline: none; transition: .2s ease;
    }
    .input-wrap select { cursor: pointer; }
    .input-wrap input:focus, .input-wrap select:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(105,87,245,0.1); }
    .results * { min-width: 0; }
    @media (max-width: 980px) {
      .hero { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(3, minmax(130px, 1fr)); }
      .results-columns { grid-template-columns: 1fr !important; }
      #featureForm .grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 768px) {
      .topbar {
        flex-wrap: wrap;
        padding: 10px 14px;
        border-radius: 14px;
        top: 8px;
        margin-bottom: 14px;
        gap: 8px 10px;
      }
      .brand { font-size: 1.25rem; order: 1; }
      #themeToggle { order: 2; padding: 5px 10px !important; font-size: .8rem !important; }
      nav {
        order: 3;
        width: 100%;
        display: flex;
        gap: 4px;
        justify-content: space-between;
        border-top: 1px solid var(--line);
        padding-top: 6px;
      }
      .nav-btn {
        flex: 1;
        padding: 6px 4px;
        font-size: .78rem;
        text-align: center;
        white-space: nowrap;
      }
      .hero, .results, #featureForm, #metrics, .section, #detailsSection {
        scroll-margin-top: 110px;
      }
    }
    @media (max-width: 640px) {
      .wrap { padding: 0 10px 30px; margin: 6px auto; }
      .hero { padding: 18px 14px; border-radius: 18px; gap: 16px; }
      h1 { font-size: 1.85rem; margin: 12px 0 10px; }
      .lead { font-size: 1rem; line-height: 1.35; }
      .pills { gap: 6px; margin-top: 12px; }
      .pill { padding: 5px 9px; font-size: .72rem; }
      .panel { padding: 14px; border-radius: 16px; }
      .primary-btn { padding: 13px 16px; font-size: .95rem; }
      .metrics { grid-template-columns: repeat(2, 1fr); gap: 8px; margin-top: 14px; }
      .metric { padding: 10px 12px; min-height: 75px; border-radius: 12px; }
      .metric .k { font-size: .62rem; }
      .metric .v { font-size: 1.15rem; }
      .results { margin-top: 14px; gap: 12px; }
      .section { padding: 14px 12px; border-radius: 14px; }
      .section h2 { font-size: 1rem; margin-bottom: 8px; }
      .section h4 { font-size: .75rem; }
      .bar-row { grid-template-columns: minmax(80px, 1fr) 1.2fr 44px; gap: 6px; font-size: .78rem; }
      .bar-track { height: 12px; }
      .table-wrap, .preview-wrap { border-radius: 12px; margin-top: 8px; }
      table { min-width: 500px; }
      thead th, tbody td { padding: 8px 10px; font-size: .78rem; }
    }
    .deck {
      display: flex;
      flex-direction: column;
      gap: 16px;
      margin-top: 14px;
    }
    .deck-card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
      overflow: hidden;
      transition: border-color .2s ease, box-shadow .2s ease;
    }
    .deck-card.open {
      border-color: rgba(105, 87, 245, 0.4);
    }
    .deck-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 16px 22px;
      cursor: pointer;
      user-select: none;
      background: var(--panel-bg);
      border-bottom: 1px solid transparent;
      transition: background .2s ease;
      gap: 12px;
    }
    .deck-card.open .deck-header {
      border-bottom: 1px solid var(--line);
    }
    .deck-header:hover {
      background: rgba(105, 87, 245, 0.05);
    }
    .deck-title-group {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }
    .deck-icon-badge {
      width: 38px;
      height: 38px;
      border-radius: 12px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 1.25rem;
      background: var(--pill-bg);
      border: 1px solid var(--pill-border);
      flex-shrink: 0;
    }
    .deck-title {
      font-size: 1.15rem;
      font-weight: 800;
      color: var(--text);
      letter-spacing: -0.02em;
      margin: 0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .deck-subtitle {
      font-size: 0.8rem;
      color: var(--muted);
      margin-top: 2px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .deck-meta-group {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-shrink: 0;
    }
    .deck-badge {
      font-size: 0.78rem;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 999px;
      background: var(--pill-bg);
      color: var(--primary);
      border: 1px solid var(--pill-border);
      white-space: nowrap;
    }
    .deck-badge.success {
      background: rgba(29, 191, 115, 0.1);
      color: #0f8d56;
      border-color: rgba(29, 191, 115, 0.25);
    }
    .deck-badge.muted {
      background: rgba(110, 122, 166, 0.08);
      color: var(--muted);
      border-color: var(--line);
    }
    .deck-chevron {
      width: 30px;
      height: 30px;
      border-radius: 8px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: var(--card-solid);
      border: 1px solid var(--line);
      font-size: 0.75rem;
      color: var(--muted);
      transition: transform .25s ease;
    }
    .deck-card.open .deck-chevron {
      transform: rotate(180deg);
      color: var(--primary);
    }
    .deck-body {
      display: none;
      padding: 22px;
      animation: deckFade .22s ease-in-out;
    }
    .deck-card.open .deck-body {
      display: block;
    }
    @keyframes deckFade {
      from { opacity: 0; transform: translateY(-4px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .empty-state {
      background: var(--card);
      border: 1px dashed var(--line);
      border-radius: 20px;
      padding: 40px 20px;
      text-align: center;
      margin-top: 6px;
    }
    .empty-state h3 { margin: 8px 0 4px; font-size: 1.15rem; font-weight: 800; color: var(--text); }
    .empty-state p { color: var(--muted); font-size: .88rem; max-width: 440px; margin: 0 auto; line-height: 1.5; }
    @media (max-width: 640px) {
      .deck-header { padding: 13px 14px; }
      .deck-icon-badge { width: 32px; height: 32px; font-size: 1.1rem; }
      .deck-title { font-size: 1rem; }
      .deck-subtitle { display: none; }
      .deck-body { padding: 14px 12px; }
    }
    .footer {
      margin-top: 40px;
      padding: 24px 16px 14px;
      border-top: 1px solid var(--line);
      text-align: center;
      color: var(--muted);
      font-size: .88rem;
    }
    .footer strong {
      color: var(--text);
      font-weight: 700;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="topbar">
      <div class="brand">AutoML Studio</div>
      <nav id="topNav">
        <button class="nav-btn active" type="button" onclick="navigateTo('card1', this)">Dataset</button>
        <button class="nav-btn" type="button" onclick="navigateTo('card2', this)">Leaderboard</button>
        <button class="nav-btn" type="button" onclick="navigateTo('card3', this)">Insights</button>
        <button class="nav-btn" type="button" onclick="navigateTo('card4', this)">Predict</button>
      </nav>
      <button id="themeToggle" type="button" onclick="toggleTheme()" aria-label="Toggle Dark/Light Mode" style="border:1px solid var(--line); background:var(--card-solid); color:var(--text); padding:7px 12px; border-radius:10px; font-weight:700; font-size:.82rem; cursor:pointer; display:inline-flex; align-items:center; gap:6px; transition:.2s ease; white-space:nowrap;">
        <span id="themeIcon">🌙</span> <span id="themeText">Dark</span>
      </button>
    </header>

    <div class="deck">
      <!-- CARD 1: DATASET & TRAINING -->
      <div class="deck-card open" id="card1">
        <div class="deck-header" onclick="toggleCard('card1')">
          <div class="deck-title-group">
            <span class="deck-icon-badge">📂</span>
            <div>
              <h2 class="deck-title">1. Dataset & AutoML Pipeline</h2>
              <div class="deck-subtitle">Upload dataset, configure target column, and start automated training</div>
            </div>
          </div>
          <div class="deck-meta-group">
            <span class="deck-badge" id="card1Badge">Ready</span>
            <span class="deck-chevron">▼</span>
          </div>
        </div>
        <div class="deck-body">
          <section class="hero" id="uploadSection" style="margin-top:0;">
            <div>
              <span class="tag">Python AutoML Pipeline</span>
              <h1>Train smarter. Predict faster.</h1>
              <div class="lead">Upload a dataset, auto-train multiple models, then predict instantly using the saved artifact.</div>
              <div class="pills">
                <div class="pill">Progress tracking</div>
                <div class="pill">Model download</div>
                <div class="pill">Saved model</div>
                <div class="pill">Instant prediction</div>
              </div>
            </div>

            <div class="panel">
              <h3>Dataset file</h3>
              <div class="field">
                <input id="fileInput" type="file" accept=".csv,.xlsx,.xls,.json,text/csv,text/plain,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/json" style="width:100%; max-width:100%; padding:10px 12px; border:1px solid var(--line); border-radius:12px; background:var(--input-bg); color:var(--text); font-size:.88rem; cursor:pointer;" />
                <div style="margin-top:10px; display:flex; gap:6px; flex-wrap:wrap; align-items:center;">
                  <span style="font-size:.78rem; font-weight:700; color:var(--muted);">Try demo:</span>
                  <button type="button" class="pill" onclick="loadDemo('titanic')" style="cursor:pointer; border:1px solid var(--line); font-size:.75rem; padding:4px 10px; background:var(--card);">Titanic Survival</button>
                </div>
                <div id="fileNotice" style="margin-top:8px; padding:10px 14px; border-radius:10px; font-size:.9rem; font-weight:700; background:var(--pill-bg); display:none;"></div>
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

          <div class="section" id="previewSection" style="margin-top:18px; display:none;">
            <h4>Dataset preview</h4>
            <h2>First five rows</h2>
            <div class="preview-wrap"><table id="previewTable"></table></div>
          </div>
        </div>
      </div>

      <!-- CARD 2: RESULTS & LEADERBOARD -->
      <div class="deck-card" id="card2">
        <div class="deck-header" onclick="toggleCard('card2')">
          <div class="deck-title-group">
            <span class="deck-icon-badge">🏆</span>
            <div>
              <h2 class="deck-title">2. Model Leaderboard & Comparisons</h2>
              <div class="deck-subtitle">Evaluation metrics (PR-AUC, ROC-AUC, F1), rankings, and export</div>
            </div>
          </div>
          <div class="deck-meta-group">
            <span class="deck-badge muted" id="card2Badge">Waiting for training</span>
            <span class="deck-chevron">▼</span>
          </div>
        </div>
        <div class="deck-body">
          <div id="resultsContent" style="display:none;">
            <section class="metrics" id="metrics" style="margin-top:0;"></section>

            <div class="section" style="margin-top:18px;">
              <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                <div>
                  <h4>Comprehensive evaluation</h4>
                  <h2 style="margin:0;">All Models Performance Leaderboard</h2>
                </div>
                <button type="button" onclick="exportLeaderboardCSV()" class="nav-btn" style="border:1px solid var(--line); background:var(--card-solid); color:var(--primary); font-size:.82rem; font-weight:700; padding:6px 14px; cursor:pointer; display:inline-flex; align-items:center; gap:6px;" title="Export table data to CSV file">
                  📥 Export CSV
                </button>
              </div>
              <div class="tiny" style="margin:8px 0 10px;">Detailed comparison of all trained machine learning models across evaluation metrics (including Accuracy, F1, ROC-AUC, and PR-AUC).</div>
              <div class="table-wrap" style="height:auto; max-height:420px;"><table id="comparisonTable"></table></div>
            </div>

            <div class="section" style="margin-top:18px;">
              <h4>Model comparison</h4>
              <h2>Top 3 model comparison</h2>
              <div class="bar-list" id="comparisonBars"></div>
            </div>

            <div class="section" id="downloadSection" style="margin:14px 0 0; padding:14px 18px; background:linear-gradient(135deg, rgba(105,87,245,0.06), rgba(142,123,255,0.1)); border:1px solid var(--line); border-radius:14px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
              <div>
                <div style="font-weight:800; font-size:.95rem; color:var(--text); display:flex; align-items:center; gap:6px;">
                  <span>📦</span> <strong>Export Best Model</strong>
                </div>
                <div style="font-size:.8rem; color:var(--muted); margin-top:2px;">Download serialized Python pickle (<code style="font-size:.78rem; background:var(--code-bg); padding:1px 5px; border-radius:4px;">.pkl</code>) for offline inference.</div>
              </div>
              <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
                <button type="button" onclick="navigateTo('card4')" class="nav-btn" style="background:var(--pill-bg); color:var(--primary); font-size:.85rem; font-weight:700; padding:8px 16px; border-radius:10px; cursor:pointer;">
                  🎯 Test Predictions &rarr;
                </button>
                <a href="/api/download-model" class="primary-btn" style="width:auto; margin:0; text-decoration:none; padding:9px 18px; font-size:.88rem; display:inline-flex; align-items:center; justify-content:center; gap:6px; border-radius:10px; white-space:nowrap;">
                  📥 Download Model (.pkl)
                </a>
              </div>
            </div>
          </div>

          <div id="resultsEmpty" class="empty-state">
            <div style="font-size:2.8rem; margin-bottom:8px;">🏆</div>
            <h3>No Models Trained Yet</h3>
            <p>Upload a dataset and click <strong>Run AutoML Pipeline</strong> in Section 1 to view model comparisons and leaderboard rankings.</p>
            <button type="button" class="primary-btn" onclick="navigateTo('card1')" style="width:auto; padding:10px 24px; font-size:.92rem; margin-top:14px; display:inline-block;">🚀 Go to Dataset & Train</button>
          </div>
        </div>
      </div>

      <!-- CARD 3: DETAILS & EDA -->
      <div class="deck-card" id="card3">
        <div class="deck-header" onclick="toggleCard('card3')">
          <div class="deck-title-group">
            <span class="deck-icon-badge">📊</span>
            <div>
              <h2 class="deck-title">3. Insights & Exploratory Data Analysis</h2>
              <div class="deck-subtitle">Feature importances, target distributions, data health & tuning logs</div>
            </div>
          </div>
          <div class="deck-meta-group">
            <span class="deck-badge muted" id="card3Badge">Insights ready after train</span>
            <span class="deck-chevron">▼</span>
          </div>
        </div>
        <div class="deck-body">
          <div id="detailsContent" style="display:none;">
            <div class="section" id="edaSection" style="display:none; margin-bottom:16px;">
              <h4>Exploratory Data Analysis</h4>
              <h2>Dataset Insights & Target Distribution</h2>
              <div class="results-columns" style="margin-top:10px;">
                <div class="results-col">
                  <div style="background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px;">
                    <h4 style="margin:0 0 8px; color:var(--primary);">🎯 Target Class Distribution</h4>
                    <div id="targetDistChart"></div>
                  </div>
                </div>
                <div class="results-col">
                  <div style="background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px;">
                    <h4 style="margin:0 0 8px; color:var(--primary);">🩺 Data Health & Missing Values</h4>
                    <div id="missingDistChart"></div>
                  </div>
                </div>
              </div>
            </div>

            <div class="results-columns" id="detailsSection">
              <div class="results-col">
                <div class="section">
                  <h4>Model interpretability</h4>
                  <h2>Feature importance highlights</h2>
                  <div id="importanceBox"></div>
                </div>
              </div>

              <div class="results-col">
                <div class="section">
                  <h4>Pipeline steps</h4>
                  <h2>Data preprocessing summary</h2>
                  <div class="accordion" id="prepAccordion"></div>
                </div>
                <div class="section" style="margin-top:14px;">
                  <h4>Hyperparameter tuning</h4>
                  <h2>Best tuned models</h2>
                  <div id="tuningBox"></div>
                </div>
              </div>
            </div>
          </div>

          <div id="detailsEmpty" class="empty-state">
            <div style="font-size:2.8rem; margin-bottom:8px;">🔍</div>
            <h3>No Details Available</h3>
            <p>Run the AutoML pipeline to inspect EDA visualizations, feature importance rankings, and hyperparameter tuning logs.</p>
            <button type="button" class="primary-btn" onclick="navigateTo('card1')" style="width:auto; padding:10px 24px; font-size:.92rem; margin-top:14px; display:inline-block;">🚀 Go to Dataset & Train</button>
          </div>
        </div>
      </div>

      <!-- CARD 4: PREDICT -->
      <div class="deck-card" id="card4">
        <div class="deck-header" onclick="toggleCard('card4')">
          <div class="deck-title-group">
            <span class="deck-icon-badge">🎯</span>
            <div>
              <h2 class="deck-title">4. Live Model Predictions</h2>
              <div class="deck-subtitle">Test trained model with instant interactive inputs & class probabilities</div>
            </div>
          </div>
          <div class="deck-meta-group">
            <span class="deck-badge muted" id="card4Badge">Model not loaded</span>
            <span class="deck-chevron">▼</span>
          </div>
        </div>
        <div class="deck-body">
          <form id="featureForm" style="display:none; margin-top:0;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:14px;">
              <div>
                <h3 style="margin:0;">Instant Model Prediction</h3>
                <div class="tiny" style="margin-top:4px;">Test your trained model immediately with new feature inputs.</div>
              </div>
              <button type="button" onclick="autofillSampleValues()" class="nav-btn" style="border:1px solid var(--line); background:var(--card-solid); color:var(--primary); font-size:.85rem; font-weight:700; padding:7px 14px; cursor:pointer;" title="Fill input fields with sample values from the dataset">
                🎲 Autofill Sample Values
              </button>
            </div>
            <div class="grid" id="featureGrid"></div>
            <button class="primary-btn" id="predictBtn" type="submit">Predict</button>
            <div id="predictionResult" style="display:none; margin-top:16px; padding:16px 20px; border-radius:14px; border:1px solid var(--line); background:var(--card); text-align:center;"></div>
          </form>

          <div id="predictEmpty" class="empty-state">
            <div style="font-size:2.8rem; margin-bottom:8px;">🎯</div>
            <h3>Predictor Not Ready</h3>
            <p>Please train a model first in Section 1. Once trained, your model will be loaded here for live predictions.</p>
            <button type="button" class="primary-btn" onclick="navigateTo('card1')" style="width:auto; padding:10px 24px; font-size:.92rem; margin-top:14px; display:inline-block;">🚀 Go to Dataset & Train</button>
          </div>
        </div>
      </div>
    </div>

    <footer class="footer">
      <div>Made by <strong>N Samuel Reddy</strong></div>
    </footer>
  </div>

  <script>
    function initTheme() {
      const saved = localStorage.getItem('theme');
      const theme = saved || 'light';
      applyTheme(theme);
    }
    function applyTheme(theme) {
      document.documentElement.setAttribute('data-theme', theme);
      localStorage.setItem('theme', theme);
      const icon = document.getElementById('themeIcon');
      const text = document.getElementById('themeText');
      if (icon) icon.textContent = theme === 'dark' ? '☀️' : '🌙';
      if (text) text.textContent = theme === 'dark' ? 'Light' : 'Dark';
    }
    function toggleTheme() {
      const current = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      applyTheme(current);
    }
    initTheme();

    function autofillSampleValues() {
      const preview = state.result?.dataset?.preview;
      if (!preview || !preview.length) {
        alert('No dataset preview available to autofill from. Please run AutoML pipeline first.');
        return;
      }
      const sample = preview[0];
      for (const el of document.querySelectorAll('#featureGrid input, #featureGrid select')) {
        if (sample[el.name] !== undefined && sample[el.name] !== null) {
          el.value = sample[el.name];
        }
      }
    }

    function exportLeaderboardCSV() {
      const table = document.getElementById('comparisonTable');
      if (!table || !table.rows || !table.rows.length) {
        alert('No leaderboard data available to export.');
        return;
      }
      const lines = [];
      for (const row of table.rows) {
        const rowVals = Array.from(row.cells).map(cell => `"${(cell.innerText || '').trim().replace(/"/g, '""')}"`);
        lines.push(rowVals.join(','));
      }
      const blob = new Blob([lines.join(String.fromCharCode(10))], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'model_leaderboard.csv';
      link.click();
    }

    async function loadDemo(name) {
      if (fileNotice) {
        fileNotice.style.display = 'block';
        fileNotice.style.background = 'var(--pill-bg)';
        fileNotice.style.color = 'var(--primary)';
        fileNotice.innerHTML = 'Loading demo dataset...';
      }
      try {
        const res = await fetch('/api/load-demo?name=' + encodeURIComponent(name));
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || 'Failed to load demo dataset');
        }
        const data = await res.json();
        state.file = null;
        state.file_id = data.file_id;
        state.uploadPromise = Promise.resolve(data.file_id);
        fileInput.value = '';
        const cols = data.columns || [];
        targetInput.innerHTML = cols.map(c => `<option value="${c}"${c === data.default_target ? ' selected' : ''}>${c}</option>`).join('');
        if (data.default_target) targetInput.value = data.default_target;
        if (fileNotice) {
          fileNotice.style.display = 'block';
          fileNotice.style.background = 'rgba(29,191,115,0.08)';
          fileNotice.style.color = '#0f8d56';
          fileNotice.innerHTML = '⚡ <strong>' + data.label + ' demo loaded!</strong> Target <code>' + data.default_target + '</code> selected. Click <strong>Run AutoML Pipeline</strong> to start.';
        }
      } catch (e) {
        if (fileNotice) {
          fileNotice.style.display = 'block';
          fileNotice.style.background = 'rgba(239,68,68,0.1)';
          fileNotice.style.color = '#dc2626';
          fileNotice.innerHTML = '❌ <strong>Error:</strong> ' + e.message;
        }
      }
    }

    function toggleCard(cardId, forceOpen) {
      const card = document.getElementById(cardId);
      if (!card) return;
      const isCurrentlyOpen = card.classList.contains('open');
      const shouldOpen = (forceOpen !== undefined) ? forceOpen : !isCurrentlyOpen;
      if (shouldOpen) {
        card.classList.add('open');
      } else {
        card.classList.remove('open');
      }
      updateNavHighlight();
    }

    function openCard(cardId, scroll) {
      toggleCard(cardId, true);
      if (scroll) {
        const card = document.getElementById(cardId);
        if (card) {
          setTimeout(() => {
            card.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }, 80);
        }
      }
    }

    function updateNavHighlight() {
      const navBtns = document.querySelectorAll('#topNav .nav-btn');
      const cards = ['card1', 'card2', 'card3', 'card4'];
      let activeCard = null;
      for (const cid of cards) {
        const el = document.getElementById(cid);
        if (el && el.classList.contains('open')) {
          activeCard = cid;
          break;
        }
      }
      navBtns.forEach((btn) => {
        const onclickAttr = btn.getAttribute('onclick') || '';
        if (activeCard && onclickAttr.indexOf("'" + activeCard + "'") !== -1) {
          btn.classList.add('active');
        } else {
          btn.classList.remove('active');
        }
      });
    }

    function navigateTo(target, btnEl) {
      const map = {
        'upload': 'card1', 'card1': 'card1',
        'results': 'card2', 'card2': 'card2',
        'insights': 'card3', 'card3': 'card3', 'details': 'card3',
        'predict': 'card4', 'card4': 'card4'
      };
      const cardId = map[target] || 'card1';
      openCard(cardId, true);
      if (btnEl) {
        document.querySelectorAll('#topNav .nav-btn').forEach(b => b.classList.remove('active'));
        btnEl.classList.add('active');
      }
    }

    const state = { result: null, file: null };

    const fileInput = document.getElementById('fileInput');
    const fileNotice = document.getElementById('fileNotice');
    const targetInput = document.getElementById('target');
    const metrics = document.getElementById('metrics');
    const results = document.getElementById('results');
    const previewTable = document.getElementById('previewTable');
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
      state.file_id = null;
      state.uploadPromise = null;

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

      if (state.currentXhr) {
        try { state.currentXhr.abort(); } catch (_) {}
      }

      const fd = new FormData();
      fd.append('file', file);

      state.uploadProgress = 0;
      state.uploadPromise = new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        state.currentXhr = xhr;
        xhr.open('POST', '/api/upload');
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const pct = Math.round((event.loaded / event.total) * 100);
            state.uploadProgress = pct;
            if (pct >= 100) {
              if (fileNotice && (!state.file_id)) {
                fileNotice.innerHTML = '⚡ <strong>' + file.name + ' transfer complete!</strong> Finalizing on server...';
              }
              if (state.isWaitingForUpload) {
                statusBox.innerHTML = 'Upload transfer complete (100%). Saving and initializing dataset on server...';
                progressBar.style.width = '15%';
              }
            } else {
              if (fileNotice && (!state.file_id)) {
                fileNotice.innerHTML = '⚡ <strong>' + file.name + ' ready!</strong> (' + (localCols ? localCols.length : '') + ' columns detected). Uploading: <strong>' + pct + '%</strong>';
              }
              if (state.isWaitingForUpload) {
                statusBox.innerHTML = 'Uploading dataset to server: <strong>' + pct + '%</strong>...';
                progressBar.style.width = Math.max(pct * 0.1, 4) + '%';
              }
            }
          }
        };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const data = JSON.parse(xhr.responseText);
              state.file_id = data.file_id;
              const serverCols = data.columns || [];
              if (serverCols.length > 0) {
                const currentVal = targetInput.value;
                targetInput.innerHTML = serverCols.map((c, index) => `<option value="${c}"${c === currentVal || (!currentVal && index === serverCols.length - 1) ? ' selected' : ''}>${c}</option>`).join('');
                if (currentVal && serverCols.includes(currentVal)) {
                  targetInput.value = currentVal;
                } else if (serverCols.length) {
                  targetInput.selectedIndex = serverCols.length - 1;
                }
              }
              if (fileNotice) {
                fileNotice.style.display = 'block';
                fileNotice.style.background = 'rgba(29,191,115,0.1)';
                fileNotice.style.color = '#0f8d56';
                fileNotice.innerHTML = '✓ <strong>' + (data.filename || file.name) + ' uploaded (100%)!</strong> (' + (serverCols.length || (localCols ? localCols.length : 0)) + ' columns ready).';
              }
              resolve(data.file_id);
            } catch (err) {
              reject(err);
            }
          } else {
            let detail = 'Upload failed';
            try { detail = JSON.parse(xhr.responseText).detail || detail; } catch (_) {}
            reject(new Error(detail));
          }
        };
        xhr.onerror = () => reject(new Error('Network error during upload.'));
        xhr.send(fd);
      }).catch((e) => {
        if (fileNotice) {
          fileNotice.style.display = 'block';
          fileNotice.style.background = 'rgba(239,68,68,0.1)';
          fileNotice.style.color = '#dc2626';
          fileNotice.innerHTML = '❌ <strong>Upload error:</strong> ' + e.message;
        }
        state.file_id = null;
        state.uploadPromise = null;
        throw e;
      });
    }

    fileInput.addEventListener('change', onFileSelected);

    trainBtn.addEventListener('click', async () => {
      let fileId = state.file_id;
      if (!fileId && state.uploadPromise) {
        state.isWaitingForUpload = true;
        statusBox.classList.add('show');
        statusBox.style.background = '';
        statusBox.style.color = '';
        statusBox.innerHTML = 'Uploading dataset to server: <strong>' + (state.uploadProgress || 0) + '%</strong>...';
        progressBar.style.width = Math.max((state.uploadProgress || 0) * 0.1, 4) + '%';
        try {
          fileId = await state.uploadPromise;
        } catch (e) {
          statusBox.innerHTML = '❌ <strong>Upload error:</strong> ' + e.message;
          statusBox.style.background = 'rgba(239,68,68,0.1)';
          statusBox.style.color = '#dc2626';
          return;
        } finally {
          state.isWaitingForUpload = false;
        }
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

      state.result = null;
      statusBox.classList.add('show');
      statusBox.style.background = '';
      statusBox.style.color = '';
      statusBox.innerHTML = 'Starting pipeline...';
      progressBar.style.width = '4%';

      const b1 = document.getElementById('card1Badge');
      if (b1) { b1.textContent = 'Training...'; b1.className = 'deck-badge'; }

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
          es.close();
          const payload = JSON.parse(e.data);
          state.result = payload;
          renderPreview(payload.dataset?.preview || []);
          renderMetrics(payload.summary || payload.dashboard_summary || {});
          renderResults(payload);
          renderFeatureFields(payload.feature_schema || payload.selected_feature_names || payload.feature_names || []);
          statusBox.innerHTML = '✓ Pipeline completed successfully. <span id="statusPct">100%</span>';
          statusBox.style.background = 'rgba(29,191,115,0.1)';
          statusBox.style.color = '#0f8d56';
          progressBar.style.width = '100%';

          const b1 = document.getElementById('card1Badge');
          if (b1) { b1.textContent = 'Completed ✓'; b1.className = 'deck-badge success'; }
          const b2 = document.getElementById('card2Badge');
          if (b2) { b2.textContent = 'Leaderboard Ready ✓'; b2.className = 'deck-badge success'; }
          const b3 = document.getElementById('card3Badge');
          if (b3) { b3.textContent = 'Insights Ready ✓'; b3.className = 'deck-badge success'; }
          const b4 = document.getElementById('card4Badge');
          if (b4) { b4.textContent = 'Predictor Active ✓'; b4.className = 'deck-badge success'; }

          setTimeout(() => {
            toggleCard('card1', false);
            openCard('card2', true);
          }, 450);
        } catch (err) {
          console.error('Bad result event', err);
        }
      });
      es.addEventListener('error', (e) => {
        es.close();
        if (state.result) return;
        const b1 = document.getElementById('card1Badge');
        if (b1) { b1.textContent = 'Failed ❌'; b1.className = 'deck-badge'; }
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
      });
    });

    featureForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const payload = {};
      for (const el of document.querySelectorAll('#featureGrid input, #featureGrid select')) {
        payload[el.name] = el.value;
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
      const predLabel = data?.prediction_label ?? prediction;
      if (prediction === undefined || prediction === null) {
        alert('Prediction failed: the backend did not return a valid prediction value.');
        return;
      }

      const resBox = document.getElementById('predictionResult');
      if (resBox) {
        resBox.style.display = 'block';
        let html = `<div style="font-size:1.25rem; font-weight:800; color:var(--primary);">🎯 Predicted Result: <span style="color:var(--text);">${predLabel}</span></div>`;
        if (data?.probabilities && Object.keys(data.probabilities).length) {
          html += `<div style="margin-top:12px; display:flex; gap:8px; justify-content:center; flex-wrap:wrap;">`;
          for (const [cls, prob] of Object.entries(data.probabilities)) {
            const pct = (Number(prob) * 100).toFixed(1);
            html += `<span class="pill" style="padding:5px 12px; font-size:.8rem;">Class <strong>${cls}</strong>: ${pct}%</span>`;
          }
          html += `</div>`;
        }
        resBox.innerHTML = html;
        resBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        alert('🎯 Prediction: ' + predLabel);
      }
    });

    function renderMetrics(summary) {
      const missingValueLabel = Number(summary.missing_values || 0) === 0 ? 'No missing values' : (summary.missing_values || 0);
      const metricValue = formatMetric(summary.best_metric_value ?? summary.best_metric ?? '—', summary.best_metric_label);
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
      const rc = document.getElementById('resultsContent');
      if (rc) rc.style.display = 'block';
      const re = document.getElementById('resultsEmpty');
      if (re) re.style.display = 'none';

      const pe = document.getElementById('predictEmpty');
      if (pe) pe.style.display = 'none';

      const dc = document.getElementById('detailsContent');
      if (dc) dc.style.display = 'block';
      const de = document.getElementById('detailsEmpty');
      if (de) de.style.display = 'none';

      const preprocessing = payload.preprocessing || {};
      const droppedReasons = payload.dataset?.dropped_column_reasons || {};
      prepAccordion.innerHTML = [
        ['Dropped columns', payload.dataset?.dropped_columns || []],
        ['Dropped column reasons', droppedReasons],
        ['Missing values filled', Object.keys(preprocessing.missing_value_report || {}).length ? preprocessing.missing_value_report : 'No missing values'],
        ['Encoding applied', preprocessing.encoding_report || {}],
      ].map(([title, data]) => `<details><summary>${title}</summary><div class="body">${formatBlock(data)}</div></details>`).join('');

      const allComparison = payload.comparison || [];
      const metric = payload.primary_metric || (payload.problem_type === 'Regression' ? 'R2' : 'Accuracy');
      const topComparison = allComparison.slice(0, 3);
      const maxValue = Math.max(...topComparison.map((row) => Number(row[metric]) || 0), 1);
      comparisonBars.innerHTML = topComparison.map((row) => `<div class="bar-row"><div>${row.Model}</div><div class="bar-track"><div class="bar-fill" style="width:${Math.max(((Number(row[metric]) || 0) / maxValue) * 100, 4)}%"></div></div><div>${formatMetric(row[metric], metric)}</div></div>`).join('');

      if (allComparison.length) {
        const cols = Object.keys(allComparison[0]);
        comparisonTable.innerHTML = `<thead><tr>${cols.map((k) => `<th>${k}</th>`).join('')}</tr></thead><tbody>${allComparison.map((row, idx) => `<tr style="${idx === 0 ? 'font-weight:700; background:rgba(29,191,115,0.06);' : ''}">${cols.map((k) => `<td>${k === 'Model' ? (idx === 0 ? '🏆 <strong>' + row[k] + '</strong>' : row[k]) : formatMetric(row[k], k)}</td>`).join('')}</tr>`).join('')}</tbody>`;
      } else {
        comparisonTable.innerHTML = '<tbody><tr><td class="tiny">No model comparisons available.</td></tr></tbody>';
      }

      renderEda(payload.eda);

      const importance = payload.feature_importance || {};
      const options = Object.keys(importance);
      if (!options.length) {
        importanceBox.innerHTML = '<div class="tiny">No feature-importance data available.</div>';
      } else {
        const firstModel = options[0];
        importanceBox.innerHTML = `<div class="field"><label class="label">Model</label><select id="importanceModel">${options.map((name) => `<option value="${name}">${name}</option>`).join('')}</select></div><div id="importanceList"></div>`;
        const drawImportance = (name) => {
          const rows = (importance[name] || []).slice(0, 10);
          document.getElementById('importanceList').innerHTML = rows.length
            ? `<div class="bar-list">${rows.map((row) => `<div class="bar-row"><div>${row.feature}</div><div class="bar-track"><div class="bar-fill" style="width:${Math.max((Number(row.importance) || 0) * 100, 4)}%"></div></div><div>${(Number(row.importance) * 100 || 0).toFixed(1)}%</div></div>`).join('')}</div>`
            : '<div class="tiny">No feature-importance data for this model.</div>';
        };
        drawImportance(firstModel);
        document.getElementById('importanceModel').onchange = (e) => drawImportance(e.target.value);
      }

      const tuningSummary = payload.tuning_summary || {};
      const tunedModels = Object.entries(tuningSummary)
        .filter(([, info]) => info && info.was_tuned)
        .slice(0, 3);
      tuningBox.innerHTML = tunedModels.length
        ? `<div class="tune-list">${tunedModels.map(([name, info]) => `<div class="tune-card"><div class="title">${name}</div><div class="meta">Baseline CV: ${formatMetric(info.baseline_cv_score)}<br/>Tuned CV: ${formatMetric(info.tuned_cv_score)}<br/>Best params: ${formatBlock(info.best_params || 'No tuned params')}</div></div>`).join('')}</div>`
        : '<div class="tiny">No tuned models were retained.</div>';
    }

    function renderEda(eda) {
      const edaSection = document.getElementById('edaSection');
      if (!edaSection) return;
      if (!eda || (!eda.target_distribution && !eda.missing_distribution)) {
        edaSection.style.display = 'none';
        return;
      }
      edaSection.style.display = 'block';

      const targetDistChart = document.getElementById('targetDistChart');
      const targetData = eda.target_distribution || {};
      const targetEntries = Object.entries(targetData);
      if (!targetEntries.length) {
        targetDistChart.innerHTML = '<div class="tiny">No target distribution data available.</div>';
      } else {
        const totalCount = targetEntries.reduce((sum, [, c]) => sum + Number(c), 0) || 1;
        const maxCount = Math.max(...targetEntries.map(([, c]) => Number(c)), 1);
        const dominantRatio = maxCount / totalCount;
        let imbalanceAlert = '';
        if (targetEntries.length >= 2 && dominantRatio >= 0.75) {
          const dominantEntry = targetEntries.find(([, c]) => Number(c) === maxCount);
          const domPct = (dominantRatio * 100).toFixed(1);
          imbalanceAlert = `
            <div style="margin-bottom:10px; padding:8px 12px; border-radius:10px; background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.3); color:#d97706; font-size:.8rem; font-weight:600; display:flex; align-items:center; gap:6px;">
              <span>⚠️</span>
              <span><strong>Imbalanced Target:</strong> Class "${dominantEntry ? dominantEntry[0] : ''}" is ${domPct}%. F1-Score & ROC-AUC are prioritized over Accuracy.</span>
            </div>
          `;
        }
        targetDistChart.innerHTML = `
          ${imbalanceAlert}
          <div style="font-size:.82rem; color:var(--muted); margin-bottom:10px;">Total labeled samples: <strong>${totalCount.toLocaleString()}</strong></div>
          <div class="bar-list">
            ${targetEntries.map(([label, count]) => {
              const pct = ((Number(count) / totalCount) * 100).toFixed(1);
              const barWidth = Math.max((Number(count) / maxCount) * 100, 6);
              return `
                <div class="bar-row">
                  <div style="font-weight:600;" title="${label}">${label}</div>
                  <div class="bar-track"><div class="bar-fill" style="width:${barWidth}%;"></div></div>
                  <div><strong>${pct}%</strong> <span style="color:var(--muted); font-size:.78rem;">(${Number(count).toLocaleString()})</span></div>
                </div>
              `;
            }).join('')}
          </div>
        `;
      }

      const missingDistChart = document.getElementById('missingDistChart');
      const missingList = eda.missing_distribution || [];
      const totalMissing = eda.total_missing || 0;
      const numNum = eda.num_numerical_cols || 0;
      const numCat = eda.num_categorical_cols || 0;

      let compositionHtml = `
        <div style="display:flex; gap:10px; margin-bottom:12px; flex-wrap:wrap;">
          <div style="background:rgba(105,87,245,0.08); padding:6px 12px; border-radius:10px; font-size:.82rem;">
            🔢 Numerical: <strong>${numNum}</strong>
          </div>
          <div style="background:rgba(29,191,115,0.08); padding:6px 12px; border-radius:10px; font-size:.82rem;">
            🔤 Categorical: <strong>${numCat}</strong>
          </div>
          <div style="background:${totalMissing === 0 ? 'rgba(29,191,115,0.08)' : 'rgba(239,68,68,0.08)'}; padding:6px 12px; border-radius:10px; font-size:.82rem; color:${totalMissing === 0 ? '#0f8d56' : '#dc2626'};">
            ${totalMissing === 0 ? '✓ Zero Missing Values' : `⚠️ ${totalMissing} Missing Values`}
          </div>
        </div>
      `;

      if (missingList.length === 0) {
        missingDistChart.innerHTML = compositionHtml + `
          <div style="padding:14px; background:rgba(29,191,115,0.06); border:1px solid rgba(29,191,115,0.2); border-radius:10px; color:#0f8d56; font-size:.86rem; text-align:center;">
            ✨ <strong>100% Complete Data!</strong> No missing values detected in any feature.
          </div>
        `;
      } else {
        missingDistChart.innerHTML = compositionHtml + `
          <div style="font-size:.82rem; color:var(--muted); margin-bottom:6px;">Top columns with missing data:</div>
          <div class="bar-list">
            ${missingList.map((m) => `
              <div class="bar-row">
                <div title="${m.column}">${m.column}</div>
                <div class="bar-track"><div class="bar-fill" style="width:${Math.max(m.percentage, 4)}%; background:linear-gradient(90deg, #f59e0b, #ef4444);"></div></div>
                <div><span style="color:#dc2626; font-weight:600;">${m.percentage}%</span> <span style="color:var(--muted); font-size:.78rem;">(${m.missing_count})</span></div>
              </div>
            `).join('')}
          </div>
        `;
      }
    }

    function formatBlock(value) {
      if (!value || (Array.isArray(value) && !value.length)) return '<div class="tiny">None</div>';
      if (Array.isArray(value)) return value.map((item) => `<div>${item}</div>`).join('');
      if (typeof value === 'object') return Object.entries(value).map(([k, v]) => `<div><strong>${k}</strong>: ${v}</div>`).join('');
      return `<div>${value}</div>`;
    }

    function renderPreview(rows) {
      const ps = document.getElementById('previewSection');
      if (ps) ps.style.display = (rows && rows.length) ? 'block' : 'none';
      if (!rows || !rows.length) {
        previewTable.innerHTML = '<tbody><tr><td class="tiny">No preview available.</td></tr></tbody>';
        return;
      }
      const columns = Object.keys(rows[0]);
      previewTable.innerHTML = `<thead><tr>${columns.map((col) => `<th>${col}</th>`).join('')}</tr></thead><tbody>${rows.map((row) => `<tr>${columns.map((col) => `<td>${row[col] ?? ''}</td>`).join('')}</tr>`).join('')}</tbody>`;
    }

    function formatMetric(value, key) {
      if (value === null || value === undefined || value === '—') return '—';
      const n = Number(value);
      if (isNaN(n)) return String(value);
      const k = String(key || '').toLowerCase();
      if (k.includes('acc') || k.includes('f1') || k.includes('prec') || k.includes('rec') || k.includes('auc') || k.includes('roc')) {
        return (n * 100).toFixed(2) + '%';
      }
      return Number.isInteger(n) ? String(n) : n.toFixed(4);
    }

    function renderFeatureFields(schemaOrNames) {
      if (!schemaOrNames || !schemaOrNames.length) return;
      featureGrid.innerHTML = schemaOrNames.map((item) => {
        const isObj = typeof item === 'object' && item !== null;
        const name = isObj ? item.name : item;
        const type = isObj ? item.type : 'number';
        const help = isObj && item.help ? `<div class="tiny" style="margin-top:6px; font-size:.76rem; color:var(--muted);">${item.help}</div>` : '';

        if (type === 'select' && item.options && item.options.length) {
          const defaultVal = item.default !== undefined ? item.default : item.options[0].value;
          const optionsHtml = item.options.map((opt) => `<option value="${opt.value}" ${opt.value === defaultVal ? 'selected' : ''}>${opt.label}</option>`).join('');
          return `
            <div class="input-wrap">
              <label for="field_${name}">${name}</label>
              <select id="field_${name}" name="${name}">
                ${optionsHtml}
              </select>
              ${help}
            </div>
          `;
        }

        const defaultVal = isObj && item.default !== undefined ? item.default : 0;
        const step = isObj && item.step !== undefined ? item.step : 'any';
        const minAttr = isObj && item.min !== undefined && item.min !== null ? `min="${item.min}"` : '';
        const maxAttr = isObj && item.max !== undefined && item.max !== null ? `max="${item.max}"` : '';

        return `
          <div class="input-wrap">
            <label for="field_${name}">${name}</label>
            <input id="field_${name}" name="${name}" type="number" value="${defaultVal}" step="${step}" ${minAttr} ${maxAttr} />
            ${help}
          </div>
        `;
      }).join('');
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
    with open(dest, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
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
    return JSONResponse({'file_id': file_id, 'filename': file.filename or 'data.csv', 'columns': columns})


DEMO_FILES = {
    'titanic': {'file': 'titanic_survival.csv', 'label': 'Titanic Survival'},
}


@app.get('/api/load-demo')
async def load_demo_dataset(name: str):
    info = DEMO_FILES.get(name.lower())
    if not info:
        return JSONResponse({'detail': f'Demo dataset "{name}" not found.'}, status_code=400)
    src = DEMO_DIR / info['file']
    if not src.exists():
        return JSONResponse({'detail': f'Demo file "{info["file"]}" is missing on server.'}, status_code=404)
    file_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{file_id}.csv"
    shutil.copyfile(src, dest)
    df = pd.read_csv(dest, nrows=5)
    cols = [str(c).strip() for c in df.columns]
    target = detect_default_target(cols)
    return JSONResponse({
        'file_id': file_id,
        'filename': info['file'],
        'columns': cols,
        'default_target': target,
        'label': info['label'],
    })


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


@app.post('/api/predict')
async def predict(payload: dict):
    model = MODEL_STATE.get('best_model_object')
    features = MODEL_STATE.get('selected_feature_names') or MODEL_STATE.get('feature_names') or []
    if model is None or not features:
        return JSONResponse({'detail': 'Train a model first.'}, status_code=400)

    scaling_params = MODEL_STATE.get('scaling_params') or {}
    row = {}
    for name in features:
        val = float(payload.get(name, 0))
        if name in scaling_params:
            mean = scaling_params[name].get('mean', 0.0)
            scale = scaling_params[name].get('scale', 1.0)
            if scale != 0:
                val = (val - mean) / scale
        row[name] = val

    df = pd.DataFrame([row], columns=features)
    raw_pred = model.predict(df)[0]
    is_regression = str(MODEL_STATE.get('problem_type', '')).lower() == 'regression'
    pred = float(raw_pred) if is_regression else int(raw_pred)

    probs = {}
    if hasattr(model, 'predict_proba') and not is_regression:
        try:
            classes = [str(c) for c in model.classes_.tolist()]
            vals = model.predict_proba(df)[0].tolist()
            probs = {cls: float(v) for cls, v in zip(classes, vals, strict=False)}
        except Exception:
            pass

    target_mapping = MODEL_STATE.get('target_label_mapping') or {}
    pred_label = target_mapping.get(str(pred), str(pred))
    return {'prediction': pred, 'prediction_label': pred_label, 'probabilities': probs}


@app.get('/api/download-model')
async def download_model():
    from fastapi.responses import Response
    import pickle

    model = MODEL_STATE.get('best_model_object')
    if model is None:
        return JSONResponse(
            {'detail': 'No trained model available to download. Please run the AutoML pipeline first.'},
            status_code=400,
        )

    best_name = str(MODEL_STATE.get('best_model_name', 'model')).replace(' ', '_').lower()
    return Response(
        content=pickle.dumps(model),
        media_type='application/octet-stream',
        headers={'Content-Disposition': f'attachment; filename="{best_name}.pkl"'},
    )



if __name__ == '__main__':
  import uvicorn
  uvicorn.run('main:app', host='0.0.0.0', port=int(os.getenv('PORT', 8080)), reload=False)