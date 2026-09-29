import os
import shutil
import uuid
import json
import threading
import queue
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
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
    :root { --bg: #edf2ff; --bg-gradient: linear-gradient(180deg, #edf2ff, #f8f9ff); --card: rgba(255, 255, 255, 0.85); --card-solid: #ffffff; --panel-bg: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(245,247,255,0.85)); --soft: #f5f7ff; --line: #dfe7ff; --text: #1f2a44; --muted: #6e7aa6; --primary: #6957f5; --primary-2: #8e7bff; --success: #1dbf73; --shadow: 0 20px 45px rgba(108, 92, 231, 0.12); --input-bg: rgba(255, 255, 255, 0.9); --table-bg: rgba(255, 255, 255, 0.85); --code-bg: rgba(245, 247, 255, 0.8); --pill-bg: rgba(105, 87, 245, 0.08); --pill-border: rgba(105, 87, 245, 0.14); }
    [data-theme="dark"] { --bg: #0b0f19; --bg-gradient: linear-gradient(180deg, #0b0f19, #131b2e); --card: rgba(21, 30, 48, 0.88); --card-solid: #151e30; --panel-bg: linear-gradient(180deg, rgba(21, 30, 48, 0.95), rgba(16, 23, 38, 0.9)); --soft: #19243a; --line: #263554; --text: #f1f5f9; --muted: #94a3b8; --primary: #818cf8; --primary-2: #a5b4fc; --success: #10b981; --shadow: 0 20px 45px rgba(0, 0, 0, 0.45); --input-bg: #19243a; --table-bg: rgba(21, 30, 48, 0.85); --code-bg: #101726; --pill-bg: rgba(129, 140, 248, 0.14); --pill-border: rgba(129, 140, 248, 0.25); }
    *, *::before, *::after { box-sizing: border-box; }
    html { width: 100%; max-width: 100vw; overflow-x: hidden; overflow-y: auto; -webkit-text-size-adjust: 100%; }
    body { margin: 0; padding: 0; width: 100%; max-width: 100vw; min-height: 100%; overflow-x: hidden; overflow-y: auto; font-family: Inter, Arial, sans-serif; background: var(--bg-gradient); color: var(--text); transition: background 0.25s ease, color 0.25s ease; -webkit-font-smoothing: antialiased; }
    
    /* Layout safety & word break across all mobile screen sizes */
    .wrap, .screen-view, .hero, .hero > div, .panel, .section, .card, .hub-banner, .hub-banner > div, .hub-card, .results-columns, .results-col, .metrics, .metric, .hub-grid, .hub-kpi-ribbon, .hub-kpi-pill, .field, .input-wrap, #featureForm, #predictionResult, .table-wrap, .preview-wrap, .spoke-bar, #downloadSection {
      min-width: 0;
      max-width: 100%;
      box-sizing: border-box;
    }
    h1, h2, h3, h4, p, .lead, .tiny, .status, .pill, #fileNotice, .hub-card-desc, .hub-card-title, .bar-row, tbody td, thead th, select, input, button, code, #hubDatasetMeta, .accordion summary, .accordion .body, .tune-card, #targetDistChart, #missingDistChart {
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    input[type="file"] {
      width: 100% !important;
      max-width: 100% !important;
      box-sizing: border-box !important;
      overflow: hidden !important;
      text-overflow: ellipsis !important;
      display: block;
    }
    
    .wrap { max-width: 1180px; margin: 18px auto; padding: 0 20px 40px; }
    .topbar { display: flex; align-items: center; justify-content: space-between; background: var(--card); border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow); padding: 12px 20px; margin-bottom: 20px; backdrop-filter: blur(10px); position: sticky; top: 12px; z-index: 100; gap: 12px; }
    .hero, .results, #featureForm, #metrics, .section, #detailsSection { scroll-margin-top: 90px; }
    .brand { font-size: 1.55rem; font-weight: 800; letter-spacing: -0.05em; color: var(--primary); white-space: nowrap; order: 1; }
    nav#topNav { display: flex; gap: 6px; order: 2; }
    .nav-btn { border: none; background: transparent; color: var(--muted); padding: 7px 12px; border-radius: 10px; font-weight: 700; cursor: pointer; transition: .2s ease; font-size: .88rem; white-space: nowrap; }
    .nav-btn.active, .nav-btn:hover { background: var(--pill-bg); color: var(--primary); }
    #themeToggle { order: 3; }
    .hero { display: grid; grid-template-columns: 1.3fr 0.9fr; gap: 26px; background: var(--card); border: 1px solid var(--line); border-radius: 28px; padding: 34px 30px; box-shadow: var(--shadow); }
    .tag { display: inline-flex; align-items: center; border-radius: 999px; padding: 7px 14px; font-size: 0.8rem; background: var(--pill-bg); color: var(--primary); font-weight: 700; border: 1px solid var(--pill-border); max-width: 100%; }
    h1 { font-size: clamp(2.2rem, 4.5vw, 4.5rem); line-height: 1.08; letter-spacing: -0.06em; margin: 18px 0 16px; }
    .lead { font-size: clamp(1rem, 1.8vw, 1.35rem); color: var(--muted); line-height: 1.45; font-weight: 500; }
    .pills { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }
    .pill { background: var(--pill-bg); color: var(--primary); padding: 9px 15px; border-radius: 999px; border: 1px solid var(--pill-border); font-weight: 700; font-size: .82rem; transition: transform .2s ease, box-shadow .2s ease; max-width: 100%; }
    .pill:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(105,87,245,0.12); }
    .panel { background: var(--panel-bg); border: 1px solid var(--line); border-radius: 24px; padding: 20px; box-shadow: var(--shadow); }
    .panel h3 { margin: 0 0 14px; font-size: 1.15rem; }
    .field { margin-top: 14px; }
    .label { display:block; font-size: .8rem; font-weight: 700; color: var(--muted); margin-bottom: 8px; }
    select, input[type=text] { width: 100%; padding: 14px 16px; border-radius: 12px; border: 1px solid var(--line); background: var(--input-bg); color: var(--text); font-size: 1rem; outline: none; transition: .2s ease; max-width: 100%; }
    select:focus, input[type=text]:focus { border-color: var(--primary); box-shadow: 0 0 0 4px rgba(105,87,245,0.1); }
    .tiny { color: var(--muted); font-size: .82rem; margin-top: 10px; }
    .primary-btn { width: 100%; margin-top: 16px; border: none; border-radius: 14px; padding: 16px 18px; cursor: pointer; font-size: 1.05rem; font-weight: 800; color: white; background: linear-gradient(135deg, var(--primary), var(--primary-2)); box-shadow: 0 16px 30px rgba(105,87,245,0.22); transition: transform .2s ease, box-shadow .2s ease; max-width: 100%; }
    .primary-btn:hover { transform: translateY(-2px); box-shadow: 0 18px 35px rgba(105,87,245,0.28); }
    .status { margin-top: 16px; background: rgba(29,191,115,0.08); color: #0f8d56; border: 1px solid rgba(29,191,115,0.25); border-radius: 12px; padding: 12px 14px; font-weight: 700; display:none; }
    .status.show { display:block; }
    .progress { height: 12px; border-radius: 999px; overflow: hidden; background: var(--pill-bg); margin-top: 8px; }
    .progress > span { display: block; height: 100%; width: 0; background: linear-gradient(90deg, var(--primary), var(--primary-2)); border-radius: 999px; transition: width .2s ease; }
    .metrics { display: grid; grid-template-columns: repeat(6, minmax(130px, 1fr)); gap: 14px; margin-top: 20px; }
    .metric { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 14px 16px; min-height: 100px; display:flex; flex-direction:column; justify-content:space-between; }
    .metric .k { font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800; line-height: 1.2; }
    .metric .v { font-size: clamp(1.2rem, 1.6vw, 1.8rem); font-weight: 800; letter-spacing: -0.05em; line-height: 1.15; margin-top: auto; padding-top: 6px; }
    .results { display:none; margin-top: 18px; gap: 14px; }
    .results.show { display:grid; }
    .results-columns { display:grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }
    .results-col { display:flex; flex-direction:column; gap: 14px; }
    #importanceList { max-height: 330px; overflow-y: auto; padding-right: 4px; }
    .section { background: var(--panel-bg); border: 1px solid var(--line); border-radius: 16px; padding: 16px 18px 14px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05); height: fit-content; }
    .section h2 { margin: 0 0 10px; font-size: 1.05rem; letter-spacing: -0.03em; }
    .section h4 { margin: 0 0 4px; color: var(--muted); font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .preview-wrap { margin-top: 10px; height: auto; max-height: 220px; overflow: auto; border: 1px solid var(--line); border-radius: 14px; -webkit-overflow-scrolling: touch; }
    .accordion details { border: 1px solid var(--line); border-radius: 12px; background: var(--card); margin-top: 8px; overflow: hidden; transition: border-color .2s ease, box-shadow .2s ease; }
    .accordion details[open] { border-color: var(--primary); box-shadow: 0 10px 24px rgba(105,87,245,0.06); }
    .accordion summary { cursor: pointer; list-style: none; padding: 12px 14px; font-weight: 700; display: flex; align-items: center; justify-content: space-between; gap: 12px; color: var(--text); }
    .accordion summary::-webkit-details-marker { display:none; }
    .accordion summary::after { content: '▾'; color: var(--primary); font-size: 1rem; transition: transform .2s ease; transform: rotate(0deg); }
    .accordion details[open] summary::after { transform: rotate(180deg); }
    .accordion .body { padding: 0 14px 12px; color: var(--muted); line-height: 1.6; }
    .bar-list, #comparisonBars { display:grid; gap: 10px; margin-top: 10px; }
    .bar-row { display:grid; grid-template-columns: minmax(110px, 1.2fr) 2fr 64px; gap: 10px; align-items:center; }
    .bar-row > div:first-child { overflow:hidden; text-overflow: ellipsis; white-space: nowrap; }
    .bar-track { height: 16px; border-radius: 999px; background: var(--pill-bg); overflow:hidden; }
    .bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--primary), var(--primary-2)); }
    .tune-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin-top: 10px; }
    .tune-card { border: 1px solid var(--line); border-radius: 14px; background: var(--card); padding: 14px 18px; margin-top: 0; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.03); }
    .tune-card .title { font-weight: 800; font-size: 1.05rem; color: var(--primary); margin-bottom: 6px; }
    .tune-card .meta { color: var(--muted); font-size: .85rem; line-height: 1.5; }
    .table-wrap { width: 100%; max-height: 380px; overflow: auto; border: 1px solid var(--line); border-radius: 14px; margin-top: 10px; background: var(--card-solid); box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03); -webkit-overflow-scrolling: touch; }
    table { width: 100%; min-width: 100%; border-collapse: collapse; background: var(--table-bg); color: var(--text); font-size: 0.88rem; text-align: left; }
    thead th { background: var(--soft); color: var(--text); font-weight: 700; font-size: 0.82rem; padding: 12px 16px; border-bottom: 2px solid var(--line); border-right: 1px solid var(--line); position: sticky; top: 0; z-index: 2; white-space: nowrap; }
    thead th:last-child { border-right: none; }
    tbody td { padding: 11px 16px; border-bottom: 1px solid var(--line); border-right: 1px solid var(--line); white-space: nowrap; font-size: 0.88rem; }
    tbody td:last-child { border-right: none; }
    tbody tr:last-child td { border-bottom: none; }
    tbody tr:nth-child(even) { background: rgba(105, 87, 245, 0.025); }
    [data-theme="dark"] tbody tr:nth-child(even) { background: rgba(255, 255, 255, 0.02); }
    tbody tr:hover { background: rgba(105, 87, 245, 0.06); }
    [data-theme="dark"] tbody tr:hover { background: rgba(129, 140, 248, 0.1); }
    #featureForm { display: none; margin-top: 28px; }
    #featureForm .grid { display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 16px; }
    .input-wrap { background: var(--card); border: 1px solid var(--line); padding: 14px; border-radius: 14px; }
    .input-wrap label { display:block; font-weight:700; margin-bottom:8px; color: var(--muted); }
    .input-wrap input, .input-wrap select { width: 100%; padding: 12px 14px; border-radius: 10px; border: 1px solid var(--line); background: var(--input-bg); color: var(--text); font-size: 0.95rem; font-family: inherit; outline: none; transition: .2s ease; max-width: 100%; }
    .input-wrap select { cursor: pointer; }
    .input-wrap input:focus, .input-wrap select:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(105,87,245,0.1); }
    .screen-view { display: none; animation: screenFade .22s ease-in-out; }
    .screen-view.active { display: block; }
    @keyframes screenFade { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    .hub-banner { background: linear-gradient(135deg, rgba(105, 87, 245, 0.08), rgba(142, 123, 255, 0.12)); border: 1px solid var(--line); border-radius: 20px; padding: 22px 24px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px; }
    .hub-kpi-ribbon { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 20px; }
    .hub-kpi-pill { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 12px 16px; display: flex; flex-direction: column; gap: 3px; box-shadow: var(--shadow); }
    .hub-kpi-pill .k { font-size: .68rem; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); font-weight: 700; }
    .hub-kpi-pill .v { font-size: 1.15rem; font-weight: 800; color: var(--text); overflow: hidden; text-overflow: ellipsis; }
    .hub-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
    .hub-card { background: var(--card); border: 1px solid var(--line); border-radius: 20px; padding: 22px; box-shadow: var(--shadow); cursor: pointer; user-select: none; transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease; display: flex; flex-direction: column; justify-content: space-between; gap: 14px; }
    .hub-card:hover { transform: translateY(-3px); border-color: rgba(105, 87, 245, 0.45); box-shadow: 0 16px 36px rgba(105, 87, 245, 0.16); }
    .hub-card-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
    .hub-card-title-group { display: flex; align-items: center; gap: 12px; }
    .hub-card-icon { width: 44px; height: 44px; border-radius: 14px; display: inline-flex; align-items: center; justify-content: center; font-size: 1.4rem; background: var(--pill-bg); border: 1px solid var(--pill-border); flex-shrink: 0; }
    .hub-card-title { font-size: 1.15rem; font-weight: 800; color: var(--text); margin: 0; }
    .hub-card-desc { font-size: .88rem; color: var(--muted); line-height: 1.45; margin: 0; }
    .hub-card-footer { display: flex; align-items: center; justify-content: space-between; border-top: 1px solid var(--line); padding-top: 14px; font-size: .85rem; font-weight: 700; color: var(--primary); }
    .hub-card-footer span.arrow { transition: transform .2s ease; }
    .hub-card:hover .hub-card-footer span.arrow { transform: translateX(4px); }
    .spoke-bar { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 18px; }
    .back-btn { border: 1px solid var(--line); background: var(--card-solid); color: var(--primary); padding: 8px 16px; border-radius: 10px; font-weight: 700; font-size: .85rem; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; transition: all .2s ease; max-width: 100%; }
    .back-btn:hover { background: var(--pill-bg); border-color: rgba(105, 87, 245, 0.4); transform: translateX(-2px); }
    .deck-badge { font-size: 0.78rem; font-weight: 700; padding: 4px 10px; border-radius: 999px; background: var(--pill-bg); color: var(--primary); border: 1px solid var(--pill-border); white-space: nowrap; }
    .deck-badge.success { background: rgba(29, 191, 115, 0.1); color: #0f8d56; border-color: rgba(29, 191, 115, 0.25); }
    .deck-badge.muted { background: rgba(110, 122, 166, 0.08); color: var(--muted); border-color: var(--line); }
    .footer { margin-top: 40px; padding: 24px 16px 14px; border-top: 1px solid var(--line); text-align: center; color: var(--muted); font-size: .88rem; }
    .footer strong { color: var(--text); font-weight: 700; }

    /* Responsive Breakpoints */
    @media (max-width: 980px) {
      .hero { grid-template-columns: 1fr; gap: 20px; }
      .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .results-columns { grid-template-columns: 1fr !important; }
      #featureForm .grid { grid-template-columns: 1fr; }
      .hub-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 768px) {
      .topbar { flex-wrap: wrap; padding: 8px 12px; border-radius: 14px; top: 6px; margin-bottom: 12px; gap: 6px 10px; }
      .brand { font-size: 1.22rem; order: 1; }
      #themeToggle { order: 2; padding: 5px 10px !important; font-size: .78rem !important; }
      nav#topNav { order: 3; width: 100%; display: flex; gap: 5px; overflow-x: auto; -webkit-overflow-scrolling: touch; scrollbar-width: none; border-top: 1px solid var(--line); padding-top: 7px; padding-bottom: 2px; }
      nav#topNav::-webkit-scrollbar { display: none; }
      .nav-btn { flex-shrink: 0; padding: 6px 11px; font-size: .8rem; text-align: center; white-space: nowrap; border-radius: 8px; }
      .hero, .results, #featureForm, #metrics, .section, #detailsSection { scroll-margin-top: 110px; }
      .hub-kpi-ribbon { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    }
    @media (max-width: 640px) {
      .wrap { padding: 0 10px 24px; margin: 4px auto; }
      .hero { padding: 18px 14px; border-radius: 18px; gap: 16px; }
      h1 { font-size: clamp(1.75rem, 6.5vw, 2.4rem); line-height: 1.15; margin: 10px 0 10px; letter-spacing: -0.04em; }
      .lead { font-size: .95rem; line-height: 1.45; }
      .pills { gap: 6px; margin-top: 12px; }
      .pill { padding: 6px 10px; font-size: .72rem; }
      .panel { padding: 14px; border-radius: 16px; }
      .panel h3 { font-size: 1.05rem; margin-bottom: 10px; }
      select, input[type=text] { padding: 11px 12px; font-size: .9rem; border-radius: 10px; }
      .primary-btn { padding: 13px 16px; font-size: .95rem; border-radius: 12px; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 14px; }
      .metric { padding: 10px 12px; min-height: 72px; border-radius: 12px; }
      .metric .k { font-size: .62rem; }
      .metric .v { font-size: 1.1rem; }
      .results { margin-top: 14px; gap: 12px; }
      .section { padding: 14px 12px; border-radius: 14px; }
      .section h2 { font-size: 1rem; margin-bottom: 6px; }
      .section h4 { font-size: .74rem; }
      .bar-row { grid-template-columns: minmax(70px, 1fr) 1.2fr 44px; gap: 6px; font-size: .76rem; }
      .bar-track { height: 12px; }
      .table-wrap, .preview-wrap { border-radius: 12px; margin-top: 8px; max-height: 320px; }
      table { min-width: 440px; }
      thead th, tbody td { padding: 8px 10px; font-size: .78rem; }
      .hub-banner { padding: 16px 14px; border-radius: 16px; gap: 12px; }
      .hub-banner h2 { font-size: 1.28rem !important; }
      .hub-banner .back-btn { width: 100%; justify-content: center; }
      .hub-kpi-ribbon { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-bottom: 14px; }
      .hub-kpi-pill { padding: 10px 12px; border-radius: 12px; }
      .hub-kpi-pill .v { font-size: 1.05rem; }
      .hub-grid { gap: 12px; }
      .hub-card { padding: 16px; border-radius: 16px; gap: 10px; }
      .hub-card-title { font-size: 1.05rem; }
      .hub-card-desc { font-size: .82rem; }
      .spoke-bar { gap: 8px; margin-bottom: 12px; }
      .spoke-bar .back-btn, .spoke-bar .nav-btn { padding: 6px 10px; font-size: .78rem; }
      #downloadSection { flex-direction: column; align-items: stretch !important; gap: 10px; padding: 12px !important; }
      #downloadSection select, #downloadSection a { width: 100% !important; justify-content: center; }
      #predictModelSelect { min-width: 0 !important; width: 100% !important; }
    }
    @media (max-width: 360px) {
      .wrap { padding: 0 6px 20px; }
      .hero { padding: 14px 10px; border-radius: 14px; }
      h1 { font-size: 1.55rem; }
      .panel { padding: 10px; }
      .hub-kpi-ribbon { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: 1fr; }
      .bar-row { grid-template-columns: minmax(60px, 1fr) 1fr 38px; font-size: .72rem; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="topbar">
      <div class="brand" onclick="showScreen('hub')" style="cursor:pointer;" title="Go to Dashboard">AutoML Studio</div>
      <nav id="topNav">
        <button class="nav-btn active" type="button" onclick="showScreen('hub', this)" id="navHub">Dashboard</button>
        <button class="nav-btn" type="button" onclick="showScreen('leaderboard', this)" id="navLeaderboard">Leaderboard</button>
        <button class="nav-btn" type="button" onclick="showScreen('insights', this)" id="navInsights">Insights</button>
        <button class="nav-btn" type="button" onclick="showScreen('predict', this)" id="navPredict">Predict</button>
        <button class="nav-btn" type="button" onclick="showScreen('dataset', this)" id="navDataset">Dataset</button>
      </nav>
      <button id="themeToggle" type="button" onclick="toggleTheme()" aria-label="Toggle Dark/Light Mode" style="border:1px solid var(--line); background:var(--card-solid); color:var(--text); padding:7px 12px; border-radius:10px; font-weight:700; font-size:.82rem; cursor:pointer; display:inline-flex; align-items:center; gap:6px; transition:.2s ease; white-space:nowrap;">
        <span id="themeIcon">🌙</span> <span id="themeText">Dark</span>
      </button>
    </header>

    <!-- SCREEN 1: UPLOAD & TRAIN (Default before training) -->
    <div id="screen-upload" class="screen-view active">
      <section class="hero" id="uploadSection" style="margin-top:0;">
        <div>
          <span class="tag">Python AutoML Pipeline</span>
          <h1>Train smarter. Predict faster.</h1>
          <div class="lead">Upload a dataset, auto-train multiple machine learning algorithms, and explore comprehensive comparisons and predictions.</div>
          <div class="pills">
            <div class="pill">⚡ Real-time SSE Progress</div>
            <div class="pill">🏆 9+ Model Leaderboard</div>
            <div class="pill">🎯 Multi-Model Inference</div>
            <div class="pill">📊 EDA & Feature Insights</div>
          </div>
        </div>

        <div class="panel">
          <h3>Dataset file</h3>
          <div class="field">
            <input id="fileInput" type="file" accept=".csv,.xlsx,.xls,.json,text/csv,text/plain,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/json" style="width:100%; max-width:100%; box-sizing:border-box; overflow:hidden; text-overflow:ellipsis; display:block; padding:10px 12px; border:1px solid var(--line); border-radius:12px; background:var(--input-bg); color:var(--text); font-size:.88rem; cursor:pointer;" />
            <div style="margin-top:10px; display:flex; gap:6px; flex-wrap:wrap; align-items:center;">
              <span style="font-size:.78rem; font-weight:700; color:var(--muted);">Try demo:</span>
              <button type="button" class="pill" onclick="loadDemo('titanic')" style="cursor:pointer; border:1px solid var(--line); font-size:.75rem; padding:4px 10px; background:var(--card);">Titanic Survival</button>
            </div>
            <div id="fileNotice" style="margin-top:8px; padding:10px 14px; border-radius:10px; font-size:.88rem; font-weight:700; background:var(--pill-bg); display:none; overflow-wrap:anywhere; word-break:break-word; max-width:100%; box-sizing:border-box;"></div>
          </div>
          <div class="tiny" id="fileMeta">Maximum file size: 20 MB · CSV, Excel or JSON</div>

          <div class="field">
            <label class="label" for="target">Target column</label>
            <select id="target">
              <option value="" disabled selected>Upload dataset or click demo above</option>
            </select>
          </div>

          <button class="primary-btn" type="button" id="trainBtn">Run AutoML Pipeline</button>
          <div class="status" id="statusBox">Pipeline completed successfully. <span id="statusPct">100%</span></div>
          <div class="progress"><span id="progressBar"></span></div>
        </div>
      </section>
    </div>

    <!-- SCREEN 2: DASHBOARD HUB (Shows the 4 interactive overview cards after training) -->
    <div id="screen-hub" class="screen-view">
      <div class="hub-banner">
        <div>
          <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
            <span class="tag" style="margin:0;">AutoML Completed</span>
            <span id="hubDatasetMeta" style="font-size:.84rem; font-weight:700; color:var(--muted);"></span>
          </div>
          <h2 style="margin:8px 0 4px; font-size:1.6rem;">Model Exploration Dashboard</h2>
          <div style="font-size:.9rem; color:var(--muted);">Select any section below to inspect deep performance metrics, explainability, or run predictions.</div>
        </div>
        <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
          <button type="button" class="back-btn" onclick="showScreen('upload')" style="background:var(--pill-bg); color:var(--primary);" title="Upload a new dataset to train">
            🔄 Train New Dataset
          </button>
        </div>
      </div>

      <div class="hub-kpi-ribbon">
        <div class="hub-kpi-pill">
          <div class="k">Best Model</div>
          <div class="v" id="hubBestModel" style="color:var(--primary);">—</div>
        </div>
        <div class="hub-kpi-pill">
          <div class="k" id="hubMetricLabel">Top Metric (Accuracy)</div>
          <div class="v" id="hubAccuracy">—</div>
        </div>
        <div class="hub-kpi-pill">
          <div class="k">PR-AUC</div>
          <div class="v" id="hubPrAuc">—</div>
        </div>
        <div class="hub-kpi-pill">
          <div class="k">ROC-AUC</div>
          <div class="v" id="hubRocAuc">—</div>
        </div>
      </div>

      <div class="hub-grid">
        <!-- HUB CARD 1: LEADERBOARD -->
        <div class="hub-card" onclick="showScreen('leaderboard')">
          <div class="hub-card-header">
            <div class="hub-card-title-group">
              <span class="hub-card-icon">🏆</span>
              <div>
                <h3 class="hub-card-title">1. Model Leaderboard</h3>
                <div style="font-size:.78rem; font-weight:700; color:var(--primary); margin-top:2px;">Performance Rankings & Models</div>
              </div>
            </div>
            <span class="deck-badge success" id="hubBadge1">Ready ✓</span>
          </div>
          <p class="hub-card-desc" id="hubLeaderboardSubtitle">
            Compare all trained machine learning models across Accuracy, PR-AUC, ROC-AUC, and F1 Score. Export comparison to CSV or download the serialized .pkl model.
          </p>
          <div class="hub-card-footer">
            <span>View Full Leaderboard & Top 3</span>
            <span class="arrow">&rarr;</span>
          </div>
        </div>

        <!-- HUB CARD 2: INSIGHTS & EDA -->
        <div class="hub-card" onclick="showScreen('insights')">
          <div class="hub-card-header">
            <div class="hub-card-title-group">
              <span class="hub-card-icon">📊</span>
              <div>
                <h3 class="hub-card-title">2. Insights & EDA</h3>
                <div style="font-size:.78rem; font-weight:700; color:var(--primary); margin-top:2px;">Visualizations & Explainability</div>
              </div>
            </div>
            <span class="deck-badge success" id="hubBadge2">Ready ✓</span>
          </div>
          <p class="hub-card-desc" id="hubInsightsSubtitle">
            Inspect target class distributions, missing value health charts, feature importance rankings, and hyperparameter tuning optimization logs.
          </p>
          <div class="hub-card-footer">
            <span>Explore Charts & Feature Weights</span>
            <span class="arrow">&rarr;</span>
          </div>
        </div>

        <!-- HUB CARD 3: PREDICTION PLAYGROUND -->
        <div class="hub-card" onclick="showScreen('predict')">
          <div class="hub-card-header">
            <div class="hub-card-title-group">
              <span class="hub-card-icon">🎯</span>
              <div>
                <h3 class="hub-card-title">3. Prediction Playground</h3>
                <div style="font-size:.78rem; font-weight:700; color:var(--primary); margin-top:2px;">Multi-Model Live Inference</div>
              </div>
            </div>
            <span class="deck-badge success" id="hubBadge3">Active ✓</span>
          </div>
          <p class="hub-card-desc" id="hubPredictSubtitle">
            Test custom feature values with any trained algorithm (Random Forest, XGBoost, Decision Tree, etc.) and view live class probabilities.
          </p>
          <div class="hub-card-footer">
            <span>Run Predictions & Select Models</span>
            <span class="arrow">&rarr;</span>
          </div>
        </div>

        <!-- HUB CARD 4: DATASET & PREPROCESSING -->
        <div class="hub-card" onclick="showScreen('dataset')">
          <div class="hub-card-header">
            <div class="hub-card-title-group">
              <span class="hub-card-icon">📂</span>
              <div>
                <h3 class="hub-card-title">4. Dataset & Pipeline</h3>
                <div style="font-size:.78rem; font-weight:700; color:var(--primary); margin-top:2px;">Data Audit & Pipeline Steps</div>
              </div>
            </div>
            <span class="deck-badge success" id="hubBadge4">Clean ✓</span>
          </div>
          <p class="hub-card-desc" id="hubDatasetSubtitle">
            Review raw data sample rows, column types, dropped columns, missing value handling report, and categorical encoding details.
          </p>
          <div class="hub-card-footer">
            <span>Inspect Raw Sample & Cleaning Steps</span>
            <span class="arrow">&rarr;</span>
          </div>
        </div>
      </div>
    </div>

    <!-- SPOKE 1: LEADERBOARD SCREEN -->
    <div id="screen-leaderboard" class="screen-view">
      <div class="spoke-bar">
        <button type="button" class="back-btn" onclick="showScreen('hub')">
          <span>&larr;</span> <span>Back to Dashboard</span>
        </button>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
          <button type="button" onclick="exportLeaderboardCSV()" class="nav-btn" style="border:1px solid var(--line); background:var(--card-solid); color:var(--primary); font-size:.82rem; font-weight:700; padding:6px 14px; cursor:pointer;" title="Export table data to CSV file">
            📥 Export CSV
          </button>
          <button type="button" onclick="showScreen('predict')" class="nav-btn" style="background:var(--pill-bg); color:var(--primary); font-size:.82rem; font-weight:700; padding:6px 14px; cursor:pointer;">
            🎯 Test Predictions &rarr;
          </button>
        </div>
      </div>

      <div id="resultsContent" style="display:none;">
        <section class="metrics" id="metrics" style="margin-top:0;"></section>

        <div class="section" style="margin-top:18px;">
          <h4>Comprehensive evaluation</h4>
          <h2 style="margin:0 0 6px;">All Models Performance Leaderboard</h2>
          <div class="tiny" style="margin-bottom:12px;">Detailed comparison of all trained machine learning models across evaluation metrics (including Accuracy, F1, ROC-AUC, and PR-AUC).</div>
          <div class="table-wrap" style="height:auto; max-height:440px;"><table id="comparisonTable"></table></div>
        </div>

        <div class="section" style="margin-top:18px;">
          <h4>Model comparison</h4>
          <h2>Top 3 model comparison</h2>
          <div class="bar-list" id="comparisonBars"></div>
        </div>

        <div class="section" id="downloadSection" style="margin:14px 0 0; padding:16px 20px; background:linear-gradient(135deg, rgba(105,87,245,0.06), rgba(142,123,255,0.1)); border:1px solid var(--line); border-radius:14px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
          <div>
            <div style="font-weight:800; font-size:.95rem; color:var(--text); display:flex; align-items:center; gap:6px;">
              <span>📦</span> <strong>Download Trained Model Artifact</strong>
            </div>
            <div style="font-size:.8rem; color:var(--muted); margin-top:2px;">Select any trained model to download serialized Python pickle (<code style="font-size:.78rem; background:var(--code-bg); padding:1px 5px; border-radius:4px;">.pkl</code>) for offline inference.</div>
          </div>
          <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
            <select id="downloadModelSelect" style="padding:8px 12px; border-radius:10px; border:1px solid var(--line); background:var(--input-bg); color:var(--text); font-size:.82rem; font-weight:700;" onchange="updateDownloadLink()"></select>
            <a id="downloadModelLink" href="/api/download-model" class="primary-btn" style="width:auto; margin:0; text-decoration:none; padding:9px 18px; font-size:.88rem; display:inline-flex; align-items:center; justify-content:center; gap:6px; border-radius:10px; white-space:nowrap;">
              📥 Download (.pkl)
            </a>
          </div>
        </div>
      </div>

      <div id="resultsEmpty" class="empty-state">
        <div style="font-size:2.8rem; margin-bottom:8px;">🏆</div>
        <h3>No Models Trained Yet</h3>
        <p>Upload a dataset and click <strong>Run AutoML Pipeline</strong> to view model comparisons and leaderboard rankings.</p>
        <button type="button" class="primary-btn" onclick="showScreen('upload')" style="width:auto; padding:10px 24px; font-size:.92rem; margin-top:14px; display:inline-block;">🚀 Go to Upload & Train</button>
      </div>
    </div>

    <!-- SPOKE 2: INSIGHTS & EDA SCREEN -->
    <div id="screen-insights" class="screen-view">
      <div class="spoke-bar">
        <button type="button" class="back-btn" onclick="showScreen('hub')">
          <span>&larr;</span> <span>Back to Dashboard</span>
        </button>
        <button type="button" onclick="showScreen('predict')" class="nav-btn" style="background:var(--pill-bg); color:var(--primary); font-size:.82rem; font-weight:700; padding:6px 14px; cursor:pointer;">
          🎯 Try Predictions &rarr;
        </button>
      </div>

      <div id="detailsContent" style="display:none;">
        <div class="section" id="edaSection" style="margin-bottom:16px;">
          <h4>Exploratory Data Analysis</h4>
          <h2>Dataset Insights & Target Distribution</h2>
          <div class="results-columns" style="margin-top:10px;">
            <div class="results-col"><div class="section" style="margin:0;"><h4 style="margin:0 0 8px; color:var(--primary);">🎯 Target Class Distribution</h4><div id="targetDistChart"></div></div></div>
            <div class="results-col"><div class="section" style="margin:0;"><h4 style="margin:0 0 8px; color:var(--primary);">🩺 Data Health & Missing Values</h4><div id="missingDistChart"></div></div></div>
          </div>
        </div>
        <div class="results-columns" id="detailsSection">
          <div class="results-col"><div class="section"><h4>Model interpretability</h4><h2>Feature importance highlights</h2><div id="importanceBox"></div></div></div>
          <div class="results-col"><div class="section"><h4>Pipeline steps</h4><h2>Data preprocessing summary</h2><div class="accordion" id="prepAccordion"></div></div></div>
        </div>
        <div class="section" style="margin-top:16px;"><h4>Hyperparameter tuning</h4><h2>Best tuned models</h2><div id="tuningBox"></div></div>
      </div>
      <div id="detailsEmpty" class="empty-state">
        <div style="font-size:2.8rem; margin-bottom:8px;">🔍</div>
        <h3>No Details Available</h3>
        <p>Run the AutoML pipeline to inspect EDA visualizations, feature importance rankings, and hyperparameter tuning logs.</p>
        <button type="button" class="primary-btn" onclick="showScreen('upload')" style="width:auto; padding:10px 24px; font-size:.92rem; margin-top:14px; display:inline-block;">🚀 Go to Upload & Train</button>
      </div>
    </div>

    <!-- SPOKE 3: PREDICTION PLAYGROUND SCREEN -->
    <div id="screen-predict" class="screen-view">
      <div class="spoke-bar">
        <button type="button" class="back-btn" onclick="showScreen('hub')">
          <span>&larr;</span> <span>Back to Dashboard</span>
        </button>
        <button type="button" onclick="autofillSampleValues()" class="nav-btn" style="border:1px solid var(--line); background:var(--card-solid); color:var(--primary); font-size:.82rem; font-weight:700; padding:6px 14px; cursor:pointer;" title="Fill input fields with sample values from the dataset">
          🎲 Autofill Sample Values
        </button>
      </div>

      <form id="featureForm" style="display:none; margin-top:0;">
        <!-- Model Selection Widget for Multi-Model Prediction -->
        <div style="background:var(--card); border:1px solid var(--line); border-radius:16px; padding:16px 20px; margin-bottom:18px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
          <div>
            <label for="predictModelSelect" style="font-weight:800; font-size:.95rem; color:var(--text); display:flex; align-items:center; gap:8px;">
              <span>🤖</span> <span>Select Trained Model for Prediction:</span>
            </label>
            <div class="tiny" style="margin-top:2px;">Compare how different algorithms (Random Forest, Decision Tree, Logistic Regression, etc.) evaluate the exact same input features.</div>
          </div>
          <select id="predictModelSelect" style="min-width:240px; padding:8px 12px; border-radius:10px; border:1px solid var(--line); background:var(--input-bg); color:var(--text); font-weight:700; font-size:.88rem; cursor:pointer;"></select>
        </div>

        <div class="grid" id="featureGrid"></div>
        <button class="primary-btn" id="predictBtn" type="submit" style="margin-top:16px;">Predict with Selected Model</button>
        <div id="predictionResult" style="display:none; margin-top:16px; padding:18px 22px; border-radius:16px; border:1px solid var(--line); background:var(--card); text-align:center;"></div>
      </form>

      <div id="predictEmpty" class="empty-state">
        <div style="font-size:2.8rem; margin-bottom:8px;">🎯</div>
        <h3>Predictor Not Ready</h3>
        <p>Please train a model first in the Dataset tab. Once trained, you will be able to test inputs across all trained models here.</p>
        <button type="button" class="primary-btn" onclick="showScreen('upload')" style="width:auto; padding:10px 24px; font-size:.92rem; margin-top:14px; display:inline-block;">🚀 Go to Upload & Train</button>
      </div>
    </div>

    <!-- SPOKE 4: DATASET & STATISTICAL SCREEN -->
    <div id="screen-dataset" class="screen-view">
      <div class="spoke-bar">
        <button type="button" class="back-btn" onclick="showScreen('hub')">
          <span>&larr;</span> <span>Back to Dashboard</span>
        </button>
        <button type="button" onclick="showScreen('upload')" class="nav-btn" style="background:var(--pill-bg); color:var(--primary); font-size:.82rem; font-weight:700; padding:6px 14px; cursor:pointer;">
          🔄 Retrain or Change Dataset
        </button>
      </div>

      <div id="previewSection" style="display:none;">
        <div class="section">
          <h4>Dataset preview</h4>
          <h2>First rows of raw data</h2>
          <div class="preview-wrap"><table id="previewTable"></table></div>
        </div>

        <div class="section" style="margin-top:16px;">
          <h4>Numerical Summary</h4>
          <h2>Statistical Distributions (df.describe)</h2>
          <div class="tiny" style="margin-bottom:10px;">Central tendency, standard deviation, and quartile dispersion for numeric features.</div>
          <div class="table-wrap"><table id="numericStatsTable"></table></div>

          <div style="margin-top:24px;">
            <h4>Categorical Summary</h4>
            <h2>Discrete & Categorical Distributions</h2>
            <div class="tiny" style="margin-bottom:10px;">Unique categories, most frequent modal values, and class concentration.</div>
            <div class="table-wrap"><table id="categoricalStatsTable"></table></div>
          </div>
        </div>
      </div>

      <div id="datasetEmpty" class="empty-state">
        <div style="font-size:2.8rem; margin-bottom:8px;">📂</div>
        <h3>No Dataset Loaded</h3>
        <p>Upload a CSV, Excel, or JSON file to inspect row previews and statistical distributions.</p>
        <button type="button" class="primary-btn" onclick="showScreen('upload')" style="width:auto; padding:10px 24px; font-size:.92rem; margin-top:14px; display:inline-block;">🚀 Go to Upload & Train</button>
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

    const state = { result: null, file: null, file_id: null, filename: null, uploadPromise: null };

    const fileInput = document.getElementById('fileInput');
    const fileNotice = document.getElementById('fileNotice');
    const targetInput = document.getElementById('target');
    const metrics = document.getElementById('metrics');
    const results = document.getElementById('results');
    const previewTable = document.getElementById('previewTable');
    const prepAccordion = document.getElementById('prepAccordion');
    const byId = id => document.getElementById(id);
    const comparisonBars = byId('comparisonBars'), comparisonTable = byId('comparisonTable'), importanceBox = byId('importanceBox');
    const tuningBox = byId('tuningBox'), featureForm = byId('featureForm'), featureGrid = byId('featureGrid');
    const statusBox = byId('statusBox'), progressBar = byId('progressBar'), trainBtn = byId('trainBtn'), predictBtn = byId('predictBtn');

    const notify = (msg, bg = 'rgba(29,191,115,0.08)', col = '#0f8d56') => {
      if (!fileNotice) return;
      Object.assign(fileNotice.style, { display: 'block', background: bg, color: col });
      fileNotice.innerHTML = msg;
    };

    function autofillSampleValues() {
      const sample = state.result?.dataset?.preview?.[0];
      if (!sample) return alert('No dataset preview available to autofill from. Please run AutoML pipeline first.');
      document.querySelectorAll('#featureGrid input, #featureGrid select').forEach(el => {
        if (sample[el.name] != null) el.value = sample[el.name];
      });
    }

    function exportLeaderboardCSV() {
      const table = byId('comparisonTable');
      if (!table?.rows?.length) return alert('No leaderboard data available to export.');
      const lines = Array.from(table.rows).map(r => Array.from(r.cells).map(c => `"${(c.innerText || '').trim().replace(/"/g, '""')}"`).join(','));
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([lines.join(String.fromCharCode(10))], { type: 'text/csv;charset=utf-8;' }));
      a.download = 'model_leaderboard.csv';
      a.click();
    }

    async function loadDemo(name) {
      notify('Loading demo dataset...', 'var(--pill-bg)', 'var(--primary)');
      try {
        const res = await fetch('/api/load-demo?name=' + encodeURIComponent(name));
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'Failed to load demo dataset');
        const data = await res.json();
        Object.assign(state, { file: null, file_id: data.file_id, filename: data.filename, uploadPromise: Promise.resolve(data.file_id) });
        if (fileInput) fileInput.value = '';
        if (targetInput) {
          targetInput.innerHTML = (data.columns || []).map(c => `<option value="${c}"${c === data.default_target ? ' selected' : ''}>${c}</option>`).join('');
          if (data.default_target) targetInput.value = data.default_target;
        }
        notify(`⚡ <strong>${data.label} demo loaded!</strong> Target <code>${data.default_target}</code> selected. Click <strong>Run AutoML Pipeline</strong> to start.`);
      } catch (e) {
        notify('❌ <strong>Error:</strong> ' + e.message, 'rgba(239,68,68,0.1)', '#dc2626');
      }
    }

    function showScreen(screenId, pushToHistory = true) {
      const shouldPush = (typeof pushToHistory === 'boolean') ? pushToHistory : true;
      if (screenId === 'hub' && (!state.result || !state.result.comparison)) screenId = 'upload';
      const screens = ['upload', 'hub', 'leaderboard', 'insights', 'predict', 'dataset'];
      if (!screens.includes(screenId)) screenId = state.result ? 'hub' : 'upload';

      screens.forEach(s => byId('screen-' + s)?.classList.toggle('active', s === screenId));
      const navMap = { upload: 'navHub', hub: 'navHub', leaderboard: 'navLeaderboard', insights: 'navInsights', predict: 'navPredict', dataset: 'navDataset' };
      const activeNavId = navMap[screenId] || 'navHub';
      document.querySelectorAll('#topNav .nav-btn').forEach(btn => btn.classList.toggle('active', btn.id === activeNavId));
      const activeBtn = byId(activeNavId);
      if (activeBtn) activeBtn.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
      window.scrollTo({ top: 0, behavior: 'smooth' });

      if (shouldPush && window.history && location.hash !== '#' + screenId) {
        history.pushState({ screen: screenId }, '', '#' + screenId);
      }
    }

    function navigateTo(target) {
      const map = { card1: 'upload', upload: 'upload', card2: 'leaderboard', results: 'leaderboard', leaderboard: 'leaderboard', card3: 'insights', details: 'insights', insights: 'insights', card4: 'predict', predict: 'predict', dataset: 'dataset', hub: 'hub' };
      showScreen(map[target] || 'hub');
    }

    function updateDownloadLink() {
      const s = byId('downloadModelSelect'), l = byId('downloadModelLink');
      if (s && l) l.href = '/api/download-model?model=' + encodeURIComponent(s.value);
    }

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
      const file = fileInput.files?.[0];
      if (!file) { if (fileNotice) { fileNotice.style.display = 'none'; fileNotice.textContent = ''; } return; }

      const ext = '.' + file.name.split('.').pop().toLowerCase();
      if (!['.csv', '.xlsx', '.xls', '.json'].includes(ext)) {
        notify(`❌ <strong>Invalid file format (${ext || 'none'}):</strong> Supported formats: .csv, .xlsx, .xls, .json`, 'rgba(239,68,68,0.1)', '#dc2626');
        fileInput.value = ''; Object.assign(state, { file: null, file_id: null }); targetInput.innerHTML = ''; return;
      }
      if (file.size > 20 * 1024 * 1024) {
        notify('❌ <strong>File too large:</strong> Maximum allowed size is 20 MB.', 'rgba(239,68,68,0.1)', '#dc2626');
        fileInput.value = ''; Object.assign(state, { file: null, file_id: null }); targetInput.innerHTML = ''; return;
      }

      Object.assign(state, { file, file_id: null, uploadPromise: null });
      const localCols = await parseColumnsLocally(file);
      if (localCols?.length) {
        targetInput.innerHTML = localCols.map((c, i) => `<option value="${c}"${i === localCols.length - 1 ? ' selected' : ''}>${c}</option>`).join('');
        targetInput.selectedIndex = localCols.length - 1;
        notify(`⚡ <strong>${file.name} ready!</strong> (${localCols.length} columns detected instantly). Syncing with server...`);
      } else {
        notify(`⏳ <strong>Analyzing ${file.name}...</strong>`, 'rgba(105,87,245,0.08)', 'var(--primary)');
      }

      if (state.currentXhr) try { state.currentXhr.abort(); } catch (_) {}
      const fd = new FormData();
      fd.append('file', file);
      state.uploadProgress = 0;
      state.uploadPromise = new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        state.currentXhr = xhr;
        xhr.open('POST', '/api/upload');
        xhr.upload.onprogress = e => {
          if (!e.lengthComputable) return;
          const pct = Math.round((e.loaded / e.total) * 100);
          state.uploadProgress = pct;
          if (pct >= 100) {
            if (!state.file_id) notify(`⚡ <strong>${file.name} transfer complete!</strong> Finalizing on server...`);
            if (state.isWaitingForUpload) { statusBox.innerHTML = 'Upload transfer complete (100%). Saving dataset...'; progressBar.style.width = '15%'; }
          } else {
            if (!state.file_id) notify(`⚡ <strong>${file.name} ready!</strong> (${localCols?.length || ''} cols). Uploading: <strong>${pct}%</strong>`);
            if (state.isWaitingForUpload) { statusBox.innerHTML = `Uploading dataset to server: <strong>${pct}%</strong>...`; progressBar.style.width = Math.max(pct * 0.1, 4) + '%'; }
          }
        };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const data = JSON.parse(xhr.responseText);
              state.file_id = data.file_id;
              const serverCols = data.columns || [];
              if (serverCols.length) {
                const cur = targetInput.value, def = data.default_target;
                const chosen = cur && serverCols.includes(cur) ? cur : (def && serverCols.includes(def) ? def : serverCols[serverCols.length - 1]);
                targetInput.innerHTML = serverCols.map(c => `<option value="${c}"${c === chosen ? ' selected' : ''}>${c}</option>`).join('');
                targetInput.value = chosen;
              }
              const tgtNote = targetInput.value ? ` Target <code>${targetInput.value}</code> selected.` : '';
              notify(`✓ <strong>${data.filename || file.name} uploaded (100%)!</strong>${tgtNote} (${serverCols.length || localCols?.length || 0} cols ready).`, 'rgba(29,191,115,0.1)', '#0f8d56');
              resolve(data.file_id);
            } catch (err) { reject(err); }
          } else {
            let detail = 'Upload failed';
            try { detail = JSON.parse(xhr.responseText).detail || detail; } catch (_) {}
            reject(new Error(detail));
          }
        };
        xhr.onerror = () => reject(new Error('Network error during upload.'));
        xhr.send(fd);
      }).catch(e => {
        notify('❌ <strong>Upload error:</strong> ' + e.message, 'rgba(239,68,68,0.1)', '#dc2626');
        state.file_id = state.uploadPromise = null;
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

          setTimeout(() => {
            showScreen('hub');
          }, 450);
        } catch (err) {
          console.error('Bad result event', err);
        }
      });
      es.addEventListener('error', (e) => {
        es.close();
        if (state.result) return;
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
      const modelSelect = document.getElementById('predictModelSelect');
      if (modelSelect && modelSelect.value) {
        payload['_model_name'] = modelSelect.value;
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
        if (data?.model_used) {
          html += `<div style="font-size:.85rem; font-weight:700; color:var(--muted); margin-top:4px;">Inference Algorithm: <strong style="color:var(--text);">${data.model_used}</strong></div>`;
        }
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
      ['resultsContent', 'featureForm', 'detailsContent', 'previewSection'].forEach(id => { const el = byId(id); if (el) el.style.display = 'block'; });
      ['resultsEmpty', 'predictEmpty', 'detailsEmpty', 'datasetEmpty'].forEach(id => { const el = byId(id); if (el) el.style.display = 'none'; });
      renderPreview(payload.dataset?.preview || []);
      renderStatistics(payload.dataset?.describe_numerical || [], payload.dataset?.describe_categorical || []);

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

      // Populate Predict Model Select dropdown
      const pSelect = document.getElementById('predictModelSelect');
      if (pSelect && allComparison.length) {
        const bestName = payload.best_model_name || (payload.best_model && payload.best_model.Model);
        pSelect.innerHTML = allComparison.map(m => {
          const isBest = m.Model === bestName;
          const star = isBest ? ' 🏆 (Best Model)' : '';
          return `<option value="${m.Model}"${isBest ? ' selected' : ''}>${m.Model}${star}</option>`;
        }).join('');
      }

      // Populate Download Model Select dropdown
      const dSelect = document.getElementById('downloadModelSelect');
      if (dSelect && allComparison.length) {
        const bestName = payload.best_model_name || (payload.best_model && payload.best_model.Model);
        dSelect.innerHTML = allComparison.map(m => {
          const isBest = m.Model === bestName;
          const star = isBest ? ' 🏆 (Best)' : '';
          return `<option value="${m.Model}"${isBest ? ' selected' : ''}>${m.Model}${star}</option>`;
        }).join('');
        updateDownloadLink();
      }

      // Populate Hub Header & KPIs
      const hubMeta = document.getElementById('hubDatasetMeta');
      if (hubMeta) {
        hubMeta.textContent = `${state.filename || 'Dataset'} · ${payload.summary?.rows || payload.dashboard_summary?.rows || 0} Rows · ${payload.summary?.columns || payload.dashboard_summary?.columns || 0} Cols · ${payload.summary?.problem_type || payload.dashboard_summary?.problem_type || 'Classification'}`;
      }
      const hubBest = document.getElementById('hubBestModel');
      if (hubBest) hubBest.textContent = payload.best_model_name || '—';

      const hubAcc = document.getElementById('hubAccuracy');
      const hubMetricLbl = document.getElementById('hubMetricLabel');
      const metricLabel = payload.summary?.best_metric_label || payload.primary_metric || 'Accuracy';
      if (hubMetricLbl) hubMetricLbl.textContent = `${metricLabel} (Top Metric)`;
      if (hubAcc) {
        const val = payload.summary?.best_metric_value ?? payload.best_model?.[metricLabel] ?? payload.best_model?.Accuracy ?? '—';
        hubAcc.textContent = formatMetric(val, metricLabel);
      }
      const hubPr = document.getElementById('hubPrAuc');
      if (hubPr) {
        const val = payload.best_model?.['PR-AUC'] ?? '—';
        hubPr.textContent = formatMetric(val, 'PR-AUC');
      }
      const hubRoc = document.getElementById('hubRocAuc');
      if (hubRoc) {
        const val = payload.best_model?.['ROC-AUC'] ?? '—';
        hubRoc.textContent = formatMetric(val, 'ROC-AUC');
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
      const sec = document.getElementById('edaSection');
      if (!sec) return;
      if (!eda || (!eda.target_distribution && !eda.missing_distribution)) { sec.style.display = 'none'; return; }
      sec.style.display = 'block';

      // 1. Target class distribution
      const tc = document.getElementById('targetDistChart');
      const tgt = Object.entries(eda.target_distribution || {});
      if (!tgt.length) {
        tc.innerHTML = '<div class="tiny">No target distribution data available.</div>';
      } else {
        const total = tgt.reduce((s, [, c]) => s + Number(c), 0) || 1;
        const max = Math.max(...tgt.map(([, c]) => Number(c)), 1);
        const imbalanced = tgt.length >= 2 && (max / total) >= 0.75;
        const dominant = imbalanced ? tgt.find(([, c]) => Number(c) === max) : null;
        const alertHtml = imbalanced ? `<div style="margin-bottom:8px; padding:6px 10px; border-radius:8px; background:rgba(245,158,11,0.12); color:#d97706; font-size:.78rem; font-weight:700;">⚠️ Imbalanced Target: Class "${dominant ? dominant[0] : ''}" is ${((max/total)*100).toFixed(1)}%. F1/ROC-AUC prioritized.</div>` : '';
        tc.innerHTML = alertHtml + `<div class="tiny" style="margin-bottom:8px;">Total labeled samples: <strong>${total.toLocaleString()}</strong></div><div class="bar-list">` +
          tgt.map(([k, c]) => `<div class="bar-row"><div title="${k}">${k}</div><div class="bar-track"><div class="bar-fill" style="width:${Math.max((Number(c)/max)*100, 6)}%;"></div></div><div><strong>${((Number(c)/total)*100).toFixed(1)}%</strong> <span class="tiny">(${Number(c).toLocaleString()})</span></div></div>`).join('') + `</div>`;
      }

      // 2. Data health & missing values
      const mc = document.getElementById('missingDistChart');
      const miss = eda.missing_distribution || [];
      const totalMiss = eda.total_missing || 0;
      const badges = `<div style="display:flex; gap:8px; margin-bottom:10px; flex-wrap:wrap; font-size:.8rem;">
        <span class="pill" style="padding:4px 8px;">🔢 Numerical: <strong>${eda.num_numerical_cols || 0}</strong></span>
        <span class="pill" style="padding:4px 8px;">🔤 Categorical: <strong>${eda.num_categorical_cols || 0}</strong></span>
        <span class="pill" style="padding:4px 8px; color:${totalMiss === 0 ? '#0f8d56' : '#dc2626'};">${totalMiss === 0 ? '✓ Zero Missing Values' : `⚠️ ${totalMiss} Missing Values`}</span>
      </div>`;
      mc.innerHTML = badges + (miss.length === 0
        ? '<div style="padding:12px; background:rgba(29,191,115,0.06); border-radius:8px; color:#0f8d56; font-size:.84rem; text-align:center;">✨ <strong>100% Complete Data!</strong> No missing values detected in any feature.</div>'
        : `<div class="tiny" style="margin-bottom:6px;">Top columns with missing data:</div><div class="bar-list">` +
          miss.map(m => `<div class="bar-row"><div title="${m.column}">${m.column}</div><div class="bar-track"><div class="bar-fill" style="width:${Math.max(m.percentage, 4)}%; background:linear-gradient(90deg, #f59e0b, #ef4444);"></div></div><div><strong style="color:#dc2626;">${m.percentage}%</strong> <span class="tiny">(${m.missing_count})</span></div></div>`).join('') + `</div>`);
    }

    function formatBlock(value) {
      if (!value || (Array.isArray(value) && !value.length)) return '<div class="tiny">None</div>';
      if (Array.isArray(value)) return value.map(item => `<div>${item}</div>`).join('');
      if (typeof value === 'object') return Object.entries(value).map(([k, v]) => `<div><strong>${k}</strong>: ${v}</div>`).join('');
      return `<div>${value}</div>`;
    }

    function renderPreview(rows) {
      const tbl = document.getElementById('previewTable');
      if (!tbl) return;
      if (!rows || !rows.length) { tbl.innerHTML = '<tbody><tr><td class="tiny" style="text-align:center; padding:16px;">No preview records available.</td></tr></tbody>'; return; }
      const cols = Object.keys(rows[0] || {});
      const tgt = (state.target_column || state.result?.target_column || (targetInput && targetInput.value) || '').trim().toLowerCase();
      const isTgt = c => tgt && (c.toLowerCase() === tgt || c.toLowerCase().replace(/ /g, '_') === tgt.replace(/ /g, '_'));

      tbl.innerHTML = `<thead><tr><th style="width:40px; text-align:center;">#</th>` +
        cols.map(c => isTgt(c) ? `<th style="background:rgba(105, 87, 245, 0.16); color:var(--primary); font-weight:800; border-bottom:2px solid var(--primary); white-space:nowrap;">🎯 ${c} <span class="deck-badge" style="background:var(--primary); color:#fff; font-size:.68rem; margin-left:4px;">Target</span></th>` : `<th>${c}</th>`).join('') +
        `</tr></thead><tbody>` +
        rows.slice(0, 10).map((r, i) => `<tr><td style="font-weight:700; color:var(--muted); text-align:center;">${i + 1}</td>` +
          cols.map(c => isTgt(c) ? `<td style="background:rgba(105, 87, 245, 0.05); font-weight:700; color:var(--primary);">${r[c] ?? '<span style="color:var(--muted); font-style:italic;">null</span>'}</td>` : `<td>${r[c] ?? '<span style="color:var(--muted); font-style:italic;">null</span>'}</td>`).join('') +
        `</tr>`).join('') + `</tbody>`;
    }

    function renderStatistics(numStats = [], catStats = []) {
      const numTbl = document.getElementById('numericStatsTable');
      const catTbl = document.getElementById('categoricalStatsTable');
      if (numTbl) {
        numTbl.innerHTML = !numStats.length ? '<tbody><tr><td class="tiny" style="text-align:center; padding:16px;">No numerical features available for describe statistics.</td></tr></tbody>' :
          `<thead><tr><th>Feature</th><th>Count</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>25%</th><th>Median (50%)</th><th>75%</th><th>Max</th></tr></thead><tbody>` +
          numStats.map(s => `<tr><td><strong>${s.column}</strong></td><td>${Number(s.count).toLocaleString()}</td><td><strong>${s.mean}</strong></td><td>${s.std}</td><td>${s.min}</td><td>${s.q25}</td><td><span style="color:var(--primary); font-weight:700;">${s.median}</span></td><td>${s.q75}</td><td>${s.max}</td></tr>`).join('') + `</tbody>`;
      }
      if (catTbl) {
        catTbl.innerHTML = !catStats.length ? '<tbody><tr><td class="tiny" style="text-align:center; padding:16px;">No categorical features available.</td></tr></tbody>' :
          `<thead><tr><th>Feature</th><th>Count</th><th>Unique Classes</th><th>Most Frequent (Mode)</th><th>Mode Frequency</th><th>Class Dominance</th></tr></thead><tbody>` +
          catStats.map(c => `<tr><td><strong>${c.column}</strong></td><td>${Number(c.count).toLocaleString()}</td><td><strong>${c.unique}</strong></td><td><span class="tag-sample">${c.top}</span></td><td>${Number(c.freq).toLocaleString()} occurrences</td><td><span class="mini-bar-track"><span class="mini-bar-fill" style="width:${Math.max(c.freq_pct, 4)}%; background:var(--primary);"></span></span> <strong>${c.freq_pct}%</strong></td></tr>`).join('') + `</tbody>`;
      }
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
    // Browser back button & mobile swipe gesture navigation support
    window.addEventListener('popstate', (e) => {
      const scr = e.state?.screen || (location.hash ? location.hash.replace('#', '') : (state.result ? 'hub' : 'upload'));
      showScreen(scr, false);
    });
    const initHash = location.hash ? location.hash.replace('#', '') : 'upload';
    if (!history.state) {
      history.replaceState({ screen: initHash }, '', '#' + initHash);
    }

    // Auto-load Titanic demo on initial visit if no file is selected yet
    window.addEventListener('DOMContentLoaded', () => {
      if (!state.file_id) {
        loadDemo('titanic');
      }
    });
    if (!state.file_id && document.readyState !== 'loading') {
      loadDemo('titanic');
    }
    </script>
</body>
</html>
"""


@app.api_route("/", methods=["GET", "HEAD"])
def home(request: Request):
    ua = request.headers.get("user-agent", "").lower()
    if "cron-job" in ua or "uptime" in ua or "pingdom" in ua or "betteruptime" in ua:
        return PlainTextResponse("OK")
    return HTMLResponse(content=HTML)


@app.api_route("/health", methods=["GET", "HEAD"])
@app.api_route("/api/health", methods=["GET", "HEAD"])
@app.api_route("/ping", methods=["GET", "HEAD"])
def health_check():
    return PlainTextResponse("OK")


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
            trained_model_objects = result.get('trained_model_objects') or {}
            serializable_result = _to_serializable(dict(result))
            if isinstance(serializable_result, dict):
                serializable_result.pop('best_model_object', None)
                serializable_result.pop('trained_model_objects', None)
            MODEL_STATE.clear()
            if isinstance(serializable_result, dict):
                MODEL_STATE.update(serializable_result)
            if best_model_object is not None:
                MODEL_STATE['best_model_object'] = best_model_object
            if trained_model_objects:
                MODEL_STATE['trained_model_objects'] = trained_model_objects
            MODEL_STATE['raw_dataset_path'] = str(found)
            MODEL_STATE['raw_dataset_name'] = found.name
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
    model_name = payload.get('_model_name')
    trained_models = MODEL_STATE.get('trained_model_objects') or {}
    if model_name and model_name in trained_models:
        model = trained_models[model_name]
        selected_model_name = model_name
    else:
        model = MODEL_STATE.get('best_model_object')
        selected_model_name = MODEL_STATE.get('best_model_name', 'Best Model')

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
    return {
        'prediction': pred,
        'prediction_label': pred_label,
        'probabilities': probs,
        'model_used': selected_model_name,
    }


@app.get('/api/download-model')
async def download_model(model: str | None = None):
    from fastapi.responses import Response
    import pickle

    trained_models = MODEL_STATE.get('trained_model_objects') or {}
    if model and model in trained_models:
        target_model = trained_models[model]
        file_name = model.replace(' ', '_').lower()
    else:
        target_model = MODEL_STATE.get('best_model_object')
        file_name = str(MODEL_STATE.get('best_model_name', 'model')).replace(' ', '_').lower()

    if target_model is None:
        return JSONResponse(
            {'detail': 'No trained model available to download. Please run the AutoML pipeline first.'},
            status_code=400,
        )

    return Response(
        content=pickle.dumps(target_model),
        media_type='application/octet-stream',
        headers={'Content-Disposition': f'attachment; filename="{file_name}.pkl"'},
    )


@app.get('/api/download-raw-dataset')
async def download_raw_dataset():
    from fastapi.responses import FileResponse
    path_str = MODEL_STATE.get('raw_dataset_path')
    if not path_str or not Path(path_str).exists():
        uploaded_files = sorted(UPLOAD_DIR.glob('*.csv'), key=os.path.getmtime, reverse=True)
        if uploaded_files:
            path_str = str(uploaded_files[0])
        else:
            return JSONResponse({'detail': 'No dataset file found on server. Please upload or train first.'}, status_code=404)
    file_path = Path(path_str)
    filename = MODEL_STATE.get('raw_dataset_name') or 'dataset.csv'
    return FileResponse(path=file_path, filename=filename, media_type='text/csv')




if __name__ == '__main__':
  import uvicorn
  uvicorn.run('main:app', host='0.0.0.0', port=int(os.getenv('PORT', 8080)), reload=False)