import altair as alt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder


STREAMLIT_THEME = """
<style>
    .stApp {
        background: radial-gradient(circle at top left, rgba(37, 99, 235, 0.18), transparent 32%),
                    linear-gradient(135deg, #020817 0%, #0b1220 40%, #111827 100%);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(15, 23, 42, 0.9));
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 12px;
        background: linear-gradient(135deg, #f97316, #ef4444);
        color: white;
        font-weight: 700;
        padding: 0.7rem 1rem;
        box-shadow: 0 12px 25px rgba(239, 68, 68, 0.25);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #fb923c, #f87171);
        filter: brightness(1.03);
        transform: translateY(-1px);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #7dd3fc !important;
        letter-spacing: 0.02em;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #7dd3fc !important;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #bbf7d0;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #4ade80;
        box-shadow: 0 0 12px rgba(74, 222, 128, 0.85);
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.3); opacity: 0.7; }
        100% { transform: scale(1); opacity: 1; }
    }
    .glass-panel {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 16px;
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.18);
        padding: 1rem 1.2rem;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .prediction-card {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 0.7rem 0.8rem 0.2rem 0.8rem;
        margin-bottom: 0.9rem;
    }
    .prediction-card .field-label {
        color: #e2e8f0 !important;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.45rem;
        display: block;
    }
    .prediction-card .stNumberInput, .prediction-card .stSelectbox, .prediction-card .stTextInput {
        background: rgba(148, 163, 184, 0.08);
        border-radius: 10px;
    }
    div[data-testid="stFileUploader"] > section {
        background: rgba(30,58,138,0.07) !important;
        border: 2px dashed rgba(96,165,250,0.35) !important;
        border-radius: 16px !important;
        transition: border-color 0.2s, background 0.2s !important;
    }
    div[data-testid="stFileUploader"] > section:hover {
        border-color: rgba(96,165,250,0.75) !important;
        background: rgba(30,58,138,0.14) !important;
    }
    div[data-testid="stFileUploader"] > section button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(59,130,246,0.3) !important;
    }
</style>
"""


def render_intro_card():
    st.markdown(
        """
        <div class="glass-panel" style="max-width: 900px; margin: 1.5rem 0 0 0;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.5rem;">How it works</div>
            <div style="color: #cbd5e1; line-height: 1.7;">
                1. Upload a CSV, Excel, or JSON dataset.<br>
                2. Choose the target column you want the model to predict.<br>
                3. Click <strong>Run AutoML Pipeline</strong> to clean, train, compare, and evaluate models.<br>
                4. Use the prediction tab afterward to test the best model on new values.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_card():
    st.markdown(
        """
        <div class="glass-panel" style="max-width: 520px; margin: 0.8rem 0 0.6rem 0; min-height: 150px; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.75rem;">Upload dataset</div>
            <div style="color: #cbd5e1; line-height: 1.6; margin-bottom: 0.25rem;">
                Upload a CSV, Excel, or JSON dataset. The maximum file size is 10 MB.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
