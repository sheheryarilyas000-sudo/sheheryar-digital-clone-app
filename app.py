import os
import io
import glob
import requests
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Sheheryar Ilyas — AI Clone",
    page_icon="https://api.iconify.design/fluent:bot-24-filled.svg",
    layout="wide"
)

# =========================================================================
# FUTURISTIC NEON TECH — Exact Reference Style
# =========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body { background-color: #060212 !important; }
[class*="css"] { font-family: 'Inter', sans-serif !important; }

#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 1rem 1.8rem 2rem 1.8rem !important; max-width: 100% !important; }

/* ════ FORCE ALL CONTAINERS TRANSPARENT ════ */
.stApp, .stApp > div, .stApp > div > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stMain"], [data-testid="stMain"] > div,
.main, .main > div, section.main, section.main > div,
[data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"],
[data-testid="column"], [data-testid="stChatMessageContainer"],
.stChatFloatingInputContainer, div[data-baseweb], div[data-baseweb] > div,
div[class*="st-emotion-cache"] {
    background-color: transparent !important;
    background: transparent !important;
}

/* ════ TRUE DARK NAVY BASE ════ */
.stApp {
    background-color: #060212 !important;
    min-height: 100vh;
}

/* ════ NEON LIGHT BEAMS via body pseudo (reference style) ════ */
body::before {
    content: '';
    position: fixed;
    top: -30%;
    left: -20%;
    width: 65%;
    height: 80%;
    background: linear-gradient(145deg,
        rgba(0, 120, 255, 0.55) 0%,
        rgba(0, 80, 200, 0.3) 30%,
        transparent 65%);
    transform: rotate(10deg);
    filter: blur(70px);
    pointer-events: none;
    z-index: 0;
}

body::after {
    content: '';
    position: fixed;
    bottom: -20%;
    right: -15%;
    width: 55%;
    height: 70%;
    background: linear-gradient(325deg,
        rgba(220, 0, 255, 0.55) 0%,
        rgba(160, 0, 200, 0.3) 30%,
        transparent 65%);
    transform: rotate(-10deg);
    filter: blur(80px);
    pointer-events: none;
    z-index: 0;
}

/* ════ GLASSMORPHISM CARD BASE (matches reference) ════ */
.glass-card {
    background: rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(30px) !important;
    -webkit-backdrop-filter: blur(30px) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 20px !important;
}

/* ── TEXT ── */
h1 { font-size:1.8rem !important; font-weight:800 !important; color:#fff !important;
     letter-spacing:-0.5px !important; text-shadow:0 0 40px rgba(100,160,255,0.6) !important; }
h2 { font-size:1rem !important; font-weight:700 !important; color:#c8d8f0 !important; }
h3 { font-size:0.85rem !important; font-weight:600 !important; color:#7090b0 !important;
     text-transform:uppercase !important; letter-spacing:1.5px !important; }
p, li { color:#9ab0cc !important; line-height:1.65 !important; }

/* ── ALERT BANNER ── */
[data-testid="stAlert"] {
    background: rgba(180,20,40,0.12) !important;
    border: 1px solid rgba(220,50,70,0.3) !important;
    border-left: 3px solid #ef4444 !important;
    border-radius: 12px !important;
    color: #fca5a5 !important;
    font-size: 13px !important;
    backdrop-filter: blur(20px) !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: rgba(6,2,18,0.9) !important;
    border-right: 1px solid rgba(100,150,255,0.1) !important;
    backdrop-filter: blur(30px) !important;
}
[data-testid="stSidebar"] * { color: #7090b0 !important; }
[data-testid="stSidebar"] strong, [data-testid="stSidebar"] b { color: #c8d8f0 !important; }

/* ── TABS — glass pill style (matches reference cards) ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    padding: 4px !important;
}
[data-testid="stTabs"] button[role="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 9px !important;
    color: #4a6080 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(30,60,180,0.6), rgba(80,40,160,0.6)) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 0 25px rgba(80,120,255,0.3) !important;
}

/* ── BUTTONS — reference inner card style ── */
.stButton > button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #a0c0e8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 9px 10px !important;
    backdrop-filter: blur(20px) !important;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
}
.stButton > button:hover {
    background: rgba(80,120,255,0.18) !important;
    border-color: rgba(100,160,255,0.45) !important;
    color: #ffffff !important;
    box-shadow: 0 0 25px rgba(80,120,255,0.25), 0 4px 20px rgba(0,0,0,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.045) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    padding: 16px 18px !important;
    margin-bottom: 10px !important;
    backdrop-filter: blur(25px) !important;
    box-shadow: 0 4px 30px rgba(0,0,0,0.5) !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(100,160,255,0.25) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(25px) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e2e8f0 !important;
    font-size: 14px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(100,160,255,0.55) !important;
    box-shadow: 0 0 0 3px rgba(80,120,255,0.1), 0 0 40px rgba(80,120,255,0.12) !important;
}

/* ── EXPANDER CARDS ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(25px) !important;
    margin-bottom: 8px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
    overflow: hidden !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(100,160,255,0.3) !important;
    box-shadow: 0 4px 30px rgba(0,0,0,0.5), 0 0 20px rgba(80,120,255,0.1) !important;
}
[data-testid="stExpander"] summary {
    color: #80b0e0 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 13px 16px !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.045) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    padding: 22px 18px !important;
    text-align: center !important;
    backdrop-filter: blur(25px) !important;
    box-shadow: 0 4px 30px rgba(0,0,0,0.4) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5), 0 0 25px rgba(80,120,255,0.12) !important;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important; font-weight: 800 !important;
    color: #60a5fa !important;
    text-shadow: 0 0 25px rgba(96,165,250,0.5) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important; color: #3a5070 !important;
    text-transform: uppercase !important; letter-spacing: 1px !important; font-weight: 600 !important;
}

/* ── INFO BOX ── */
[data-testid="stInfo"] {
    background: rgba(20,50,120,0.2) !important;
    border: 1px solid rgba(100,160,255,0.25) !important;
    border-left: 3px solid #3b82f6 !important;
    border-radius: 10px !important;
    color: #80b0e0 !important;
    backdrop-filter: blur(15px) !important;
}

/* ── RADIO ── */
[data-testid="stRadio"] label { color: #7090b0 !important; font-size: 13px !important; }
[data-testid="stRadio"] label:has(input:checked) { color: #80b0e0 !important; }

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #9ab0cc !important;
}

/* ── CHECKBOX ── */
[data-testid="stCheckbox"] label { color: #7090b0 !important; font-size: 13px !important; }

/* ── TABLE ── */
table { background: rgba(255,255,255,0.02) !important; border-radius: 10px !important;
        border-collapse: separate !important; border-spacing: 0 !important; overflow: hidden !important; }
thead tr th {
    background: rgba(30,60,160,0.2) !important;
    color: #80b0e0 !important; font-size: 11px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.8px !important; padding: 12px 14px !important;
    border-bottom: 1px solid rgba(100,160,255,0.15) !important;
}
tbody tr td {
    color: #7090b0 !important; font-size: 12.5px !important; padding: 10px 14px !important;
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
}
tbody tr:hover td { background: rgba(80,120,255,0.05) !important; }

/* ── CODE ── */
code {
    background: rgba(80,120,255,0.12) !important; color: #80b0e0 !important;
    border-radius: 4px !important; padding: 2px 7px !important; font-size: 12px !important;
}

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] > button {
    background: rgba(5,120,80,0.1) !important;
    border-color: rgba(16,185,129,0.25) !important; color: #6ee7b7 !important;
}

/* ── CAPTIONS ── */
small, .stCaption, [data-testid="stCaptionContainer"] p {
    color: #2a4060 !important; font-size: 11.5px !important;
}

/* ── DIVIDER ── */
hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 12px 0 !important; }

/* ── MARKDOWN ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { color: #7090b0 !important; font-size: 13.5px !important; }
[data-testid="stMarkdownContainer"] strong { color: #c8d8f0 !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(80,120,255,0.3); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(100,160,255,0.5); }
</style>
""", unsafe_allow_html=True)

/* ── HEADINGS ── */
h1 {
    font-size: 1.9rem !important;
    font-weight: 900 !important;
    color: #ffffff !important;
    letter-spacing: -0.5px !important;
    text-shadow: 0 0 30px rgba(99, 179, 255, 0.5) !important;
}
h2 {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #cbd5e1 !important;
    letter-spacing: 0.3px !important;
}
h3 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
}
p, li, span { color: #cbd5e1 !important; line-height: 1.65 !important; }

/* ── ALERT BANNER ── */
[data-testid="stAlert"] {
    background: rgba(220, 20, 60, 0.08) !important;
    border: 1px solid rgba(220, 20, 60, 0.3) !important;
    border-left: 3px solid #ef4444 !important;
    border-radius: 10px !important;
    color: #fca5a5 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    backdrop-filter: blur(20px) !important;
    padding: 10px 16px !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: rgba(5, 5, 20, 0.85) !important;
    border-right: 1px solid rgba(99, 179, 255, 0.12) !important;
    backdrop-filter: blur(40px) !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #e2e8f0 !important; }

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(99,179,255,0.1) !important;
    border-radius: 10px !important;
    padding: 3px !important;
    gap: 2px !important;
}
[data-testid="stTabs"] button[role="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    color: #475569 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a8a, #312e81) !important;
    color: #e2e8f0 !important;
    font-weight: 700 !important;
    box-shadow: 0 0 20px rgba(99,179,255,0.25) !important;
}

/* ── BUTTONS — Neon glass style ── */
.stButton > button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(99,179,255,0.2) !important;
    border-radius: 8px !important;
    color: #93c5fd !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 8px 10px !important;
    letter-spacing: 0.2px !important;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
    backdrop-filter: blur(10px) !important;
}
.stButton > button:hover {
    background: rgba(99,179,255,0.1) !important;
    border-color: rgba(99,179,255,0.5) !important;
    color: #ffffff !important;
    box-shadow: 0 0 20px rgba(99,179,255,0.2), 0 0 60px rgba(99,179,255,0.05) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 16px 18px !important;
    margin-bottom: 10px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-color: rgba(99,179,255,0.15) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), 0 0 1px rgba(99,179,255,0.3) !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,179,255,0.2) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(20px) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e2e8f0 !important;
    font-size: 14px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #334155 !important; }
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99,179,255,0.45) !important;
    box-shadow: 0 0 0 3px rgba(99,179,255,0.08), 0 0 30px rgba(99,179,255,0.1) !important;
}

/* ── EXPANDER CARDS ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.3) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    overflow: hidden !important;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(99,179,255,0.25) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), 0 0 20px rgba(99,179,255,0.06) !important;
}
[data-testid="stExpander"] summary {
    color: #93c5fd !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 13px 16px !important;
    letter-spacing: 0.1px !important;
}
[data-testid="stExpander"] summary:hover { color: #e2e8f0 !important; }

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    padding: 20px 16px !important;
    text-align: center !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 20px rgba(99,179,255,0.08) !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    color: #60a5fa !important;
    text-shadow: 0 0 20px rgba(96,165,250,0.4) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* ── INFO BOX ── */
[data-testid="stInfo"] {
    background: rgba(30,58,138,0.15) !important;
    border: 1px solid rgba(99,179,255,0.2) !important;
    border-left: 3px solid #3b82f6 !important;
    border-radius: 10px !important;
    color: #93c5fd !important;
    font-size: 13px !important;
    backdrop-filter: blur(10px) !important;
}

/* ── SELECT / RADIO / CHECKBOX ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #cbd5e1 !important;
}
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label { color: #94a3b8 !important; font-size: 13px !important; }
[data-testid="stRadio"] label:has(input:checked) { color: #93c5fd !important; }

/* ── TABLE ── */
table {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 10px !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    overflow: hidden !important;
    width: 100% !important;
}
thead tr th {
    background: rgba(30,58,138,0.25) !important;
    color: #93c5fd !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    padding: 12px 14px !important;
    border-bottom: 1px solid rgba(99,179,255,0.15) !important;
}
tbody tr { transition: background 0.15s !important; }
tbody tr:hover td { background: rgba(99,179,255,0.04) !important; }
tbody tr td {
    color: #94a3b8 !important;
    font-size: 12.5px !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
}

/* ── CODE INLINE ── */
code {
    background: rgba(99,179,255,0.1) !important;
    color: #60a5fa !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-size: 12px !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] > button {
    background: rgba(5,150,105,0.1) !important;
    border-color: rgba(16,185,129,0.25) !important;
    color: #6ee7b7 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(5,150,105,0.2) !important;
    box-shadow: 0 0 20px rgba(16,185,129,0.15) !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] [role="slider"] {
    background: #3b82f6 !important;
    border: 2px solid #60a5fa !important;
    box-shadow: 0 0 10px rgba(96,165,250,0.5) !important;
}

/* ── CAPTIONS ── */
small, .stCaption, [data-testid="stCaptionContainer"] p {
    color: #334155 !important;
    font-size: 11.5px !important;
}

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    margin: 12px 0 !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(99,179,255,0.25);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(99,179,255,0.45); }

/* ── MARKDOWN TEXT ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #94a3b8 !important;
    font-size: 13.5px !important;
}
[data-testid="stMarkdownContainer"] strong { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:0;overflow:hidden;">
    <div style="position:absolute;top:-25%;left:-18%;width:70%;height:80%;
        background:linear-gradient(140deg, rgba(0,120,255,0.65) 0%, rgba(0,60,180,0.3) 40%, transparent 70%);
        filter:blur(80px);transform:rotate(8deg);"></div>
    <div style="position:absolute;bottom:-25%;right:-12%;width:60%;height:75%;
        background:linear-gradient(320deg, rgba(200,0,255,0.65) 0%, rgba(140,0,200,0.3) 40%, transparent 70%);
        filter:blur(90px);transform:rotate(-8deg);"></div>
</div>
""", unsafe_allow_html=True)

# ── HEADER ──
st.markdown("""
<div style="
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 20px;
    padding: 22px 28px;
    margin-bottom: 16px;
    backdrop-filter: blur(35px);
    -webkit-backdrop-filter: blur(35px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08);
    position: relative; overflow: hidden; z-index: 1;
">
    <div style="position:absolute;top:-30px;right:-30px;width:180px;height:180px;
        background:radial-gradient(circle, rgba(80,120,255,0.12) 0%, transparent 70%);pointer-events:none;"></div>
    <div style="position:absolute;bottom:-20px;left:40%;width:150px;height:150px;
        background:radial-gradient(circle, rgba(180,0,255,0.08) 0%, transparent 70%);pointer-events:none;"></div>
    <div style="display:flex;align-items:center;gap:18px;position:relative;z-index:2;">
        <div style="width:50px;height:50px;border-radius:14px;
            background:linear-gradient(135deg,#1a3aaa,#5020c0);
            display:flex;align-items:center;justify-content:center;
            box-shadow:0 0 30px rgba(80,120,255,0.45);font-size:22px;">🤖</div>
        <div>
            <div style="font-size:1.45rem;font-weight:800;color:#ffffff;letter-spacing:-0.4px;
                text-shadow:0 0 50px rgba(100,160,255,0.5);">
                Sheheryar Ilyas &mdash; Digital Clone
            </div>
            <div style="font-size:11.5px;color:#3a5878;margin-top:3px;letter-spacing:0.8px;font-weight:400;">
                HYBRID RAG &nbsp;&middot;&nbsp; BM25 + DENSE EMBEDDINGS &nbsp;&middot;&nbsp; GEMINI POWERED
            </div>
        </div>
        <div style="margin-left:auto;display:flex;gap:8px;">
            <div style="background:rgba(16,180,120,0.12);border:1px solid rgba(16,180,120,0.3);
                border-radius:20px;padding:4px 14px;font-size:11px;color:#34d399;font-weight:700;
                letter-spacing:0.8px;text-shadow:0 0 10px rgba(52,211,153,0.4);">LIVE</div>
            <div style="background:rgba(80,120,255,0.12);border:1px solid rgba(80,120,255,0.28);
                border-radius:20px;padding:4px 14px;font-size:11px;color:#80b0ff;font-weight:700;letter-spacing:0.5px;">
                AI CLONE</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.warning("This is an AI clone, not the real person. It answers in first person strictly using Sheheryar Ilyas's personal documents.")

DATA_DIR = "data"
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")

# -------------------------------------------------------------------------
# 1. LOAD & CHUNK
# -------------------------------------------------------------------------
@st.cache_resource
def build_rag_indices(chunk_size=220, overlap=40):
    os.makedirs(DATA_DIR, exist_ok=True)
    md_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.md")))
    chunks = []
    step = chunk_size - overlap
    for filepath in md_files:
        fname = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        words = text.split()
        if not words:
            continue
        c_idx = 1
        for i in range(0, len(words), step):
            slice_words = words[i: i + chunk_size]
            if not slice_words:
                break
            chunks.append({
                "id": f"{fname}#chunk_{c_idx}",
                "source": fname,
                "word_count": len(slice_words),
                "text": " ".join(slice_words)
            })
            c_idx += 1
            if i + chunk_size >= len(words):
                break
    if not chunks:
        return [], None, None, None, 0
    tokenized_chunks = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    if os.path.exists(EMBEDDINGS_PATH):
        embeddings = np.load(EMBEDDINGS_PATH)
        if embeddings.shape[0] != len(chunks):
            embeddings = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
            np.save(EMBEDDINGS_PATH, embeddings)
    else:
        embeddings = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
        np.save(EMBEDDINGS_PATH, embeddings)
    return chunks, bm25, embedder, embeddings, len(md_files)

@st.cache_resource
def get_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

chunks, bm25, embedder, embeddings, num_docs = build_rag_indices()

# -------------------------------------------------------------------------
# AGENTIC ROUTING
# -------------------------------------------------------------------------
def agentic_route_question(question: str):
    q = question.lower()
    out_of_scope_terms = [
        "cnic", "passport", "phone number", "home address", "bank account",
        "password", "credit card", "biryani", "weather", "stock price", "bitcoin"
    ]
    if any(term in q for term in out_of_scope_terms):
        return "Out of Scope", "Decline", "Question asks for private data or topics outside Sheheryar's knowledge base."
    opinion_terms = [
        "opinion", "hot take", "advice", "feel", "think", "unwind", "relax",
        "exhausted", "hobby", "hobbies", "free time", "values", "philosophy", "prefer"
    ]
    if any(term in q for term in opinion_terms):
        return "Opinion", "Dense only", "Classified as personal opinion/lifestyle — recommends Dense semantic search."
    return "Personal Fact", "Hybrid", "Classified as factual/technical — recommends Hybrid (BM25 + Dense)."

# -------------------------------------------------------------------------
# RETRIEVAL + FUSION
# -------------------------------------------------------------------------
def min_max_normalize(scores):
    min_s, max_s = float(np.min(scores)), float(np.max(scores))
    if max_s - min_s < 1e-9:
        return np.zeros_like(scores, dtype=float)
    return (scores - min_s) / (max_s - min_s)

def retrieve(query, mode="Hybrid", fusion_type="RRF", alpha=0.5, use_reranker=False, top_k=3):
    if not chunks:
        return []
    q_tokens = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(q_tokens), dtype=float)
    bm25_order = np.argsort(bm25_scores)[::-1]
    bm25_ranks = {idx: r + 1 for r, idx in enumerate(bm25_order)}
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    dense_scores = np.dot(embeddings, q_emb).astype(float)
    dense_order = np.argsort(dense_scores)[::-1]
    dense_ranks = {idx: r + 1 for r, idx in enumerate(dense_order)}
    bm25_norm = min_max_normalize(bm25_scores)
    dense_norm = min_max_normalize(dense_scores)
    results = []
    for idx in range(len(chunks)):
        b_rank = bm25_ranks[idx]
        d_rank = dense_ranks[idx]
        b_score = float(bm25_scores[idx])
        d_score = float(dense_scores[idx])
        rrf_score = (1.0 / (60.0 + b_rank)) + (1.0 / (60.0 + d_rank))
        weighted_score = (alpha * float(dense_norm[idx])) + ((1.0 - alpha) * float(bm25_norm[idx]))
        if mode == "BM25 only":
            final_score = b_score
        elif mode == "Dense only":
            final_score = d_score
        else:
            final_score = rrf_score if fusion_type == "RRF" else weighted_score
        results.append({
            "chunk": chunks[idx], "score": final_score,
            "bm25_score": b_score, "dense_score": d_score,
            "rrf_score": rrf_score, "weighted_score": weighted_score,
            "bm25_rank": b_rank, "dense_rank": d_rank, "rerank_score": None
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    if use_reranker and len(results) > 1:
        try:
            reranker = get_cross_encoder()
            pool = results[:min(6, len(results))]
            pairs = [[query, item["chunk"]["text"]] for item in pool]
            ce_scores = reranker.predict(pairs)
            for item, ce_val in zip(pool, ce_scores):
                item["rerank_score"] = float(ce_val)
                item["score"] = float(ce_val)
            pool.sort(key=lambda x: x["score"], reverse=True)
            return pool[:top_k]
        except Exception:
            pass
    return results[:top_k]

# -------------------------------------------------------------------------
# GENERATION
# -------------------------------------------------------------------------
def generate_answer(question, retrieved_chunks, history, category, routed_mode):
    if routed_mode == "Decline" or not retrieved_chunks:
        return "I don't have that in my notes."
    context_text = "\n\n".join(
        [f"[{item['chunk']['id']} | Source: {item['chunk']['source']}]\n{item['chunk']['text']}"
         for item in retrieved_chunks]
    )
    recent_memory = ""
    if history:
        turns = [f"{m['role'].upper()}: {m['content']}" for m in history[-4:]]
        recent_memory = "\n".join(turns)
    prompt = f"""You are the AI clone of Sheheryar Ilyas, a 7th-semester BSCS student at UMT Sialkot.
Speak in the first person ("I", "my", "me"), in a clear, friendly, and natural tone. Provide a complete 2 to 4 sentence answer.
Rules:
1. Answer ONLY using the CONTEXT below. It comes from my own documents.
2. If the context does not contain the answer, say: "I don't have that in my notes." Do not guess or invent facts about me.
3. Never share phone numbers, addresses, ID numbers or other private details.
4. If asked whether you are the real person, say you are an AI clone of Sheheryar Ilyas.

CONVERSATION MEMORY:
{recent_memory if recent_memory else "No previous turns."}

CONTEXT:
{context_text}

QUESTION:
{question}
"""
    gemini_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    hf_token = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)
    if gemini_key:
        candidate_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        headers = {"Content-Type": "application/json", "x-goog-api-key": gemini_key}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.15, "maxOutputTokens": 1024}
        }
        for model_name in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            for attempt in range(2):
                try:
                    r = requests.post(url, headers=headers, json=payload, timeout=25)
                    if r.status_code == 200:
                        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    elif r.status_code in (429, 500, 502, 503, 504):
                        import time; time.sleep(1.2)
                except Exception:
                    continue
    if hf_token:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=hf_token)
            resp = client.chat_completion(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500, temperature=0.15
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass
    return "I am currently experiencing high server traffic. Please try again in a moment."

# -------------------------------------------------------------------------
# EVAL SET
# -------------------------------------------------------------------------
EVAL_SET = [
    {"q": "Q1 (Exact Rare Keyword): What is my dual-storage setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B?",
     "query": "What is my dual-storage setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B?",
     "relevant_files": ["workflow.md"],
     "best_reason": "BM25 wins — matches exact rare hardware/model tokens (ADATA, Toshiba, Qwen 2.5 7B)."},
    {"q": "Q2 (Paraphrased / No Shared Words): How do I recharge and relax when I am exhausted from coding?",
     "query": "How do I recharge and relax when I am exhausted from coding?",
     "relevant_files": ["hobbies.md"],
     "best_reason": "Dense wins — connects 'recharge/relax' semantically to movies and reels without shared words."},
    {"q": "Q3 (FYP Architecture): How does the Two-Layer Architecture in ERPify handle custom AI modules and security?",
     "query": "How does the Two-Layer Architecture in ERPify handle custom AI modules and security?",
     "relevant_files": ["erpify_architecture.md", "projects.md", "faq.md"],
     "best_reason": "Hybrid wins — combines exact 'ERPify' keyword match with semantic security concepts."},
    {"q": "Q4 (Personal Opinion): What is my honest opinion on traditional university degrees versus practical skills?",
     "query": "What is my honest opinion on traditional university degrees versus practical skills?",
     "relevant_files": ["hobbies.md", "advice.md", "career_goals_values.md"],
     "best_reason": "Hybrid wins — ranks both my opinion in hobbies.md and junior advice in advice.md."},
    {"q": "Q5 (SQE & Testing): How do I perform software quality engineering and security testing on mobile apps?",
     "query": "How do I perform software quality engineering and security testing on mobile apps?",
     "relevant_files": ["sqe_testing_notes.md", "bio.md"],
     "best_reason": "Hybrid wins — combines exact 'SQE' terms with detailed testing methodology chunks."}
]

def evaluate_hit_at_3():
    if not chunks:
        return {}, []
    hits = {"BM25 only": 0, "Dense only": 0, "Hybrid": 0}
    rows = []
    for item in EVAL_SET:
        q, rel = item["query"], item["relevant_files"]
        b_top3 = retrieve(q, mode="BM25 only", top_k=3)
        d_top3 = retrieve(q, mode="Dense only", top_k=3)
        h_top3 = retrieve(q, mode="Hybrid", top_k=3)
        b_hit = any(r["chunk"]["source"] in rel for r in b_top3)
        d_hit = any(r["chunk"]["source"] in rel for r in d_top3)
        h_hit = any(r["chunk"]["source"] in rel for r in h_top3)
        if b_hit: hits["BM25 only"] += 1
        if d_hit: hits["Dense only"] += 1
        if h_hit: hits["Hybrid"] += 1
        rows.append({
            "Question": item["q"],
            "BM25 Top-1": f"{b_top3[0]['chunk']['id']} ({b_top3[0]['bm25_score']:.2f}) [{'HIT' if b_hit else 'MISS'}]",
            "Dense Top-1": f"{d_top3[0]['chunk']['id']} ({d_top3[0]['dense_score']:.3f}) [{'HIT' if d_hit else 'MISS'}]",
            "Hybrid Top-1": f"{h_top3[0]['chunk']['id']} ({h_top3[0]['rrf_score']:.4f}) [{'HIT' if h_hit else 'MISS'}]",
            "Best Mode & Why": item["best_reason"]
        })
    total = len(EVAL_SET)
    hit_rates = {m: f"{(c/total)*100:.0f}% ({c}/{total})" for m, c in hits.items()}
    return hit_rates, rows

# =========================================================================
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#334155;text-transform:uppercase;
        letter-spacing:1.5px;padding:8px 0 12px;">Retrieval Settings</div>
    """, unsafe_allow_html=True)

    retrieval_mode = st.radio("Mode", options=["Hybrid", "BM25 only", "Dense only"], index=0)
    fusion_type = st.selectbox("Fusion Method", options=["RRF", "Weighted"],
                               disabled=(retrieval_mode != "Hybrid"))
    alpha = 0.5
    if retrieval_mode == "Hybrid" and fusion_type == "Weighted":
        alpha = st.slider("Dense Weight (alpha)", 0.0, 1.0, 0.5, 0.05)
    top_k = st.slider("Top-K Chunks", 1, 5, 3)

    st.divider()
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#334155;text-transform:uppercase;
        letter-spacing:1.5px;padding:4px 0 10px;">Advanced Features</div>
    """, unsafe_allow_html=True)
    use_agentic_router = st.checkbox("Agentic Router Override", value=False)
    use_reranker = st.checkbox("Cross-Encoder Reranker", value=False)

    st.divider()
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);
        border-radius:10px;padding:14px 16px;">
        <div style="font-size:11px;font-weight:700;color:#334155;text-transform:uppercase;
            letter-spacing:1px;margin-bottom:10px;">Index Stats</div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="font-size:12px;color:#475569;">Documents</span>
            <span style="font-size:12px;color:#60a5fa;font-weight:700;">{num_docs}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="font-size:12px;color:#475569;">Chunks</span>
            <span style="font-size:12px;color:#60a5fa;font-weight:700;">{len(chunks)}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="font-size:12px;color:#475569;">Embedding Dim</span>
            <span style="font-size:12px;color:#60a5fa;font-weight:700;">384</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="font-size:12px;color:#475569;">LLM</span>
            <span style="font-size:12px;color:#60a5fa;font-weight:700;">Gemini</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if embeddings is not None:
        buf = io.BytesIO()
        np.save(buf, embeddings)
        st.download_button("Download embeddings.npy", data=buf.getvalue(),
                           file_name="embeddings.npy", mime="application/octet-stream",
                           use_container_width=True)
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["last_retrieved"] = []
        st.rerun()

# =========================================================================
# TABS
# =========================================================================
tab1, tab2, tab3 = st.tabs(["Chat with Clone", "Hit@3 Evaluation", "Knowledge Base"])

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ── TAB 1 ──
with tab1:
    col_chat, col_chunks = st.columns([1.2, 1.0], gap="large")

    with col_chat:
        st.subheader("Ask a Question")
        st.caption("Select a preset question or type your own below")

        c1, c2 = st.columns(2)
        q_samples = [
            ("Q1 · Exact Keyword · BM25", "What is my dual-storage setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B?"),
            ("Q2 · Paraphrased · Dense", "How do I recharge and relax when I am exhausted from coding?"),
            ("Q3 · ERPify Architecture", "How does the Two-Layer Architecture in ERPify handle custom AI modules and security?"),
            ("Q4 · University Opinion", "What is my honest opinion on traditional university degrees versus practical skills?"),
            ("Q5 · SQE Testing", "How do I perform software quality engineering and security testing on mobile apps?"),
            ("Q6 · Out-of-Scope", "What is your favorite recipe for cooking chicken biryani?")
        ]
        selected_sample = None
        for i, (btn_lbl, q_txt) in enumerate(q_samples):
            with (c1 if i % 2 == 0 else c2):
                if st.button(btn_lbl, key=f"q_btn_{i}", use_container_width=True):
                    selected_sample = q_txt

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                if "meta" in msg:
                    st.caption(msg["meta"])
                st.write(msg["content"])

        user_query = st.chat_input("Ask anything about Sheheryar...")
        active_query = selected_sample if selected_sample else user_query

        if active_query:
            st.session_state["messages"].append({"role": "user", "content": active_query})
            cat, recommended_mode, route_reason = agentic_route_question(active_query)
            if use_agentic_router:
                active_mode = recommended_mode
            else:
                active_mode = "Decline" if recommended_mode == "Decline" else retrieval_mode
            retrieved_list = [] if active_mode == "Decline" else retrieve(
                active_query, mode=active_mode, fusion_type=fusion_type,
                alpha=alpha, use_reranker=use_reranker, top_k=top_k
            )
            answer = generate_answer(active_query, retrieved_list,
                                     st.session_state["messages"][:-1], cat, active_mode)
            meta_text = f"Router: {cat}  ·  Mode: {active_mode}  ·  Reranker: {'ON' if use_reranker else 'OFF'}"
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer, "meta": meta_text}
            )
            st.session_state["last_retrieved"] = retrieved_list
            st.session_state["last_route_info"] = (cat, active_mode, recommended_mode, route_reason)
            st.rerun()

    with col_chunks:
        st.subheader("Retrieved Chunks & Scores")

        if "last_route_info" in st.session_state:
            cat, m_used, rec_m, r_reason = st.session_state["last_route_info"]
            st.info(f"**Mode:** `{m_used}`  ·  **Intent:** `{cat}`  ·  Recommended: `{rec_m}`\n\n_{r_reason}_")

        if "last_retrieved" in st.session_state and st.session_state["last_retrieved"]:
            for rank, item in enumerate(st.session_state["last_retrieved"], start=1):
                c = item["chunk"]
                rank_colors = {1: "#60a5fa", 2: "#a78bfa", 3: "#34d399"}
                rc = rank_colors.get(rank, "#64748b")
                with st.expander(f"Rank {rank}  ·  {c['id']}  ·  {item['score']:.4f}", expanded=(rank == 1)):
                    st.markdown(f"""
                    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;">
                        <span style="background:rgba(99,179,255,0.1);color:#60a5fa;
                            border:1px solid rgba(99,179,255,0.2);border-radius:5px;
                            padding:2px 10px;font-size:11px;font-weight:600;">
                            {c['source']}
                        </span>
                        <span style="background:rgba(255,255,255,0.03);color:#475569;
                            border:1px solid rgba(255,255,255,0.06);border-radius:5px;
                            padding:2px 10px;font-size:11px;">
                            {c['word_count']} words
                        </span>
                        <span style="background:rgba(52,211,153,0.08);color:{rc};
                            border:1px solid rgba(52,211,153,0.15);border-radius:5px;
                            padding:2px 10px;font-size:11px;font-weight:600;">
                            #{rank}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    ca, cb = st.columns(2)
                    with ca:
                        st.markdown(f"**BM25** `{item['bm25_score']:.2f}` _(#{item['bm25_rank']})_")
                        st.markdown(f"**RRF** `{item['rrf_score']:.4f}`")
                    with cb:
                        st.markdown(f"**Dense** `{item['dense_score']:.4f}` _(#{item['dense_rank']})_")
                        st.markdown(f"**Weighted** `{item['weighted_score']:.4f}`")
                    if item["rerank_score"] is not None:
                        st.markdown(f"**Reranker** `{item['rerank_score']:.4f}`")
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='color:#64748b;font-size:13px;line-height:1.7;"
                        f"font-family:Inter,sans-serif;'>{c['text']}</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:50px 20px;">
                <div style="width:48px;height:48px;border-radius:12px;
                    background:rgba(99,179,255,0.08);border:1px solid rgba(99,179,255,0.15);
                    margin:0 auto 14px;display:flex;align-items:center;justify-content:center;
                    font-size:20px;">&#128269;</div>
                <div style="color:#1e293b;font-size:13px;font-weight:500;">
                    Ask a question to view retrieved chunks</div>
                <div style="color:#0f172a;font-size:12px;margin-top:4px;">
                    Chunks, source files and scores appear here</div>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 2 ──
with tab2:
    st.subheader("Hand-Labeled Evaluation — Hit@3")
    if chunks:
        hit_rates, eval_rows = evaluate_hit_at_3()
        k1, k2, k3 = st.columns(3)
        k1.metric("BM25 Only", hit_rates.get("BM25 only", "0%"))
        k2.metric("Dense Only", hit_rates.get("Dense only", "0%"))
        k3.metric("Hybrid RRF", hit_rates.get("Hybrid", "0%"))
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.table(eval_rows)

# ── TAB 3 ──
with tab3:
    st.subheader(f"Knowledge Base — {num_docs} Documents · {len(chunks)} Chunks")
    for c in chunks:
        with st.expander(f"{c['id']}  ·  {c['source']}  ·  {c['word_count']} words"):
            st.markdown(
                f"<div style='color:#475569;font-size:13px;line-height:1.75;'>{c['text']}</div>",
                unsafe_allow_html=True)
