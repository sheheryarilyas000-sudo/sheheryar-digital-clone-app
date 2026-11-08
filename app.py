import os
import io
import glob
import html
import time
import requests
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Sheheryar Ilyas | Digital Clone",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# EXACT PIC-3 COLOR SHADE + CORNER NEON LASERS + 100% WHITE TABS TEXT FIX
# =========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* 1. EXACT PIC-3 DEEP MIDNIGHT ROYAL-INDIGO SHADE BETWEEN CYAN & PINK CORNERS */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 18% 12%, rgba(0, 212, 255, 0.38) 0%, rgba(13, 71, 161, 0.22) 24%, transparent 48%),
        radial-gradient(circle at 90% 90%, rgba(255, 20, 147, 0.40) 0%, rgba(123, 31, 162, 0.22) 26%, transparent 50%),
        linear-gradient(135deg, #05091e 0%, #0b1130 38%, #15133c 68%, #260d3d 100%) !important;
    background-attachment: fixed !important;
    color: #F8FAFC !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    isolation: isolate !important;
}

/* 2. CORNER-POSITIONED NEON LASERS (Matches Pic-3 Top-Left & Bottom-Right Corners) */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 55px;
    left: 175px;
    width: 470px;
    height: 6px;
    background: linear-gradient(90deg, transparent 0%, #00d2ff 18%, #ffffff 50%, #00f2fe 82%, transparent 100%);
    box-shadow:
        0 0 16px 4px #00f2fe,
        0 0 48px 18px rgba(0, 229, 255, 0.78),
        0 0 110px 42px rgba(0, 119, 255, 0.52);
    transform: rotate(-40deg);
    pointer-events: none;
    z-index: 0 !important;
    border-radius: 999px;
}

[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    bottom: 45px;
    right: -85px;
    width: 470px;
    height: 6px;
    background: linear-gradient(90deg, transparent 0%, #ff007f 18%, #ffffff 50%, #ff1493 82%, transparent 100%);
    box-shadow:
        0 0 16px 4px #ff007f,
        0 0 50px 18px rgba(255, 0, 127, 0.82),
        0 0 115px 44px rgba(168, 85, 247, 0.55);
    transform: rotate(-40deg);
    pointer-events: none;
    z-index: 0 !important;
    border-radius: 999px;
}

/* Keep Main Content Strictly Above Background Lasers */
[data-testid="stMain"],
[data-testid="stAppViewBlockContainer"],
.main,
.block-container {
    background: transparent !important;
    position: relative !important;
    z-index: 2 !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 3 !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1360px !important;
}

/* 3. SERIOUS ARCHITECTURAL 3D SMOKED-ACRYLIC SQUIRCLE ICONS */
.hw-icon-3d-lg {
    width: 66px;
    height: 66px;
    min-width: 66px;
    border-radius: 19px;
    background:
        linear-gradient(145deg, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.02) 42%, rgba(0,0,0,0.45) 100%),
        radial-gradient(circle at 28% 22%, rgba(0, 242, 254, 0.45) 0%, rgba(15, 23, 42, 0.92) 62%, rgba(236, 72, 153, 0.45) 100%);
    border: 1.5px solid rgba(255, 255, 255, 0.42);
    box-shadow:
        inset 0 2px 3px rgba(255, 255, 255, 0.65),
        inset 0 -3px 6px rgba(0, 242, 254, 0.4),
        inset 2px 0 5px rgba(255, 255, 255, 0.25),
        0 12px 26px rgba(0, 0, 0, 0.7),
        0 0 22px rgba(0, 242, 254, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
}

.hw-icon-3d-sm {
    width: 36px;
    height: 36px;
    min-width: 36px;
    border-radius: 11px;
    background:
        linear-gradient(145deg, rgba(255,255,255,0.24) 0%, rgba(255,255,255,0.02) 45%, rgba(0,0,0,0.5) 100%),
        radial-gradient(circle at 25% 20%, rgba(0, 242, 254, 0.45) 0%, rgba(15, 23, 42, 0.94) 65%, rgba(236, 72, 153, 0.45) 100%);
    border: 1.2px solid rgba(255, 255, 255, 0.42);
    box-shadow:
        inset 0 1.5px 2.5px rgba(255, 255, 255, 0.7),
        inset 0 -2px 5px rgba(0, 242, 254, 0.35),
        0 6px 14px rgba(0, 0, 0, 0.6),
        0 0 12px rgba(0, 242, 254, 0.28);
    display: inline-flex;
    align-items: center;
    justify-content: center;
}

/* 4. PIC-3 CENTERPIECE FROSTED GLASS HERO SHELL */
.neon-hero-box {
    background:
        radial-gradient(circle at 8% 12%, rgba(0, 229, 255, 0.32) 0%, transparent 42%),
        radial-gradient(circle at 94% 90%, rgba(255, 20, 147, 0.35) 0%, transparent 46%),
        linear-gradient(135deg, rgba(16, 34, 88, 0.84) 0%, rgba(25, 24, 78, 0.86) 52%, rgba(74, 19, 92, 0.86) 100%);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border-radius: 24px;
    border: 1.5px solid rgba(255, 255, 255, 0.28);
    padding: 26px 32px;
    margin-bottom: 16px;
    box-shadow:
        -10px -10px 34px rgba(0, 242, 254, 0.18),
        12px 12px 38px rgba(255, 0, 128, 0.22),
        0 18px 45px rgba(0, 0, 0, 0.65),
        inset 0 1px 1px rgba(255, 255, 255, 0.38);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 20px;
}

.hero-main-title {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: #FFFFFF !important;
    margin: 0;
    line-height: 1.2;
    text-shadow: 0 3px 16px rgba(0, 0, 0, 0.45);
}

.hero-sub-line {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.86rem;
    font-weight: 400;
    color: #E2E8F0 !important;
    margin-top: 6px;
}

.glass-pill-badge {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.22) 0%, rgba(168, 85, 247, 0.32) 100%);
    border: 1.2px solid rgba(255, 255, 255, 0.4);
    border-radius: 999px;
    padding: 7px 18px;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.35);
}

/* 5. MANDATORY RUBRIC BANNER */
.rubric-glass-banner {
    background: linear-gradient(90deg, rgba(15, 34, 86, 0.86) 0%, rgba(28, 24, 78, 0.86) 55%, rgba(68, 18, 88, 0.86) 100%);
    border: 1.2px solid rgba(255, 255, 255, 0.26);
    border-left: 4px solid #00f2fe;
    border-radius: 18px;
    padding: 13px 22px;
    margin-bottom: 22px;
    color: #FFFFFF !important;
    font-size: 0.86rem;
    font-weight: 400;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
}

/* 6. MAXIMUM-SPECIFICITY OVERRIDE FOR 3 TABS: PURE WHITE TEXT (#FFFFFF) */
html body .stApp [data-testid="stTabs"] [role="tablist"],
html body .stApp .stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(90deg, rgba(14, 28, 74, 0.88) 0%, rgba(26, 22, 72, 0.88) 55%, rgba(58, 16, 78, 0.88) 100%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.28) !important;
    border-radius: 20px !important;
    padding: 8px 12px !important;
    gap: 12px !important;
    margin-bottom: 24px !important;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.45) !important;
}

html body .stApp [data-testid="stTabs"] button[role="tab"],
html body .stApp [data-testid="stTab"],
html body .stApp .stTabs [data-baseweb="tab"],
html body .stApp button[data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.10) !important;
    border: 1px solid rgba(255, 255, 255, 0.26) !important;
    border-radius: 14px !important;
    padding: 10px 22px !important;
    height: auto !important;
    min-height: 44px !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
    transition: all 0.2s ease !important;
}

/* Force EVERY inner node (p, div, span, stMarkdownContainer) inside all 3 tabs to PURE WHITE */
html body .stApp [data-testid="stTabs"] button[role="tab"] *,
html body .stApp [data-testid="stTabs"] button[role="tab"] p,
html body .stApp [data-testid="stTabs"] button[role="tab"] div,
html body .stApp [data-testid="stTabs"] button[role="tab"] span,
html body .stApp [data-testid="stTabs"] [data-testid="stMarkdownContainer"] *,
html body .stApp [data-testid="stTab"] *,
html body .stApp .stTabs [data-baseweb="tab"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    opacity: 1 !important;
}

html body .stApp [data-testid="stTabs"] button[role="tab"]:hover,
html body .stApp .stTabs [data-baseweb="tab"]:hover {
    background: rgba(0, 242, 254, 0.25) !important;
    border-color: #00f2fe !important;
    color: #FFFFFF !important;
}

html body .stApp [data-testid="stTabs"] button[role="tab"][aria-selected="true"],
html body .stApp .stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.46) 0%, rgba(236, 72, 153, 0.52) 100%) !important;
    border: 1.5px solid #00f2fe !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.38), 0 0 20px rgba(236, 72, 153, 0.30) !important;
}

html body .stApp [data-testid="stTabs"] [data-baseweb="tab-highlight"],
html body .stApp .stTabs [data-baseweb="tab-highlight"],
html body .stApp div[data-baseweb="tab-highlight"],
html body .stApp .stTabs [data-baseweb="tab-border"] {
    background-color: transparent !important;
    background: transparent !important;
    height: 0px !important;
    border: none !important;
    display: none !important;
}

/* 7. SIDEBAR GLASS STYLING */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(7, 15, 42, 0.96) 0%, rgba(16, 14, 48, 0.96) 60%, rgba(28, 10, 48, 0.97) 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.18) !important;
    z-index: 3 !important;
}
html body .stApp [data-testid="stSidebar"] p,
html body .stApp [data-testid="stSidebar"] label,
html body .stApp [data-testid="stSidebar"] span,
html body .stApp [data-testid="stSidebar"] div {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
}

/* 8. SECTION HEADERS */
.sec-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}
.sec-bar-text {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #F8FAFC !important;
}

/* 9. DIAGNOSTIC BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, rgba(20, 36, 90, 0.84) 0%, rgba(32, 26, 82, 0.84) 55%, rgba(64, 22, 90, 0.84) 100%) !important;
    border: 1.2px solid rgba(255, 255, 255, 0.26) !important;
    border-radius: 18px !important;
    padding: 13px 18px !important;
    min-height: 58px !important;
    height: auto !important;
    white-space: normal !important;
    text-align: left !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.28) !important;
    transition: all 0.2s ease !important;
}
.stButton > button p {
    color: #F8FAFC !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    line-height: 1.4 !important;
    margin: 0 !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.34) 0%, rgba(236, 72, 153, 0.40) 100%) !important;
    border-color: #00f2fe !important;
    box-shadow: 0 0 22px rgba(0, 242, 254, 0.38) !important;
    transform: translateY(-2px) !important;
}

[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, rgba(46, 36, 106, 0.88) 0%, rgba(104, 28, 122, 0.88) 100%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.45) !important;
    border-radius: 999px !important;
    text-align: center !important;
    min-height: 46px !important;
}

/* 10. CHAT MESSAGES — ZERO 'face' & 'smart_toy' LIGATURE OVERLAP */
[data-testid="stChatMessage"] {
    background: linear-gradient(135deg, rgba(16, 34, 86, 0.86) 0%, rgba(28, 24, 78, 0.86) 55%, rgba(58, 18, 84, 0.86) 100%) !important;
    border: 1.2px solid rgba(255, 255, 255, 0.24) !important;
    border-radius: 20px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.45) !important;
}

[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] div {
    color: #F8FAFC !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
    line-height: 1.65 !important;
}

[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInputContainer"],
[data-testid="stBottomBlockContainer"] {
    background: transparent !important;
}
[data-testid="stChatInput"] > div {
    background: linear-gradient(90deg, rgba(15, 30, 78, 0.94) 0%, rgba(28, 22, 74, 0.94) 55%, rgba(58, 18, 84, 0.94) 100%) !important;
    border: 1.5px solid rgba(0, 242, 254, 0.5) !important;
    border-radius: 999px !important;
    padding: 4px 10px !important;
    box-shadow: 0 0 25px rgba(0, 242, 254, 0.22), 0 8px 24px rgba(0, 0, 0, 0.6) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #CBD5E1 !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #00f2fe, #ec4899) !important;
    border-radius: 50% !important;
    color: #FFFFFF !important;
}

/* 11. CUSTOM NATIVE HTML ACCORDION FOR CHUNKS */
details.neon-glass-acc {
    background:
        radial-gradient(circle at 8% 12%, rgba(0, 242, 254, 0.14) 0%, transparent 45%),
        radial-gradient(circle at 92% 88%, rgba(255, 0, 128, 0.18) 0%, transparent 50%),
        linear-gradient(135deg, rgba(15, 32, 82, 0.88) 0%, rgba(26, 22, 74, 0.88) 55%, rgba(54, 16, 80, 0.88) 100%);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border: 1.2px solid rgba(255, 255, 255, 0.24);
    border-radius: 18px;
    margin-bottom: 12px;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.45), inset 0 1px 1px rgba(255, 255, 255, 0.25);
    overflow: hidden;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
details.neon-glass-acc:hover {
    border-color: rgba(0, 242, 254, 0.6);
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.2), 0 10px 26px rgba(0, 0, 0, 0.5);
}
details.neon-glass-acc > summary {
    list-style: none;
    cursor: pointer;
    padding: 13px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    background: rgba(255, 255, 255, 0.04);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    color: #F1F5F9 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    user-select: none;
}
details.neon-glass-acc > summary span {
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: #F1F5F9 !important;
}
details.neon-glass-acc > summary::-webkit-details-marker {
    display: none;
}
.acc-chevron {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 8px;
    background: rgba(0, 242, 254, 0.16);
    border: 1px solid rgba(0, 242, 254, 0.4);
    color: #7DD3FC !important;
    font-size: 0.72rem !important;
    transition: transform 0.2s ease;
}
details.neon-glass-acc[open] .acc-chevron {
    transform: rotate(180deg);
    background: rgba(236, 72, 153, 0.22);
    border-color: rgba(244, 114, 182, 0.5);
    color: #F9A8D4 !important;
}
.acc-body {
    padding: 16px 20px 18px 20px;
}

/* Pic-3 "OUR SERVICES" Inner Score Container */
.services-outer-box {
    background: linear-gradient(135deg, rgba(16, 32, 80, 0.72) 0%, rgba(34, 22, 78, 0.72) 100%);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 16px;
    padding: 15px;
    margin-bottom: 14px;
}
.services-4-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}
.service-mini-card {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0.04) 100%);
    border: 1px solid rgba(255, 255, 255, 0.24);
    border-radius: 13px;
    padding: 12px 8px;
    text-align: center;
    box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.25);
}
.service-mini-card .m-lbl {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.68rem;
    font-weight: 600;
    color: #CBD5E1 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: block;
}
.service-mini-card .m-val {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.98rem;
    font-weight: 600;
    color: #FFFFFF !important;
    margin-top: 4px;
    display: block;
}
.service-mini-card .m-sub {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.68rem;
    font-weight: 400;
    color: #7DD3FC !important;
    margin-top: 3px;
    display: block;
}

.passage-text-box {
    background: rgba(8, 13, 38, 0.80);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-left: 3px solid #00f2fe;
    border-radius: 12px;
    padding: 14px 16px;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem;
    font-weight: 400;
    line-height: 1.68;
}

/* 12. METRICS & EVALUATION TABLE */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(18, 38, 92, 0.86) 0%, rgba(32, 24, 82, 0.86) 55%, rgba(68, 20, 92, 0.86) 100%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.28) !important;
    border-radius: 20px !important;
    padding: 18px 20px !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45) !important;
}
[data-testid="stMetricLabel"] p {
    color: #CBD5E1 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] div {
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}

.eval-table-shell {
    border-radius: 20px;
    overflow: hidden;
    border: 1.5px solid rgba(255, 255, 255, 0.26);
    background: linear-gradient(135deg, rgba(15, 30, 78, 0.88) 0%, rgba(28, 22, 74, 0.88) 55%, rgba(56, 18, 82, 0.88) 100%);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
    margin-top: 18px;
}
.eval-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.84rem;
}
.eval-table th {
    background: rgba(255, 255, 255, 0.1);
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 14px 16px;
    text-align: left;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
}
.eval-table td {
    padding: 13px 16px;
    color: #F8FAFC !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    vertical-align: top;
    line-height: 1.55;
}
</style>
""", unsafe_allow_html=True)

# =========================================================================
# SERIOUS ARCHITECTURAL 3D HARDWARE ICONS
# =========================================================================
ICON_HERO_HEX = (
    '<div class="hw-icon-3d-lg">'
    '<svg width="36" height="36" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<defs>'
    '<linearGradient id="hexG1" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#00F2FE"/>'
    '<stop offset="50%" stop-color="#3B82F6"/>'
    '<stop offset="100%" stop-color="#EC4899"/>'
    '</linearGradient>'
    '<linearGradient id="hexG2" x1="100%" y1="0%" x2="0%" y2="100%">'
    '<stop offset="0%" stop-color="#FFFFFF"/>'
    '<stop offset="100%" stop-color="#00F2FE"/>'
    '</linearGradient>'
    '</defs>'
    '<polygon points="32,6 54,19 54,45 32,58 10,45 10,19" stroke="url(#hexG2)" stroke-width="3" fill="rgba(0,242,254,0.12)"/>'
    '<polygon points="32,15 46,23 46,41 32,49 18,41 18,23" fill="url(#hexG1)"/>'
    '<path d="M32 15 L32 32 L46 23 M32 32 L18 23 M32 32 L32 49" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round"/>'
    '</svg>'
    '</div>'
)

ICON_TERM_3D = (
    '<span class="hw-icon-3d-sm">'
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none">'
    '<path d="M4 17L10 11L4 5" stroke="#00F2FE" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M12 19H20" stroke="#F472B6" stroke-width="2.4" stroke-linecap="round"/>'
    '</svg>'
    '</span>'
)

ICON_LAYERS_3D = (
    '<span class="hw-icon-3d-sm">'
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none">'
    '<polygon points="12 2 2 7 12 12 22 7 12 2" stroke="#00F2FE" stroke-width="2.1" fill="rgba(0,242,254,0.25)"/>'
    '<polyline points="2 12 12 17 22 12" stroke="#F472B6" stroke-width="2.1" stroke-linecap="round"/>'
    '<polyline points="2 17 12 22 22 17" stroke="#FFFFFF" stroke-width="2.1" stroke-linecap="round"/>'
    '</svg>'
    '</span>'
)

ICON_SLIDERS_3D = (
    '<span class="hw-icon-3d-sm">'
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none">'
    '<line x1="4" y1="6" x2="20" y2="6" stroke="#00F2FE" stroke-width="2.1" stroke-linecap="round"/>'
    '<line x1="4" y1="12" x2="20" y2="12" stroke="#F472B6" stroke-width="2.1" stroke-linecap="round"/>'
    '<line x1="4" y1="18" x2="20" y2="18" stroke="#FFFFFF" stroke-width="2.1" stroke-linecap="round"/>'
    '<circle cx="9" cy="6" r="2.6" fill="#FFFFFF"/><circle cx="15" cy="12" r="2.6" fill="#00F2FE"/><circle cx="10" cy="18" r="2.6" fill="#F472B6"/>'
    '</svg>'
    '</span>'
)

st.markdown(
    f'<div class="neon-hero-box">'
    f'<div style="display:flex;align-items:center;gap:20px;">'
    f'{ICON_HERO_HEX}'
    f'<div>'
    f'<div class="hero-main-title">SHEHERYAR ILYAS — DIGITAL CLONE</div>'
    f'<div class="hero-sub-line">BSCS (7th Semester) @ UMT Sialkot · Hybrid RAG (BM25 + Dense Embeddings + RRF) · Gemini 2.5 Flash</div>'
    f'</div>'
    f'</div>'
    f'<div style="display:flex;gap:12px;flex-wrap:wrap;">'
    f'<span class="glass-pill-badge">● LIVE HYBRID RAG</span>'
    f'<span class="glass-pill-badge">CLO-2 AI CLONE</span>'
    f'</div>'
    f'</div>'
    f'<div class="rubric-glass-banner">'
    f'<span>⚠️ <strong>This is an AI clone, not the real person.</strong> It answers strictly in first person using Sheheryar Ilyas\'s personal documents.</span>'
    f'<span style="font-family:\'Inter\',sans-serif;font-weight:600;font-size:0.76rem;letter-spacing:0.06em;color:#7DD3FC;">100% GROUNDED</span>'
    f'</div>',
    unsafe_allow_html=True
)

DATA_DIR = "data"
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")

# -------------------------------------------------------------------------
# 1. LOAD & CHUNK (200-300 words with overlap, keeping file name & chunk ID)
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
            slice_words = words[i : i + chunk_size]
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

    # 2. BM25 Retriever
    tokenized_chunks = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    # 3. Dense Retriever (sentence-transformers + numpy cosine similarity + .npy)
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
# OPTIONAL EXTRA #1: AGENTIC ROUTING STEP (Personal Fact, Opinion, Out of Scope)
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
        return "Opinion", "Dense only", "Classified as a personal opinion/lifestyle query; recommends Dense semantic search."

    return "Personal Fact", "Hybrid", "Classified as a factual/technical query; recommends Hybrid (BM25 + Dense) search."

# -------------------------------------------------------------------------
# 3 & 4. RETRIEVAL + HYBRID FUSION (RRF & Weighted) + RERANKER
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
            "chunk": chunks[idx],
            "score": final_score,
            "bm25_score": b_score,
            "dense_score": d_score,
            "rrf_score": rrf_score,
            "weighted_score": weighted_score,
            "bm25_rank": b_rank,
            "dense_rank": d_rank,
            "rerank_score": None
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
# 5. GENERATION (First Person, Grounded, Zero Key Leak, Multi-Model Fallback)
# -------------------------------------------------------------------------
def generate_answer(question, retrieved_chunks, history, category, routed_mode):
    if routed_mode == "Decline" or not retrieved_chunks:
        return "I don't have that in my notes."

    context_text = "\n\n".join(
        [f"[{item['chunk']['id']} | Source: {item['chunk']['source']}]\n{item['chunk']['text']}" for item in retrieved_chunks]
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
            for _ in range(2):
                try:
                    r = requests.post(url, headers=headers, json=payload, timeout=25)
                    if r.status_code == 200:
                        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    elif r.status_code in (429, 500, 502, 503, 504):
                        time.sleep(1.2)
                except Exception:
                    continue

    if hf_token:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=hf_token)
            resp = client.chat_completion(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.15
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    return "I am currently experiencing high server traffic on the language model API. Please click the question again in a moment."

# -------------------------------------------------------------------------
# OPTIONAL EXTRA #3: HAND-LABELED EVALUATION SET WITH HIT@3 FOR EACH MODE
# -------------------------------------------------------------------------
EVAL_SET = [
    {
        "q": "Q1 (Exact Rare Keyword): What is my dual-storage setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B?",
        "query": "What is my dual-storage setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B?",
        "relevant_files": ["workflow.md"],
        "best_reason": "BM25 wins because it matches exact rare hardware/model tokens (ADATA, Toshiba, Qwen 2.5 7B)."
    },
    {
        "q": "Q2 (Paraphrased / No Shared Words): How do I recharge and relax when I am exhausted from coding?",
        "query": "How do I recharge and relax when I am exhausted from coding?",
        "relevant_files": ["hobbies.md"],
        "best_reason": "Dense wins because it connects 'recharge/relax' semantically to watching movies and scrolling reels without shared words."
    },
    {
        "q": "Q3 (FYP Architecture): How does the Two-Layer Architecture in ERPify handle custom AI modules and security?",
        "query": "How does the Two-Layer Architecture in ERPify handle custom AI modules and security?",
        "relevant_files": ["erpify_architecture.md", "projects.md", "faq.md"],
        "best_reason": "Hybrid wins by combining exact 'ERPify' keyword matches with semantic security and sandbox concepts."
    },
    {
        "q": "Q4 (Personal Opinion): What is my honest opinion on traditional university degrees versus practical skills?",
        "query": "What is my honest opinion on traditional university degrees versus practical skills?",
        "relevant_files": ["hobbies.md", "advice.md", "career_goals_values.md"],
        "best_reason": "Hybrid wins by ranking both my opinion in hobbies.md and my junior advice in advice.md."
    },
    {
        "q": "Q5 (SQE & Testing): How do I perform software quality engineering and security testing on mobile apps?",
        "query": "How do I perform software quality engineering and security testing on mobile apps?",
        "relevant_files": ["sqe_testing_notes.md", "bio.md"],
        "best_reason": "Hybrid wins by combining exact 'SQE' terms with detailed testing methodology chunks."
    }
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
    hit_rates = {m: f"{(c / total) * 100:.0f}% ({c}/{total})" for m, c in hits.items()}
    return hit_rates, rows

# =========================================================================
# SIDEBAR CONTROLS
# =========================================================================
with st.sidebar:
    st.markdown(f'<div class="sec-bar">{ICON_SLIDERS_3D}<span class="sec-bar-text">Retrieval Controls</span></div>', unsafe_allow_html=True)
    retrieval_mode = st.radio(
        "Retrieval Mode Selector:",
        options=["Hybrid", "BM25 only", "Dense only"],
        index=0
    )
    fusion_type = st.radio(
        "Hybrid Fusion Strategy:",
        options=["RRF", "Weighted"],
        index=0,
        horizontal=True,
        disabled=(retrieval_mode != "Hybrid")
    )
    alpha = 0.5
    if retrieval_mode == "Hybrid" and fusion_type == "Weighted":
        alpha = st.slider("Dense Weight (alpha):", 0.0, 1.0, 0.5, 0.05)

    top_k = st.slider("Top-K Chunks:", 1, 5, 3)

    st.divider()
    st.markdown(f'<div class="sec-bar">{ICON_LAYERS_3D}<span class="sec-bar-text">Bonus Capabilities</span></div>', unsafe_allow_html=True)
    use_agentic_router = st.checkbox("Auto-Override Mode with Agentic Router", value=False)
    use_reranker = st.checkbox("Enable Cross-Encoder Reranker", value=False)

    st.divider()
    st.markdown(
        f'<div class="services-outer-box">'
        f'<div style="font-family:\'Inter\',sans-serif;font-size:0.74rem;font-weight:600;color:#7DD3FC;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">System Telemetry</div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:0.84rem;"><span style="color:#FFFFFF !important;">Indexed Docs</span><strong style="color:#FFFFFF !important;">{num_docs} files</strong></div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:0.84rem;"><span style="color:#FFFFFF !important;">Overlapping Chunks</span><strong style="color:#FFFFFF !important;">{len(chunks)} chunks</strong></div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.84rem;"><span style="color:#FFFFFF !important;">Dense Vector</span><strong style="color:#F9A8D4 !important;">384-D (.npy)</strong></div>'
        f'</div>',
        unsafe_allow_html=True
    )

    if embeddings is not None:
        buf = io.BytesIO()
        np.save(buf, embeddings)
        st.download_button(
            "Download embeddings.npy",
            data=buf.getvalue(),
            file_name="embeddings.npy",
            mime="application/octet-stream",
            use_container_width=True
        )

    if st.button("Reset Conversation Memory", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["last_retrieved"] = []
        st.rerun()

# =========================================================================
# MAIN TABS
# =========================================================================
tab1, tab2, tab3 = st.tabs([
    "Clone Studio & Live Retrieval",
    "Hit@3 Evaluation & 5 Test Questions",
    f"Indexed Knowledge Base ({num_docs} Docs)"
])

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with tab1:
    col_chat, col_chunks = st.columns([1.12, 1.0], gap="large")

    with col_chat:
        st.markdown(f'<div class="sec-bar">{ICON_TERM_3D}<span class="sec-bar-text">Diagnostic Query Console</span></div>', unsafe_allow_html=True)
        st.caption("Click any diagnostic test card below or type your own question to chat with Sheheryar's AI clone:")

        c1, c2 = st.columns(2)
        q_samples = [
            ("Q1 · Hardware & Qwen 2.5 (BM25 Win)", "What is my dual-storage setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B?"),
            ("Q2 · Recharge & Relax (Dense Win)", "How do I recharge and relax when I am exhausted from coding?"),
            ("Q3 · ERPify Two-Layer Architecture", "How does the Two-Layer Architecture in ERPify handle custom AI modules and security?"),
            ("Q4 · University vs Practical Skills", "What is my honest opinion on traditional university degrees versus practical skills?"),
            ("Q5 · SQE & Security Testing Workflow", "How do I perform software quality engineering and security testing on mobile apps?"),
            ("Q6 · Out-of-Scope Guardrail Test", "What is your favorite recipe for cooking chicken biryani?")
        ]

        selected_sample = None
        for i, (btn_lbl, q_txt) in enumerate(q_samples):
            with (c1 if i % 2 == 0 else c2):
                if st.button(btn_lbl, key=f"q_btn_{i}", use_container_width=True):
                    selected_sample = q_txt

        for msg in st.session_state["messages"]:
            avatar_icon = "👤" if msg["role"] == "user" else "⚡"
            with st.chat_message(msg["role"], avatar=avatar_icon):
                if "meta" in msg:
                    st.caption(msg["meta"])
                st.write(msg["content"])

        user_query = st.chat_input("Ask Sheheryar's AI Clone anything...")
        active_query = selected_sample if selected_sample else user_query

        if active_query:
            st.session_state["messages"].append({"role": "user", "content": active_query})

            cat, recommended_mode, route_reason = agentic_route_question(active_query)
            if use_agentic_router:
                active_mode = recommended_mode
            else:
                active_mode = "Decline" if recommended_mode == "Decline" else retrieval_mode

            if active_mode == "Decline":
                retrieved_list = []
            else:
                retrieved_list = retrieve(
                    active_query,
                    mode=active_mode,
                    fusion_type=fusion_type,
                    alpha=alpha,
                    use_reranker=use_reranker,
                    top_k=top_k
                )

            answer = generate_answer(
                active_query,
                retrieved_list,
                st.session_state["messages"][:-1],
                cat,
                active_mode
            )

            meta_text = f"Router Class: {cat} · Active Mode: {active_mode} · Reranker: {'ON' if use_reranker else 'OFF'}"
            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "meta": meta_text
            })
            st.session_state["last_retrieved"] = retrieved_list
            st.session_state["last_route_info"] = (cat, active_mode, recommended_mode, route_reason)
            st.rerun()

    # RIGHT COLUMN: CUSTOM GLASS ACCORDIONS
    with col_chunks:
        st.markdown(f'<div class="sec-bar">{ICON_LAYERS_3D}<span class="sec-bar-text">Retrieved Chunks, Sources & Scores</span></div>', unsafe_allow_html=True)

        if "last_route_info" in st.session_state:
            cat, m_used, rec_m, r_reason = st.session_state["last_route_info"]
            st.markdown(
                f'<div class="services-outer-box">'
                f'<div style="font-family:\'Inter\',sans-serif;font-size:0.85rem;font-weight:600;color:#FFFFFF;">Active Mode: <span style="color:#00f2fe;">{html.escape(m_used)}</span> · Intent: <span style="color:#F9A8D4;">{html.escape(cat)}</span> (Recommended: {html.escape(rec_m)})</div>'
                f'<div style="font-size:0.8rem;color:#E2E8F0;margin-top:4px;">{html.escape(r_reason)}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        if "last_retrieved" in st.session_state and st.session_state["last_retrieved"]:
            for rank, item in enumerate(st.session_state["last_retrieved"], start=1):
                c = item["chunk"]
                ce_str = f"{item['rerank_score']:.3f}" if item["rerank_score"] is not None else "OFF"
                safe_passage = html.escape(c["text"])
                open_attr = "open" if rank == 1 else ""

                st.markdown(
                    f'<details class="neon-glass-acc" {open_attr}>'
                    f'<summary>'
                    f'<span>Rank #{rank} &nbsp;·&nbsp; {html.escape(c["id"])} &nbsp;·&nbsp; Score: {item["score"]:.4f}</span>'
                    f'<span class="acc-chevron">▼</span>'
                    f'</summary>'
                    f'<div class="acc-body">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:12px;">'
                    f'<span class="glass-pill-badge" style="padding:5px 14px;font-size:0.7rem;">Source: {html.escape(c["source"])}</span>'
                    f'<span class="glass-pill-badge" style="padding:5px 14px;font-size:0.7rem;">{c["word_count"]} Words</span>'
                    f'</div>'
                    f'<div class="services-outer-box">'
                    f'<div class="services-4-grid">'
                    f'<div class="service-mini-card"><span class="m-lbl">BM25 Score</span><span class="m-val">{item["bm25_score"]:.2f}</span><span class="m-sub">Rank #{item["bm25_rank"]}</span></div>'
                    f'<div class="service-mini-card"><span class="m-lbl">Dense Cosine</span><span class="m-val">{item["dense_score"]:.3f}</span><span class="m-sub">Rank #{item["dense_rank"]}</span></div>'
                    f'<div class="service-mini-card"><span class="m-lbl">Hybrid RRF</span><span class="m-val">{item["rrf_score"]:.4f}</span><span class="m-sub">Weighted: {item["weighted_score"]:.2f}</span></div>'
                    f'<div class="service-mini-card"><span class="m-lbl">Reranker</span><span class="m-val">{ce_str}</span><span class="m-sub">Cross-Enc</span></div>'
                    f'</div>'
                    f'</div>'
                    f'<div class="passage-text-box">{safe_passage}</div>'
                    f'</div>'
                    f'</details>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="neon-hero-box" style="text-align:center;justify-content:center;padding:44px 28px;">'
                '<div>'
                '<div style="font-family:\'Inter\',sans-serif;font-weight:600;font-size:1.02rem;color:#FFFFFF;letter-spacing:0.04em;text-transform:uppercase;">Awaiting Query Execution</div>'
                '<div style="color:#E2E8F0;font-size:0.85rem;margin-top:8px;max-width:420px;">Select any diagnostic query card on the left to inspect BM25, Dense, and Hybrid RRF scores and source passages inside frosted glass service cards.</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

with tab2:
    st.markdown(f'<div class="sec-bar">{ICON_SLIDERS_3D}<span class="sec-bar-text">Hand-Labeled Evaluation Set (Hit@3) & 5 Test Questions</span></div>', unsafe_allow_html=True)
    if chunks:
        hit_rates, eval_rows = evaluate_hit_at_3()
        k1, k2, k3 = st.columns(3)
        k1.metric("BM25 Only (Hit@3)", hit_rates.get("BM25 only", "0%"))
        k2.metric("Dense Only (Hit@3)", hit_rates.get("Dense only", "0%"))
        k3.metric("Hybrid RRF (Hit@3)", hit_rates.get("Hybrid", "0%"))

        rows_html = ""
        for r in eval_rows:
            rows_html += (
                f'<tr>'
                f'<td style="font-weight:600;color:#FFFFFF;">{html.escape(r["Question"])}</td>'
                f'<td style="font-family:\'JetBrains Mono\',monospace;color:#7DD3FC;">{html.escape(r["BM25 Top-1"])}</td>'
                f'<td style="font-family:\'JetBrains Mono\',monospace;color:#E9D5FF;">{html.escape(r["Dense Top-1"])}</td>'
                f'<td style="font-family:\'JetBrains Mono\',monospace;color:#F9A8D4;font-weight:600;">{html.escape(r["Hybrid Top-1"])}</td>'
                f'<td>{html.escape(r["Best Mode & Why"])}</td>'
                f'</tr>'
            )

        st.markdown(
            f'<div class="eval-table-shell">'
            f'<table class="eval-table">'
            f'<thead><tr><th>Test Question</th><th>BM25 Result</th><th>Dense Result</th><th>Hybrid Result</th><th>Best Mode & Why</th></tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True
        )

with tab3:
    st.markdown(f'<div class="sec-bar">{ICON_LAYERS_3D}<span class="sec-bar-text">Indexed Knowledge Base ({num_docs} Documents · {len(chunks)} Chunks)</span></div>', unsafe_allow_html=True)
    for idx_c, c in enumerate(chunks):
        open_first = "open" if idx_c == 0 else ""
        st.markdown(
            f'<details class="neon-glass-acc" {open_first}>'
            f'<summary>'
            f'<span>{html.escape(c["id"])} &nbsp;—&nbsp; {html.escape(c["source"])} ({c["word_count"]} words)</span>'
            f'<span class="acc-chevron">▼</span>'
            f'</summary>'
            f'<div class="acc-body">'
            f'<div class="passage-text-box">{html.escape(c["text"])}</div>'
            f'</div>'
            f'</details>',
            unsafe_allow_html=True
        )
