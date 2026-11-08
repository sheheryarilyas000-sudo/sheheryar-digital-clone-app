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
    page_title="Sheheryar Ilyas | Futuristic Neon Tech Clone",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# PIC 2 (NEON CYAN-TO-MAGENTA GLASS) + PIC 3 (3D CRYSTAL SQUIRCLE ICONS)
# =========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800;900&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

/* 1. EXACT PIC-2 COLOR ATMOSPHERE: Deep Navy + Top-Left Cyan Glow + Bottom-Right Pink/Magenta Glow */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 12% 16%, rgba(0, 229, 255, 0.42) 0%, rgba(0, 110, 255, 0.22) 26%, transparent 55%),
        radial-gradient(circle at 88% 85%, rgba(255, 20, 147, 0.46) 0%, rgba(162, 28, 175, 0.28) 30%, transparent 58%),
        linear-gradient(135deg, #071330 0%, #090b24 48%, #1f0933 100%) !important;
    background-attachment: fixed !important;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
}

/* 2. VISIBLE DIAGONAL LASER RAYS (Pic 2 Top-Left Cyan & Bottom-Right Magenta) */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 6%;
    left: -6%;
    width: 340px;
    height: 6px;
    background: linear-gradient(90deg, transparent 0%, #00ffff 25%, #ffffff 55%, #00bfff 85%, transparent 100%);
    box-shadow:
        0 0 18px 4px #00ffff,
        0 0 45px 14px rgba(0, 229, 255, 0.75),
        0 0 90px 30px rgba(0, 132, 255, 0.45);
    transform: rotate(-42deg);
    pointer-events: none;
    z-index: 0;
    border-radius: 999px;
}

[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    bottom: 8%;
    right: -5%;
    width: 360px;
    height: 6px;
    background: linear-gradient(90deg, transparent 0%, #ff00a0 20%, #ffffff 50%, #ff1493 80%, transparent 100%);
    box-shadow:
        0 0 18px 4px #ff1493,
        0 0 50px 16px rgba(255, 20, 147, 0.8),
        0 0 100px 35px rgba(180, 20, 255, 0.5);
    transform: rotate(-42deg);
    pointer-events: none;
    z-index: 0;
    border-radius: 999px;
}

[data-testid="stMain"], [data-testid="stHeader"] {
    background: transparent !important;
    z-index: 1;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1360px !important;
}

/* 3. PIC-3 3D LIQUID CRYSTAL GLASS SQUIRCLE ICONS */
.crystal-icon-lg {
    width: 68px;
    height: 68px;
    min-width: 68px;
    border-radius: 20px;
    position: relative;
    background:
        radial-gradient(circle at 25% 20%, rgba(255, 255, 255, 0.55) 0%, rgba(255, 255, 255, 0.05) 38%, transparent 60%),
        linear-gradient(135deg, rgba(0, 229, 255, 0.35) 0%, rgba(30, 58, 138, 0.65) 50%, rgba(217, 70, 239, 0.45) 100%);
    border: 2px solid rgba(255, 255, 255, 0.48);
    box-shadow:
        inset 0 3px 6px rgba(255, 255, 255, 0.75),
        inset 0 -5px 10px rgba(0, 229, 255, 0.45),
        inset 3px 0 8px rgba(255, 255, 255, 0.3),
        0 12px 28px rgba(0, 0, 0, 0.65),
        0 0 25px rgba(0, 229, 255, 0.45);
    display: flex;
    align-items: center;
    justify-content: center;
    backdrop-filter: blur(16px);
}

.crystal-icon-sm {
    width: 36px;
    height: 36px;
    min-width: 36px;
    border-radius: 11px;
    background:
        radial-gradient(circle at 25% 20%, rgba(255, 255, 255, 0.6) 0%, transparent 55%),
        linear-gradient(135deg, rgba(0, 229, 255, 0.4) 0%, rgba(88, 28, 135, 0.7) 55%, rgba(236, 72, 153, 0.5) 100%);
    border: 1.5px solid rgba(255, 255, 255, 0.5);
    box-shadow:
        inset 0 2px 4px rgba(255, 255, 255, 0.8),
        inset 0 -3px 6px rgba(0, 229, 255, 0.4),
        0 6px 16px rgba(0, 0, 0, 0.5),
        0 0 14px rgba(0, 229, 255, 0.35);
    display: inline-flex;
    align-items: center;
    justify-content: center;
}

/* 4. PIC-2 MAIN FROSTED GLASS HERO CARD (Cyan Top-Left to Magenta Bottom-Right) */
.pic2-glass-hero {
    background:
        radial-gradient(circle at 8% 15%, rgba(0, 229, 255, 0.38) 0%, transparent 48%),
        radial-gradient(circle at 92% 88%, rgba(255, 42, 165, 0.42) 0%, transparent 52%),
        linear-gradient(135deg, rgba(19, 46, 108, 0.72) 0%, rgba(42, 23, 88, 0.72) 52%, rgba(102, 22, 108, 0.74) 100%);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border-radius: 26px;
    border: 1.5px solid rgba(255, 255, 255, 0.28);
    padding: 26px 32px;
    margin-bottom: 16px;
    box-shadow:
        -12px -12px 38px rgba(0, 229, 255, 0.22),
        14px 14px 42px rgba(255, 20, 147, 0.26),
        0 20px 50px rgba(0, 0, 0, 0.6),
        inset 0 1px 2px rgba(255, 255, 255, 0.45);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 20px;
}

.pic2-title {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #FFFFFF !important;
    margin: 0;
    text-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
}

.pic2-subtitle {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem;
    font-weight: 500;
    color: #E2E8F0 !important;
    margin-top: 6px;
    letter-spacing: 0.02em;
}

/* Pic-2 "GET STARTED" Style Pill Badges */
.pic2-pill {
    background: linear-gradient(135deg, rgba(88, 40, 138, 0.75) 0%, rgba(156, 39, 176, 0.65) 100%);
    border: 1.5px solid rgba(255, 255, 255, 0.42);
    border-radius: 999px;
    padding: 9px 22px;
    color: #FFFFFF !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    box-shadow:
        0 8px 20px rgba(0, 0, 0, 0.35),
        0 0 18px rgba(236, 72, 153, 0.35),
        inset 0 1px 1px rgba(255, 255, 255, 0.35);
}

/* 5. MANDATORY RUBRIC BANNER (Uniform 20px Rounded Glass) */
.pic2-banner {
    background: linear-gradient(90deg, rgba(16, 48, 108, 0.76) 0%, rgba(92, 24, 104, 0.78) 100%);
    border: 1.5px solid rgba(255, 255, 255, 0.25);
    border-radius: 20px;
    padding: 13px 22px;
    margin-bottom: 20px;
    color: #FFFFFF !important;
    font-size: 0.88rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.25);
}

/* 6. SIDEBAR GLASS & FIXED DARK SELECTBOX (No More White Box!) */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10, 22, 58, 0.92) 0%, rgba(28, 12, 54, 0.95) 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.18) !important;
    backdrop-filter: blur(28px) !important;
}
[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

div[data-baseweb="select"] > div {
    background: linear-gradient(135deg, rgba(24, 38, 88, 0.95) 0%, rgba(58, 22, 88, 0.95) 100%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.32) !important;
    border-radius: 16px !important;
    color: #FFFFFF !important;
}
div[data-baseweb="select"] * {
    color: #FFFFFF !important;
    background-color: transparent !important;
}
div[data-baseweb="popover"] div,
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li {
    background-color: #16143a !important;
    color: #FFFFFF !important;
}

/* 7. TABS: HIGH-CONTRAST BRIGHT WHITE TEXT ON GLASS PILLS */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(90deg, rgba(18, 36, 84, 0.72) 0%, rgba(68, 20, 88, 0.72) 100%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.24) !important;
    border-radius: 20px !important;
    padding: 7px !important;
    gap: 10px !important;
    margin-bottom: 20px !important;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 14px !important;
    padding: 10px 22px !important;
    color: #E2E8F0 !important;
}
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span,
.stTabs [data-baseweb="tab"] div {
    color: #E2E8F0 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.42) 0%, rgba(236, 72, 153, 0.48) 100%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.55) !important;
    box-shadow: 0 0 22px rgba(0, 229, 255, 0.35), 0 0 22px rgba(236, 72, 153, 0.3) !important;
}
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span,
.stTabs [aria-selected="true"] div {
    color: #FFFFFF !important;
}

/* 8. SECTION HEADINGS WITH 3D CRYSTAL ICON */
.crystal-sec-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}
.crystal-sec-title {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.95rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #FFFFFF !important;
}

/* 9. PIC-2 "OUR SERVICES" INNER GLASS CARDS FOR BUTTONS (Full Text, Never Cut Off!) */
.stButton > button {
    background: linear-gradient(145deg, rgba(32, 48, 108, 0.72) 0%, rgba(82, 28, 104, 0.72) 100%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.28) !important;
    border-radius: 18px !important;
    padding: 12px 16px !important;
    min-height: 58px !important;
    height: auto !important;
    white-space: normal !important;
    word-wrap: break-word !important;
    text-align: left !important;
    box-shadow:
        0 8px 22px rgba(0, 0, 0, 0.35),
        inset 0 1px 1px rgba(255, 255, 255, 0.32) !important;
    transition: all 0.22s ease !important;
}
.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    line-height: 1.35 !important;
    white-space: normal !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.38) 0%, rgba(236, 72, 153, 0.45) 100%) !important;
    border-color: rgba(255, 255, 255, 0.65) !important;
    box-shadow: 0 0 24px rgba(0, 229, 255, 0.4), 0 0 28px rgba(236, 72, 153, 0.35) !important;
    transform: translateY(-2px) !important;
}

[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, rgba(88, 40, 138, 0.85) 0%, rgba(168, 38, 168, 0.85) 100%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.45) !important;
    border-radius: 999px !important;
    text-align: center !important;
    min-height: 46px !important;
}

/* 10. CHAT MESSAGES & INPUT BAR (Uniform Rounded Glass) */
[data-testid="stChatMessage"] {
    background: linear-gradient(135deg, rgba(22, 42, 98, 0.76) 0%, rgba(68, 22, 92, 0.76) 100%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.24) !important;
    border-radius: 22px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.28) !important;
}
[data-testid="stChatMessage"] * {
    color: #FFFFFF !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
}

[data-testid="stChatInput"] {
    background: linear-gradient(90deg, rgba(20, 38, 92, 0.9) 0%, rgba(72, 22, 94, 0.9) 100%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.38) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 28px rgba(0, 229, 255, 0.25), 0 10px 30px rgba(0, 0, 0, 0.55) !important;
}
[data-testid="stChatInput"] textarea {
    color: #FFFFFF !important;
}

/* 11. EXPANDERS & PIC-2 "SERVICE ONE / TWO / THREE" SCORE CARDS */
[data-testid="stExpander"] {
    background: linear-gradient(135deg, rgba(22, 42, 98, 0.74) 0%, rgba(74, 24, 96, 0.74) 100%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.26) !important;
    border-radius: 22px !important;
    overflow: hidden !important;
    margin-bottom: 14px !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.42), inset 0 1px 1px rgba(255, 255, 255, 0.3) !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #FFFFFF !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.86rem !important;
}

.pic2-services-container {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 18px;
    padding: 14px;
    margin: 10px 0 14px 0;
}
.pic2-service-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}
.pic2-service-box {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0.04) 100%);
    border: 1px solid rgba(255, 255, 255, 0.26);
    border-radius: 14px;
    padding: 12px 8px;
    text-align: center;
    box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.3);
}
.pic2-service-box .title {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.7rem;
    font-weight: 700;
    color: #E2E8F0 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: block;
}
.pic2-service-box .val {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.02rem;
    font-weight: 800;
    color: #FFFFFF !important;
    margin-top: 4px;
    display: block;
}
.pic2-service-box .sub {
    font-size: 0.68rem;
    color: #7DD3FC !important;
    margin-top: 2px;
    display: block;
}

.pic2-passage-box {
    background: rgba(10, 14, 38, 0.68);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 16px;
    padding: 14px 16px;
    color: #F8FAFC !important;
    font-size: 0.86rem;
    line-height: 1.65;
}

/* 12. METRICS & BENCHMARK TABLE */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(22, 46, 108, 0.78) 0%, rgba(86, 24, 104, 0.78) 100%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.28) !important;
    border-radius: 22px !important;
    padding: 20px !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45), inset 0 1px 1px rgba(255, 255, 255, 0.35) !important;
}
[data-testid="stMetricLabel"] * {
    color: #E2E8F0 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] * {
    color: #FFFFFF !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 800 !important;
}

.pic2-table-wrap {
    border-radius: 22px;
    overflow: hidden;
    border: 1.5px solid rgba(255, 255, 255, 0.26);
    background: linear-gradient(135deg, rgba(18, 34, 82, 0.82) 0%, rgba(62, 20, 86, 0.82) 100%);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
}
.pic2-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
}
.pic2-table th {
    background: rgba(255, 255, 255, 0.1);
    color: #FFFFFF !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 14px 16px;
    text-align: left;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
}
.pic2-table td {
    padding: 13px 16px;
    color: #F8FAFC !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    vertical-align: top;
}
</style>
""", unsafe_allow_html=True)

# =========================================================================
# PIC-3 INSPIRED 3D GLOSSY EMBLEMS INSIDE LIQUID CRYSTAL SQUIRCLES
# =========================================================================
ICON_HERO_3D = (
    '<div class="crystal-icon-lg">'
    '<svg width="40" height="40" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<defs>'
    '<linearGradient id="gPlanet" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#00FFFF"/>'
    '<stop offset="50%" stop-color="#60A5FA"/>'
    '<stop offset="100%" stop-color="#F472B6"/>'
    '</linearGradient>'
    '<linearGradient id="gRing" x1="0%" y1="0%" x2="100%" y2="0%">'
    '<stop offset="0%" stop-color="#FFFFFF"/>'
    '<stop offset="50%" stop-color="#00F2FE"/>'
    '<stop offset="100%" stop-color="#FF1493"/>'
    '</linearGradient>'
    '</defs>'
    '<circle cx="32" cy="32" r="16" fill="url(#gPlanet)"/>'
    '<ellipse cx="26" cy="25" rx="6" ry="3.5" transform="rotate(-25 26 25)" fill="#FFFFFF" opacity="0.65"/>'
    '<path d="M8 38C12 46 52 34 56 24" stroke="url(#gRing)" stroke-width="4.5" stroke-linecap="round"/>'
    '<circle cx="48" cy="18" r="3" fill="#FFFFFF"/>'
    '</svg>'
    '</div>'
)

ICON_CHAT_3D = (
    '<span class="crystal-icon-sm">'
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none">'
    '<path d="M12 3C6.5 3 2 6.6 2 11c0 2.1 1 4 2.6 5.4L4 21l4.8-1.9c1 .3 2.1.4 3.2.4 5.5 0 10-3.6 10-8.5S17.5 3 12 3z" fill="url(#chatG)"/>'
    '<circle cx="8.5" cy="11" r="1.4" fill="#FFF"/><circle cx="12" cy="11" r="1.4" fill="#FFF"/><circle cx="15.5" cy="11" r="1.4" fill="#FFF"/>'
    '<defs><linearGradient id="chatG" x1="0" y1="0" x2="24" y2="24"><stop stop-color="#00F2FE"/><stop offset="1" stop-color="#EC4899"/></linearGradient></defs>'
    '</svg>'
    '</span>'
)

ICON_DOCS_3D = (
    '<span class="crystal-icon-sm">'
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none">'
    '<rect x="3" y="4" width="18" height="16" rx="4" fill="url(#docG)"/>'
    '<circle cx="12" cy="12" r="4.5" stroke="#FFF" stroke-width="2.2"/>'
    '<defs><linearGradient id="docG" x1="0" y1="0" x2="24" y2="24"><stop stop-color="#38BDF8"/><stop offset="1" stop-color="#A855F7"/></linearGradient></defs>'
    '</svg>'
    '</span>'
)

ICON_TUNE_3D = (
    '<span class="crystal-icon-sm">'
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none">'
    '<path d="M4 12C4 9.5 6 7.5 8.5 7.5C10.2 7.5 11.5 8.8 12 10.5C12.5 12.2 13.8 13.5 15.5 13.5C18 13.5 20 11.5 20 9" stroke="#00F2FE" stroke-width="3" stroke-linecap="round"/>'
    '<circle cx="8.5" cy="14" r="3" fill="#F472B6"/><circle cx="15.5" cy="7" r="3" fill="#00F2FE"/>'
    '</svg>'
    '</span>'
)

st.markdown(
    f'<div class="pic2-glass-hero">'
    f'<div style="display:flex;align-items:center;gap:20px;">'
    f'{ICON_HERO_3D}'
    f'<div>'
    f'<div class="pic2-title">SHEHERYAR ILYAS — DIGITAL CLONE</div>'
    f'<div class="pic2-subtitle">BSCS (7th Semester) @ UMT Sialkot · Hybrid RAG (BM25 + Dense Embeddings + RRF) · Gemini 2.5 Flash</div>'
    f'</div>'
    f'</div>'
    f'<div style="display:flex;gap:12px;flex-wrap:wrap;">'
    f'<span class="pic2-pill">● LIVE HYBRID RAG</span>'
    f'<span class="pic2-pill">CLO-2 AI CLONE</span>'
    f'</div>'
    f'</div>'
    f'<div class="pic2-banner">'
    f'<span>⚠️ <strong>This is an AI clone, not the real person.</strong> It answers strictly in first person using Sheheryar Ilyas\'s personal documents.</span>'
    f'<span style="font-family:\'Montserrat\',sans-serif;font-weight:700;font-size:0.76rem;letter-spacing:0.08em;color:#7DD3FC;">100% GROUNDED</span>'
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
    st.markdown(f'<div class="crystal-sec-header">{ICON_TUNE_3D}<span class="crystal-sec-title">Retrieval Controls</span></div>', unsafe_allow_html=True)
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
    st.markdown(f'<div class="crystal-sec-header">{ICON_DOCS_3D}<span class="crystal-sec-title">Bonus Capabilities</span></div>', unsafe_allow_html=True)
    use_agentic_router = st.checkbox("Auto-Override Mode with Agentic Router", value=False)
    use_reranker = st.checkbox("Enable Cross-Encoder Reranker", value=False)

    st.divider()
    st.markdown(
        f'<div class="pic2-services-container">'
        f'<div style="font-family:\'Montserrat\',sans-serif;font-size:0.74rem;font-weight:800;color:#FFFFFF;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;">System Telemetry</div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:0.84rem;"><span>Indexed Docs</span><strong style="color:#7DD3FC;">{num_docs} files</strong></div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:0.84rem;"><span>Overlapping Chunks</span><strong style="color:#7DD3FC;">{len(chunks)} chunks</strong></div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.84rem;"><span>Dense Vector</span><strong style="color:#F9A8D4;">384-D (.npy)</strong></div>'
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
    col_chat, col_chunks = st.columns([1.15, 1.0], gap="large")

    with col_chat:
        st.markdown(f'<div class="crystal-sec-header">{ICON_CHAT_3D}<span class="crystal-sec-title">Diagnostic Query Console</span></div>', unsafe_allow_html=True)
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
            with st.chat_message(msg["role"]):
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

    with col_chunks:
        st.markdown(f'<div class="crystal-sec-header">{ICON_DOCS_3D}<span class="crystal-sec-title">Retrieved Chunks, Sources & Scores</span></div>', unsafe_allow_html=True)

        if "last_route_info" in st.session_state:
            cat, m_used, rec_m, r_reason = st.session_state["last_route_info"]
            st.markdown(
                f'<div class="pic2-services-container">'
                f'<div style="font-family:\'Montserrat\',sans-serif;font-size:0.82rem;font-weight:700;color:#FFFFFF;">Active Mode: {html.escape(m_used)} · Intent: {html.escape(cat)} (Recommended: {html.escape(rec_m)})</div>'
                f'<div style="font-size:0.8rem;color:#E2E8F0;margin-top:4px;">{html.escape(r_reason)}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        if "last_retrieved" in st.session_state and st.session_state["last_retrieved"]:
            for rank, item in enumerate(st.session_state["last_retrieved"], start=1):
                c = item["chunk"]
                ce_str = f"{item['rerank_score']:.3f}" if item["rerank_score"] is not None else "OFF"
                safe_passage = html.escape(c["text"])

                with st.expander(f"Rank #{rank} · {c['id']} · Score: {item['score']:.4f}", expanded=(rank == 1)):
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
                        f'<span class="pic2-pill" style="padding:5px 14px;font-size:0.7rem;">Source: {html.escape(c["source"])}</span>'
                        f'<span class="pic2-pill" style="padding:5px 14px;font-size:0.7rem;">{c["word_count"]} Words</span>'
                        f'</div>'
                        f'<div class="pic2-services-container">'
                        f'<div class="pic2-service-grid">'
                        f'<div class="pic2-service-box"><span class="title">BM25 Score</span><span class="val">{item["bm25_score"]:.2f}</span><span class="sub">Rank #{item["bm25_rank"]}</span></div>'
                        f'<div class="pic2-service-box"><span class="title">Dense Cosine</span><span class="val">{item["dense_score"]:.3f}</span><span class="sub">Rank #{item["dense_rank"]}</span></div>'
                        f'<div class="pic2-service-box"><span class="title">Hybrid RRF</span><span class="val">{item["rrf_score"]:.4f}</span><span class="sub">Weighted: {item["weighted_score"]:.2f}</span></div>'
                        f'<div class="pic2-service-box"><span class="title">Reranker</span><span class="val">{ce_str}</span><span class="sub">Cross-Enc</span></div>'
                        f'</div>'
                        f'</div>'
                        f'<div class="pic2-passage-box">{safe_passage}</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.markdown(
                '<div class="pic2-glass-hero" style="text-align:center;justify-content:center;padding:42px 28px;">'
                '<div>'
                '<div style="font-family:\'Montserrat\',sans-serif;font-weight:800;font-size:1.05rem;color:#FFFFFF;letter-spacing:0.06em;text-transform:uppercase;">Awaiting Query Execution</div>'
                '<div style="color:#E2E8F0;font-size:0.86rem;margin-top:8px;max-width:420px;">Select any diagnostic query card on the left to inspect BM25, Dense, and Hybrid RRF scores and source passages inside frosted glass service cards.</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

with tab2:
    st.markdown(f'<div class="crystal-sec-header">{ICON_TUNE_3D}<span class="crystal-sec-title">Hand-Labeled Evaluation Set (Hit@3) & 5 Test Questions</span></div>', unsafe_allow_html=True)
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
                f'<td style="font-weight:700;color:#FFFFFF;">{html.escape(r["Question"])}</td>'
                f'<td style="font-family:\'JetBrains Mono\',monospace;color:#7DD3FC;">{html.escape(r["BM25 Top-1"])}</td>'
                f'<td style="font-family:\'JetBrains Mono\',monospace;color:#E9D5FF;">{html.escape(r["Dense Top-1"])}</td>'
                f'<td style="font-family:\'JetBrains Mono\',monospace;color:#F9A8D4;font-weight:700;">{html.escape(r["Hybrid Top-1"])}</td>'
                f'<td>{html.escape(r["Best Mode & Why"])}</td>'
                f'</tr>'
            )

        st.markdown(
            f'<div class="pic2-table-wrap" style="margin-top:18px;">'
            f'<table class="pic2-table">'
            f'<thead><tr><th>Test Question</th><th>BM25 Result</th><th>Dense Result</th><th>Hybrid Result</th><th>Best Mode & Why</th></tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True
        )

with tab3:
    st.markdown(f'<div class="crystal-sec-header">{ICON_DOCS_3D}<span class="crystal-sec-title">Indexed Knowledge Base ({num_docs} Documents · {len(chunks)} Chunks)</span></div>', unsafe_allow_html=True)
    for c in chunks:
        with st.expander(f"{c['id']} — {c['source']} ({c['word_count']} words)"):
            st.markdown(f'<div class="pic2-passage-box">{html.escape(c["text"])}</div>', unsafe_allow_html=True)
