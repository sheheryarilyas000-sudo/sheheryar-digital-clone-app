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
    page_title="Sheheryar Ilyas | Futuristic AI Clone",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# PIC-2 EXACT MATCH: FUTURISTIC NEON TECH GLASSMORPHISM + PURE CSS LASERS
# Zero indented HTML blocks so Streamlit never leaks raw code on screen!
# =========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

/* 1. Deep Midnight Space Base (Matches Pic 2 Background) */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 22% 18%, #101942 0%, #07051a 48%, #04020e 100%) !important;
    color: #F8FAFC !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* 2. Top-Left Cyan Laser Beam & Bottom-Right Magenta Laser Beam via Pure CSS */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: -120px;
    left: -140px;
    width: 520px;
    height: 12px;
    background: linear-gradient(90deg, transparent 0%, #00f2fe 35%, #ffffff 50%, #00c6ff 75%, transparent 100%);
    box-shadow: 0 0 25px 6px #00f2fe, 0 0 70px 25px rgba(0, 212, 255, 0.55), 0 0 140px 55px rgba(0, 119, 255, 0.35);
    transform: rotate(-38deg);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    bottom: -100px;
    right: -140px;
    width: 540px;
    height: 12px;
    background: linear-gradient(90deg, transparent 0%, #ff0844 25%, #ffffff 50%, #ff007f 75%, transparent 100%);
    box-shadow: 0 0 25px 6px #ff007f, 0 0 75px 28px rgba(255, 0, 127, 0.55), 0 0 150px 60px rgba(168, 85, 247, 0.4);
    transform: rotate(-38deg);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stMain"] {
    background: transparent !important;
    position: relative;
    z-index: 1;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1380px !important;
}

/* 3. Frosted Neon Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(12, 10, 36, 0.94) 0%, rgba(7, 5, 22, 0.97) 100%) !important;
    border-right: 1px solid rgba(0, 242, 254, 0.22) !important;
    box-shadow: 8px 0 35px rgba(0, 0, 0, 0.65) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #E2E8F0 !important;
}

/* Fix Selectbox Dropdown Visibility in Dark Mode */
[data-baseweb="select"] > div {
    background: rgba(19, 16, 48, 0.9) !important;
    border: 1px solid rgba(0, 242, 254, 0.35) !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
}
[data-baseweb="select"] span {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}
[data-baseweb="popover"] ul {
    background: #110e2e !important;
    border: 1px solid rgba(0, 242, 254, 0.4) !important;
}
[data-baseweb="popover"] li {
    color: #F8FAFC !important;
}
[data-baseweb="popover"] li:hover {
    background: rgba(0, 242, 254, 0.2) !important;
}

/* 4. Pic-2 Centerpiece Frosted Glass Card with Cyan-to-Magenta Rim Border */
.neon-hero-shell {
    background: linear-gradient(135deg, rgba(22, 32, 82, 0.62) 0%, rgba(42, 18, 78, 0.58) 55%, rgba(88, 18, 92, 0.52) 100%);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border-radius: 24px;
    padding: 24px 30px;
    margin-bottom: 16px;
    border: 1px solid rgba(0, 242, 254, 0.38);
    box-shadow:
        -10px -10px 35px rgba(0, 242, 254, 0.14),
        12px 12px 40px rgba(255, 0, 127, 0.18),
        0 20px 50px rgba(0, 0, 0, 0.65),
        inset 0 1px 1px rgba(255, 255, 255, 0.28);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 18px;
}

.hero-left {
    display: flex;
    align-items: center;
    gap: 18px;
}

.icon-3d-orb {
    width: 62px;
    height: 62px;
    border-radius: 18px;
    background: radial-gradient(circle at 30% 25%, #38bdf8 0%, #3b82f6 40%, #7e22ce 85%, #db2777 100%);
    box-shadow:
        0 10px 25px rgba(0, 242, 254, 0.35),
        0 0 25px rgba(236, 72, 153, 0.35),
        inset 0 2px 4px rgba(255, 255, 255, 0.65),
        inset 0 -4px 8px rgba(0, 0, 0, 0.45);
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(255, 255, 255, 0.35);
}

.hero-heading {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: #FFFFFF !important;
    text-shadow: 0 2px 18px rgba(0, 242, 254, 0.35);
    margin: 0;
    line-height: 1.15;
}

.hero-sub {
    font-size: 0.84rem;
    color: #CBD5E1 !important;
    font-weight: 500;
    margin-top: 4px;
    letter-spacing: 0.04em;
}

.pill-group {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.neon-pill-cyan {
    background: linear-gradient(135deg, rgba(0, 242, 254, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);
    border: 1px solid rgba(0, 242, 254, 0.55);
    color: #7DD3FC !important;
    padding: 7px 16px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    box-shadow: 0 0 18px rgba(0, 242, 254, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.25);
}

.neon-pill-pink {
    background: linear-gradient(135deg, rgba(236, 72, 153, 0.22) 0%, rgba(168, 85, 247, 0.22) 100%);
    border: 1px solid rgba(244, 114, 182, 0.55);
    color: #F9A8D4 !important;
    padding: 7px 16px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    box-shadow: 0 0 18px rgba(236, 72, 153, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.25);
}

/* 5. Mandatory AI Clone Disclaimer Banner (Glassmorphic Neon Border) */
.clone-disclaimer-glass {
    background: linear-gradient(90deg, rgba(16, 24, 64, 0.78) 0%, rgba(56, 18, 72, 0.78) 100%);
    border: 1px solid rgba(244, 114, 182, 0.45);
    border-left: 4px solid #00f2fe;
    border-radius: 14px;
    padding: 11px 18px;
    margin-bottom: 18px;
    color: #F1F5F9 !important;
    font-size: 0.85rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
}

/* 6. Section Headers with 3D Mini-Badge */
.sec-head {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.88rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #FFFFFF !important;
    margin-bottom: 12px;
}
.sec-dot-3d {
    width: 12px;
    height: 12px;
    border-radius: 4px;
    background: radial-gradient(circle at 30% 30%, #00f2fe, #9333ea);
    box-shadow: 0 0 10px #00f2fe;
    display: inline-block;
}

/* 7. Tabs (Futuristic Glass Segmented Bar) */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 14, 42, 0.82) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 16px !important;
    padding: 6px !important;
    gap: 8px !important;
    margin-bottom: 18px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #94A3B8 !important;
    border-radius: 11px !important;
    padding: 9px 20px !important;
    font-weight: 700 !important;
    font-size: 0.84rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.28) 0%, rgba(217, 70, 239, 0.32) 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(0, 242, 254, 0.55) !important;
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.25) !important;
}

/* 8. Pic-2 Service-Card Style Buttons (High-Contrast White Text on Deep Translucent Violet) */
.stButton > button {
    background: linear-gradient(145deg, rgba(26, 27, 68, 0.88) 0%, rgba(44, 19, 68, 0.85) 100%) !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 14px !important;
    padding: 11px 15px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-align: left !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.18) !important;
    transition: all 0.22s ease !important;
    min-height: 46px !important;
}
.stButton > button * {
    color: #F8FAFC !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.28) 0%, rgba(236, 72, 153, 0.32) 100%) !important;
    border-color: #00f2fe !important;
    box-shadow: 0 0 22px rgba(0, 242, 254, 0.38), 0 0 28px rgba(236, 72, 153, 0.25) !important;
    transform: translateY(-2px) !important;
}

/* Download Button Special Glow */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(90deg, rgba(0, 198, 255, 0.25) 0%, rgba(168, 85, 247, 0.35) 100%) !important;
    border: 1px solid rgba(0, 242, 254, 0.5) !important;
    color: #FFFFFF !important;
    text-align: center !important;
}

/* 9. Chat Message Bubbles (High Contrast) */
[data-testid="stChatMessage"] {
    background: linear-gradient(135deg, rgba(18, 20, 52, 0.88) 0%, rgba(30, 14, 54, 0.85) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 16px !important;
    padding: 14px 18px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.45) !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] span {
    color: #F8FAFC !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
}

/* Chat Input Bar */
[data-testid="stChatInput"] {
    background: rgba(16, 14, 44, 0.92) !important;
    border: 1px solid rgba(0, 242, 254, 0.45) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 25px rgba(0, 242, 254, 0.18), 0 8px 24px rgba(0, 0, 0, 0.6) !important;
}
[data-testid="stChatInput"] textarea {
    color: #FFFFFF !important;
}

/* 10. Retrieved Chunk Expanders & Cards */
[data-testid="stExpander"] {
    background: linear-gradient(145deg, rgba(17, 20, 52, 0.86) 0%, rgba(36, 15, 62, 0.82) 100%) !important;
    border: 1px solid rgba(0, 242, 254, 0.25) !important;
    border-radius: 16px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.45) !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.86rem !important;
}

/* 11. 3-Service-Box Style Score Grid Inside Retrieved Passages */
.service-score-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 10px 0 14px 0;
}
.service-score-box {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 12px;
    padding: 10px 8px;
    text-align: center;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2);
}
.service-score-box .s-lbl {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #93C5FD !important;
    font-weight: 700;
    display: block;
}
.service-score-box .s-val {
    font-size: 0.96rem;
    font-weight: 800;
    color: #FFFFFF !important;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 3px;
    display: block;
}
.service-score-box .s-rk {
    font-size: 0.66rem;
    color: #F472B6 !important;
    font-family: 'JetBrains Mono', monospace;
}

.passage-glass-body {
    background: rgba(7, 6, 22, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-left: 3px solid #00f2fe;
    border-radius: 12px;
    padding: 12px 14px;
    color: #E2E8F0 !important;
    font-size: 0.84rem;
    line-height: 1.65;
    font-weight: 400;
}

/* 12. Metric Cards & Benchmark Table */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(20, 26, 68, 0.85) 0%, rgba(48, 18, 74, 0.82) 100%) !important;
    border: 1px solid rgba(0, 242, 254, 0.35) !important;
    border-radius: 18px !important;
    padding: 18px 20px !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.22) !important;
}
[data-testid="stMetricLabel"] * {
    color: #93C5FD !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] * {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    text-shadow: 0 0 20px rgba(0, 242, 254, 0.45) !important;
}

.neon-eval-table {
    width: 100%;
    border-collapse: collapse;
    background: rgba(14, 13, 38, 0.88);
    border: 1px solid rgba(0, 242, 254, 0.28);
    border-radius: 16px;
    overflow: hidden;
    font-size: 0.83rem;
}
.neon-eval-table th {
    background: linear-gradient(90deg, rgba(0, 212, 255, 0.18) 0%, rgba(217, 70, 239, 0.18) 100%);
    color: #7DD3FC !important;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 13px 15px;
    text-align: left;
    border-bottom: 1px solid rgba(0, 242, 254, 0.25);
}
.neon-eval-table td {
    padding: 12px 15px;
    color: #F1F5F9 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    vertical-align: top;
}
</style>
""", unsafe_allow_html=True)

# =========================================================================
# VOLUMETRIC 3D HERO BANNER (ZERO 4-SPACE INDENTATION SO NO RAW HTML LEAKS)
# =========================================================================
SVG_3D_CORE = (
    '<svg width="36" height="36" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<defs>'
    '<radialGradient id="orbGrad" cx="30%" cy="25%" r="75%">'
    '<stop offset="0%" stop-color="#FFFFFF"/>'
    '<stop offset="30%" stop-color="#00F2FE"/>'
    '<stop offset="70%" stop-color="#7C3AED"/>'
    '<stop offset="100%" stop-color="#FF007F"/>'
    '</radialGradient>'
    '<linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#00F2FE"/>'
    '<stop offset="100%" stop-color="#FF007F"/>'
    '</linearGradient>'
    '</defs>'
    '<ellipse cx="32" cy="32" rx="28" ry="12" transform="rotate(-28 32 32)" stroke="url(#ringGrad)" stroke-width="3.2" opacity="0.9"/>'
    '<ellipse cx="32" cy="32" rx="28" ry="12" transform="rotate(28 32 32)" stroke="url(#ringGrad)" stroke-width="3.2" opacity="0.9"/>'
    '<circle cx="32" cy="32" r="13" fill="url(#orbGrad)"/>'
    '<circle cx="27" cy="27" r="4" fill="#FFFFFF" opacity="0.75"/>'
    '</svg>'
)

st.markdown(
    f'<div class="neon-hero-shell">'
    f'<div class="hero-left">'
    f'<div class="icon-3d-orb">{SVG_3D_CORE}</div>'
    f'<div>'
    f'<div class="hero-heading">SHEHERYAR ILYAS — DIGITAL CLONE</div>'
    f'<div class="hero-sub">BSCS (7TH SEM) @ UMT SIALKOT · HYBRID RAG (BM25 + DENSE EMBEDDINGS + RRF) · GEMINI 2.5 FLASH</div>'
    f'</div>'
    f'</div>'
    f'<div class="pill-group">'
    f'<span class="neon-pill-cyan">● LIVE HYBRID RAG</span>'
    f'<span class="neon-pill-pink">CLO-2 AI CLONE</span>'
    f'</div>'
    f'</div>'
    f'<div class="clone-disclaimer-glass">'
    f'<span>⚠️ <strong>This is an AI clone, not the real person.</strong> It answers strictly in first person using Sheheryar Ilyas\'s personal documents.</span>'
    f'<span style="color:#7DD3FC;font-family:\'JetBrains Mono\',monospace;font-size:0.76rem;">100% GROUNDED</span>'
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
    st.markdown('<div class="sec-head"><span class="sec-dot-3d"></span>Retrieval Controls</div>', unsafe_allow_html=True)
    retrieval_mode = st.radio(
        "Retrieval Mode Selector:",
        options=["Hybrid", "BM25 only", "Dense only"],
        index=0
    )
    fusion_type = st.selectbox(
        "Hybrid Fusion Strategy:",
        options=["RRF", "Weighted"],
        index=0,
        disabled=(retrieval_mode != "Hybrid")
    )
    alpha = 0.5
    if retrieval_mode == "Hybrid" and fusion_type == "Weighted":
        alpha = st.slider("Dense Weight (alpha):", 0.0, 1.0, 0.5, 0.05)

    top_k = st.slider("Top-K Chunks:", 1, 5, 3)

    st.divider()
    st.markdown('<div class="sec-head"><span class="sec-dot-3d"></span>Strong Student Extras</div>', unsafe_allow_html=True)
    use_agentic_router = st.checkbox("Auto-Override Mode with Agentic Router", value=False)
    use_reranker = st.checkbox("Enable Cross-Encoder Reranker", value=False)

    st.divider()
    st.markdown(
        f'<div style="background:linear-gradient(145deg,rgba(18,24,62,0.85),rgba(38,14,62,0.85));border:1px solid rgba(0,242,254,0.3);border-radius:14px;padding:14px 16px;">'
        f'<div style="font-size:0.72rem;font-weight:800;color:#7DD3FC;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">System Telemetry</div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:0.82rem;"><span>Indexed Docs</span><strong style="color:#00f2fe;">{num_docs} files</strong></div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:0.82rem;"><span>Overlapping Chunks</span><strong style="color:#00f2fe;">{len(chunks)} chunks</strong></div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;"><span>Dense Vector</span><strong style="color:#F472B6;">384-D (.npy)</strong></div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.divider()
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
        st.markdown('<div class="sec-head"><span class="sec-dot-3d"></span>Diagnostic Query Console</div>', unsafe_allow_html=True)
        st.caption("Select a preset diagnostic test below or type any question to chat with Sheheryar's AI clone:")

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
        st.markdown('<div class="sec-head"><span class="sec-dot-3d"></span>Retrieved Chunks, Sources & Scores</div>', unsafe_allow_html=True)

        if "last_route_info" in st.session_state:
            cat, m_used, rec_m, r_reason = st.session_state["last_route_info"]
            st.markdown(
                f'<div style="background:linear-gradient(90deg,rgba(0,242,254,0.12),rgba(236,72,153,0.12));border:1px solid rgba(0,242,254,0.35);border-radius:14px;padding:12px 16px;margin-bottom:12px;">'
                f'<div style="font-size:0.82rem;font-weight:700;color:#7DD3FC;">Active Mode: {html.escape(m_used)} · Intent: {html.escape(cat)} (Recommended: {html.escape(rec_m)})</div>'
                f'<div style="font-size:0.78rem;color:#CBD5E1;margin-top:3px;">{html.escape(r_reason)}</div>'
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
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                        f'<span class="neon-pill-cyan" style="padding:3px 10px;font-size:0.7rem;">Source: {html.escape(c["source"])}</span>'
                        f'<span class="neon-pill-pink" style="padding:3px 10px;font-size:0.7rem;">{c["word_count"]} Words</span>'
                        f'</div>'
                        f'<div class="service-score-grid">'
                        f'<div class="service-score-box"><span class="s-lbl">BM25 Score</span><span class="s-val">{item["bm25_score"]:.2f}</span><span class="s-rk">Rank #{item["bm25_rank"]}</span></div>'
                        f'<div class="service-score-box"><span class="s-lbl">Dense Cosine</span><span class="s-val">{item["dense_score"]:.3f}</span><span class="s-rk">Rank #{item["dense_rank"]}</span></div>'
                        f'<div class="service-score-box"><span class="s-lbl">Hybrid RRF</span><span class="s-val">{item["rrf_score"]:.4f}</span><span class="s-rk">Weighted: {item["weighted_score"]:.2f}</span></div>'
                        f'<div class="service-score-box"><span class="s-lbl">Reranker</span><span class="s-val">{ce_str}</span><span class="s-rk">Cross-Enc</span></div>'
                        f'</div>'
                        f'<div class="passage-glass-body">{safe_passage}</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.markdown(
                '<div style="background:linear-gradient(135deg,rgba(20,26,68,0.65),rgba(44,16,68,0.65));border:1px dashed rgba(0,242,254,0.35);border-radius:20px;padding:44px 24px;text-align:center;">'
                '<div style="font-weight:800;font-size:1rem;color:#FFFFFF;letter-spacing:0.04em;text-transform:uppercase;">Awaiting Query Execution</div>'
                '<div style="color:#CBD5E1;font-size:0.84rem;margin-top:6px;">Select any diagnostic query button on the left to inspect BM25, Dense, and Hybrid RRF scores and source passages here.</div>'
                '</div>',
                unsafe_allow_html=True
            )

with tab2:
    st.markdown('<div class="sec-head"><span class="sec-dot-3d"></span>Hand-Labeled Evaluation Set (Hit@3) & 5 Test Questions</div>', unsafe_allow_html=True)
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
                f'<td style="font-family:\'JetBrains Mono\',monospace;color:#C4B5FD;">{html.escape(r["Dense Top-1"])}</td>'
                f'<td style="font-family:\'JetBrains Mono\',monospace;color:#F472B6;font-weight:700;">{html.escape(r["Hybrid Top-1"])}</td>'
                f'<td>{html.escape(r["Best Mode & Why"])}</td>'
                f'</tr>'
            )

        st.markdown(
            f'<div style="margin-top:16px;">'
            f'<table class="neon-eval-table">'
            f'<thead><tr><th>Test Question</th><th>BM25 Result</th><th>Dense Result</th><th>Hybrid Result</th><th>Best Mode & Why</th></tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True
        )

with tab3:
    st.markdown(f'<div class="sec-head"><span class="sec-dot-3d"></span>Indexed Knowledge Base ({num_docs} Documents · {len(chunks)} Chunks)</div>', unsafe_allow_html=True)
    for c in chunks:
        with st.expander(f"{c['id']} — {c['source']} ({c['word_count']} words)"):
            st.markdown(f'<div class="passage-glass-body">{html.escape(c["text"])}</div>', unsafe_allow_html=True)
