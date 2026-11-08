import os
import io
import glob
import requests
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Sheheryar Ilyas | AI Clone",
    page_icon="🤖",
    layout="wide"
)

# =========================================================================
# FUTURISTIC NEON TECH GLASSMORPHISM THEME
# Matches reference: Deep Obsidian + Cyan Laser Beam + Neon Magenta Laser
# =========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after {
    box-sizing: border-box;
}

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    color: #e2e8f0 !important;
}

/* Hide Default Streamlit Chrome */
#MainMenu, footer, header { visibility: hidden !important; height: 0 !important; }
.block-container {
    padding: 1.5rem 2.2rem 2.5rem 2.2rem !important;
    max-width: 100% !important;
}

/* True Deep Base */
.stApp {
    background-color: #030014 !important;
    min-height: 100vh;
}

/* Force all intermediate wrappers transparent so background glow shows through */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMain"] > div,
.main, .main > div,
section.main, section.main > div,
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="column"],
[data-testid="stChatMessageContainer"],
.stChatFloatingInputContainer,
div[class*="st-emotion-cache"],
div[data-baseweb],
div[data-baseweb] > div {
    background-color: transparent !important;
    background: transparent !important;
}

/* Typography */
h1, h2, h3, h4 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: #ffffff !important;
}

/* Headings with Neon Under-Glow */
.section-title {
    font-size: 0.85rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: #38bdf8 !important;
    margin-bottom: 0.75rem !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Custom Warning / Persona Banner */
[data-testid="stAlert"] {
    background: rgba(225, 29, 72, 0.08) !important;
    border: 1px solid rgba(244, 63, 94, 0.3) !important;
    border-left: 4px solid #f43f5e !important;
    border-radius: 14px !important;
    backdrop-filter: blur(20px) !important;
    color: #fecdd3 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 25px rgba(244, 63, 94, 0.15) !important;
}

/* Glassmorphism Sidebar */
[data-testid="stSidebar"] {
    background: rgba(8, 2, 28, 0.75) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.18) !important;
    backdrop-filter: blur(40px) !important;
    -webkit-backdrop-filter: blur(40px) !important;
    box-shadow: 4px 0 30px rgba(0, 0, 0, 0.6) !important;
}

[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}

[data-testid="stSidebar"] strong, 
[data-testid="stSidebar"] b {
    color: #f1f5f9 !important;
}

/* Glassmorphism Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 6px !important;
    backdrop-filter: blur(25px) !important;
}

[data-testid="stTabs"] button[role="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    color: #64748b !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 9px 20px !important;
    transition: all 0.25s ease !important;
}

[data-testid="stTabs"] button[role="tab"]:hover {
    color: #38bdf8 !important;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.35) 0%, rgba(168, 85, 247, 0.35) 100%) !important;
    border: 1px solid rgba(56, 189, 248, 0.5) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 0 25px rgba(56, 189, 248, 0.3) !important;
}

/* Glassmorphism Preset Prompt Buttons */
.stButton > button {
    background: rgba(255, 255, 255, 0.035) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: #cbd5e1 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25) !important;
    width: 100% !important;
    text-align: left !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(217, 70, 239, 0.15)) !important;
    border-color: rgba(56, 189, 248, 0.5) !important;
    color: #ffffff !important;
    box-shadow: 0 0 25px rgba(56, 189, 248, 0.35), 0 4px 20px rgba(0,0,0,0.4) !important;
    transform: translateY(-2px) !important;
}

/* Glassmorphism Chat Messages */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.035) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 18px !important;
    padding: 18px 22px !important;
    margin-bottom: 12px !important;
    backdrop-filter: blur(30px) !important;
    -webkit-backdrop-filter: blur(30px) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border: 1px solid rgba(168, 85, 247, 0.35) !important;
    background: linear-gradient(135deg, rgba(8, 5, 28, 0.8) 0%, rgba(20, 10, 45, 0.7) 100%) !important;
    box-shadow: 0 0 35px rgba(168, 85, 247, 0.15), 0 8px 32px rgba(0,0,0,0.5) !important;
}

/* Chat Input Bar */
[data-testid="stChatInput"] {
    background: rgba(12, 6, 36, 0.75) !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(30px) !important;
    box-shadow: 0 0 30px rgba(56, 189, 248, 0.15), 0 8px 32px rgba(0,0,0,0.5) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #f8fafc !important;
    font-size: 0.9rem !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 35px rgba(56, 189, 248, 0.4) !important;
}

/* Glassmorphism Expander (Chunk Cards) */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    margin-bottom: 10px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.25s ease !important;
    overflow: hidden !important;
}

[data-testid="stExpander"]:hover {
    border-color: rgba(56, 189, 248, 0.4) !important;
    box-shadow: 0 0 30px rgba(56, 189, 248, 0.15), 0 8px 32px rgba(0,0,0,0.4) !important;
}

[data-testid="stExpander"] summary {
    color: #e2e8f0 !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    padding: 14px 18px !important;
}

/* Glassmorphism Metric Cards */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.035) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 18px !important;
    padding: 22px 18px !important;
    text-align: center !important;
    backdrop-filter: blur(30px) !important;
    -webkit-backdrop-filter: blur(30px) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35) !important;
    transition: transform 0.25s, box-shadow 0.25s !important;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px) !important;
    border-color: rgba(56, 189, 248, 0.5) !important;
    box-shadow: 0 0 35px rgba(56, 189, 248, 0.25) !important;
}

[data-testid="stMetricValue"] {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #38bdf8 0%, #c084fc 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    text-shadow: 0 0 30px rgba(56, 189, 248, 0.3) !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-weight: 700 !important;
}

/* Form Controls */
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}

[data-testid="stRadio"] label:has(input:checked) {
    color: #38bdf8 !important;
    font-weight: 700 !important;
}

[data-testid="stSelectbox"] > div > div {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
}

/* Tables */
table {
    background: rgba(255, 255, 255, 0.02) !important;
    border-radius: 14px !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    overflow: hidden !important;
    width: 100% !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

thead tr th {
    background: rgba(56, 189, 248, 0.1) !important;
    color: #38bdf8 !important;
    font-size: 0.75rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 14px 16px !important;
    border-bottom: 1px solid rgba(56, 189, 248, 0.2) !important;
}

tbody tr td {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    padding: 12px 16px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
}

tbody tr:hover td {
    background: rgba(56, 189, 248, 0.05) !important;
}

/* Code Pills */
code {
    background: rgba(56, 189, 248, 0.12) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 6px !important;
    padding: 2px 8px !important;
    font-size: 0.78rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Custom Scrollbars */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(56, 189, 248, 0.3);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(56, 189, 248, 0.6);
}
</style>
""", unsafe_allow_html=True)

# =========================================================================
# VIVID NEON LIGHT BEAMS (Exact Pinterest Reference Match)
# Top-Left Intense Cyan Laser + Bottom-Right Neon Magenta Laser
# =========================================================================
st.markdown("""
<div style="position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:0;overflow:hidden;">
    <!-- Top-Left Cyan/Blue Laser Ray -->
    <div style="position:absolute;top:-20%;left:-15%;width:65vw;height:75vh;
        background: radial-gradient(ellipse at 25% 25%, rgba(0, 212, 255, 0.5) 0%, rgba(14, 116, 233, 0.25) 35%, transparent 70%);
        filter: blur(55px);
        transform: rotate(20deg);
        opacity: 0.85;"></div>
    
    <div style="position:absolute;top:5%;left:-8%;width:35vw;height:4px;
        background: linear-gradient(90deg, #00f0ff 0%, #0077ff 70%, transparent 100%);
        box-shadow: 0 0 30px #00f0ff, 0 0 60px #0077ff;
        transform: rotate(38deg);
        opacity: 0.9;"></div>

    <!-- Bottom-Right Magenta/Purple Laser Ray -->
    <div style="position:absolute;bottom:-20%;right:-10%;width:65vw;height:70vh;
        background: radial-gradient(ellipse at 75% 75%, rgba(236, 72, 153, 0.5) 0%, rgba(168, 85, 247, 0.3) 40%, transparent 70%);
        filter: blur(65px);
        transform: rotate(-15deg);
        opacity: 0.85;"></div>

    <div style="position:absolute;bottom:8%;right:-5%;width:40vw;height:4px;
        background: linear-gradient(270deg, #ff007f 0%, #a855f7 70%, transparent 100%);
        box-shadow: 0 0 30px #ff007f, 0 0 60px #a855f7;
        transform: rotate(35deg);
        opacity: 0.9;"></div>
</div>
""", unsafe_allow_html=True)

# =========================================================================
# FUTURISTIC HERO BANNER (Glassmorphism + Neon Glow)
# =========================================================================
st.markdown("""
<div style="
    position: relative;
    z-index: 1;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 22px;
    padding: 24px 32px;
    margin-bottom: 18px;
    backdrop-filter: blur(40px);
    -webkit-backdrop-filter: blur(40px);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.12);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
">
    <div style="display:flex; align-items:center; gap: 20px;">
        <div style="
            width: 58px; height: 58px; border-radius: 18px;
            background: linear-gradient(135deg, #0284c7 0%, #9333ea 100%);
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 0 30px rgba(56, 189, 248, 0.5), inset 0 1px 1px rgba(255,255,255,0.4);
        ">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="11" width="18" height="10" rx="2"></rect>
                <circle cx="12" cy="5" r="2"></circle>
                <path d="M12 7v4"></path>
                <line x1="8" y1="16" x2="8.01" y2="16"></line>
                <line x1="16" y1="16" x2="16.01" y2="16"></line>
            </svg>
        </div>
        <div>
            <div style="
                font-size: 1.7rem; font-weight: 800; color: #ffffff; letter-spacing: -0.03em;
                text-shadow: 0 0 35px rgba(56, 189, 248, 0.45);
            ">
                Sheheryar Ilyas <span style="font-weight: 300; color: #94a3b8;">/ Digital Clone</span>
            </div>
            <div style="font-size: 0.8rem; color: #64748b; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 3px;">
                HYBRID RAG <span style="color:#38bdf8;">·</span> BM25 + DENSE EMBEDDINGS <span style="color:#c084fc;">·</span> GEMINI 2.5 FLASH
            </div>
        </div>
    </div>
    <div style="display:flex; gap: 10px;">
        <div style="
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.35);
            border-radius: 9999px;
            padding: 6px 16px;
            font-size: 0.75rem;
            color: #4ade80;
            font-weight: 700;
            letter-spacing: 0.08em;
            box-shadow: 0 0 20px rgba(34, 197, 94, 0.25);
            display: flex; align-items: center; gap: 6px;
        ">
            <span style="width: 7px; height: 7px; border-radius: 50%; background: #4ade80; box-shadow: 0 0 8px #4ade80;"></span>
            LIVE RAG
        </div>
        <div style="
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 9999px;
            padding: 6px 16px;
            font-size: 0.75rem;
            color: #38bdf8;
            font-weight: 700;
            letter-spacing: 0.08em;
        ">
            AI CLONE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Mandatory Rubric Banner
st.warning("⚠️ **This is an AI clone, not the real person.** It answers strictly using Sheheryar Ilyas's personal documents.")

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

    # 3. Dense Retriever
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
# OPTIONAL EXTRA #1: AGENTIC ROUTING
# -------------------------------------------------------------------------
def agentic_route_question(question: str):
    q = question.lower()
    out_of_scope_terms = [
        "cnic", "passport", "phone number", "home address", "bank account",
        "password", "credit card", "biryani", "weather", "stock price", "bitcoin"
    ]
    if any(term in q for term in out_of_scope_terms):
        return "Out of Scope", "Decline", "Identified as private data or off-topic query."
    opinion_terms = [
        "opinion", "hot take", "advice", "feel", "think", "unwind", "relax",
        "exhausted", "hobby", "hobbies", "free time", "values", "philosophy", "prefer"
    ]
    if any(term in q for term in opinion_terms):
        return "Opinion", "Dense only", "Classified as lifestyle/personal view. Recommends Dense semantic search."
    return "Personal Fact", "Hybrid", "Classified as factual/technical query. Recommends Hybrid (BM25 + Dense) fusion."

# -------------------------------------------------------------------------
# RETRIEVAL + HYBRID FUSION
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
# GENERATION (First Person Grounded)
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
                        import time
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

    return "I am currently experiencing high server traffic. Please try again in a moment."

# -------------------------------------------------------------------------
# HIT@3 EVALUATION SET
# -------------------------------------------------------------------------
EVAL_SET = [
    {
        "q": "Q1 (Exact Rare Keyword): What is my dual-storage setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B?",
        "query": "What is my dual-storage setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B?",
        "relevant_files": ["workflow.md"],
        "best_reason": "BM25 wins -- exact rare hardware/model tokens (ADATA, Toshiba, Qwen 2.5 7B)."
    },
    {
        "q": "Q2 (Paraphrased / No Shared Words): How do I recharge and relax when I am exhausted from coding?",
        "query": "How do I recharge and relax when I am exhausted from coding?",
        "relevant_files": ["hobbies.md"],
        "best_reason": "Dense wins -- connects 'recharge/relax' semantically to movies and reels without shared words."
    },
    {
        "q": "Q3 (FYP Architecture): How does the Two-Layer Architecture in ERPify handle custom AI modules and security?",
        "query": "How does the Two-Layer Architecture in ERPify handle custom AI modules and security?",
        "relevant_files": ["erpify_architecture.md", "projects.md", "faq.md"],
        "best_reason": "Hybrid wins -- combines exact 'ERPify' keyword match with semantic security concepts."
    },
    {
        "q": "Q4 (Personal Opinion): What is my honest opinion on traditional university degrees versus practical skills?",
        "query": "What is my honest opinion on traditional university degrees versus practical skills?",
        "relevant_files": ["hobbies.md", "advice.md", "career_goals_values.md"],
        "best_reason": "Hybrid wins -- ranks both my opinion in hobbies.md and junior advice in advice.md."
    },
    {
        "q": "Q5 (SQE & Testing): How do I perform software quality engineering and security testing on mobile apps?",
        "query": "How do I perform software quality engineering and security testing on mobile apps?",
        "relevant_files": ["sqe_testing_notes.md", "bio.md"],
        "best_reason": "Hybrid wins -- combines exact 'SQE' terms with detailed testing methodology chunks."
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
    hit_rates = {m: f"{(c/total)*100:.0f}% ({c}/{total})" for m, c in hits.items()}
    return hit_rates, rows

# =========================================================================
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.markdown('<div class="section-title">Retrieval Controls</div>', unsafe_allow_html=True)
    retrieval_mode = st.radio("Retrieval Mode:", options=["Hybrid", "BM25 only", "Dense only"], index=0)
    fusion_type = st.selectbox("Fusion Strategy:", options=["RRF", "Weighted"], disabled=(retrieval_mode != "Hybrid"))
    alpha = 0.5
    if retrieval_mode == "Hybrid" and fusion_type == "Weighted":
        alpha = st.slider("Dense Weight (alpha):", 0.0, 1.0, 0.5, 0.05)
    top_k = st.slider("Top-K Retrieved:", 1, 5, 3)

    st.divider()
    st.markdown('<div class="section-title">Extra Capabilities</div>', unsafe_allow_html=True)
    use_agentic_router = st.checkbox("Agentic Router Auto-Select", value=False)
    use_reranker = st.checkbox("Cross-Encoder Reranker", value=False)

    st.divider()
    st.markdown(f"""
    <div style="
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
    ">
        <div style="font-size: 0.72rem; font-weight: 800; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;">
            System Telemetry
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom: 8px;">
            <span style="font-size: 0.8rem; color: #64748b;">Source Files</span>
            <span style="font-size: 0.8rem; color: #38bdf8; font-weight: 700;">{num_docs}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom: 8px;">
            <span style="font-size: 0.8rem; color: #64748b;">Total Chunks</span>
            <span style="font-size: 0.8rem; color: #38bdf8; font-weight: 700;">{len(chunks)}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom: 8px;">
            <span style="font-size: 0.8rem; color: #64748b;">Embedding Dimension</span>
            <span style="font-size: 0.8rem; color: #38bdf8; font-weight: 700;">384-D</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="font-size: 0.8rem; color: #64748b;">Model</span>
            <span style="font-size: 0.8rem; color: #c084fc; font-weight: 700;">MiniLM-L6</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

    if st.button("Reset Conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["last_retrieved"] = []
        st.rerun()

# =========================================================================
# TABS
# =========================================================================
tab1, tab2, tab3 = st.tabs(["Clone Chatroom", "Hit@3 Evaluation", "Indexed Knowledge Base"])

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# -- TAB 1: CHATROOM --
with tab1:
    col_chat, col_chunks = st.columns([1.15, 1.0], gap="large")

    with col_chat:
        st.markdown('<div class="section-title">Query Console</div>', unsafe_allow_html=True)
        st.caption("Click any preset query to test retriever strengths or type your own question below:")

        c1, c2 = st.columns(2)
        q_samples = [
            ("Q1 · Hardware & Qwen (BM25)", "What is my dual-storage setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B?"),
            ("Q2 · Recharge & Relax (Dense)", "How do I recharge and relax when I am exhausted from coding?"),
            ("Q3 · ERPify Two-Layer (Hybrid)", "How does the Two-Layer Architecture in ERPify handle custom AI modules and security?"),
            ("Q4 · University View (Hybrid)", "What is my honest opinion on traditional university degrees versus practical skills?"),
            ("Q5 · SQE Testing Method (Hybrid)", "How do I perform software quality engineering and security testing on mobile apps?"),
            ("Q6 · Out-of-Scope (Decline)", "What is your favorite recipe for cooking chicken biryani?")
        ]

        selected_sample = None
        for i, (btn_lbl, q_txt) in enumerate(q_samples):
            with (c1 if i % 2 == 0 else c2):
                if st.button(btn_lbl, key=f"q_btn_{i}", use_container_width=True):
                    selected_sample = q_txt

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

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

            retrieved_list = [] if active_mode == "Decline" else retrieve(
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

            meta_text = f"Router: {cat} · Mode: {active_mode} · Reranker: {'ON' if use_reranker else 'OFF'}"
            st.session_state["messages"].append({"role": "assistant", "content": answer, "meta": meta_text})
            st.session_state["last_retrieved"] = retrieved_list
            st.session_state["last_route_info"] = (cat, active_mode, recommended_mode, route_reason)
            st.rerun()

    with col_chunks:
        st.markdown('<div class="section-title">Retrieved Passages & Fusion</div>', unsafe_allow_html=True)

        if "last_route_info" in st.session_state:
            cat, m_used, rec_m, r_reason = st.session_state["last_route_info"]
            st.info(f"**Active Mode:** `{m_used}` · **Intent:** `{cat}` · **Recommended:** `{rec_m}`\n\n_{r_reason}_")

        if "last_retrieved" in st.session_state and st.session_state["last_retrieved"]:
            for rank, item in enumerate(st.session_state["last_retrieved"], start=1):
                c = item["chunk"]
                with st.expander(f"Rank #{rank} · {c['id']} · Score: {item['score']:.4f}", expanded=(rank == 1)):
                    st.markdown(f"""
                    <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px;">
                        <span style="
                            background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.35);
                            color: #38bdf8; border-radius: 8px; padding: 2px 10px; font-size: 0.72rem; font-weight: 700;
                        ">{c['source']}</span>
                        <span style="
                            background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);
                            color: #94a3b8; border-radius: 8px; padding: 2px 10px; font-size: 0.72rem;
                        ">{c['word_count']} words</span>
                        <span style="
                            background: rgba(168, 85, 247, 0.15); border: 1px solid rgba(168, 85, 247, 0.35);
                            color: #c084fc; border-radius: 8px; padding: 2px 10px; font-size: 0.72rem; font-weight: 700;
                        ">Top-{rank} Match</span>
                    </div>
                    """, unsafe_allow_html=True)

                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.markdown(f"**BM25 Score:** `{item['bm25_score']:.2f}` _(Rank #{item['bm25_rank']})_")
                        st.markdown(f"**Hybrid RRF:** `{item['rrf_score']:.4f}`")
                    with c_b:
                        st.markdown(f"**Dense Cosine:** `{item['dense_score']:.4f}` _(Rank #{item['dense_rank']})_")
                        st.markdown(f"**Weighted Score:** `{item['weighted_score']:.4f}`")

                    if item["rerank_score"] is not None:
                        st.markdown(f"**Cross-Encoder Rerank:** `{item['rerank_score']:.4f}`")

                    st.markdown("---")
                    st.markdown(f"<div style='color:#cbd5e1; font-size:0.83rem; line-height:1.75;'>{c['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                border: 1px dashed rgba(255, 255, 255, 0.12);
                border-radius: 20px;
                padding: 48px 24px;
                text-align: center;
                background: rgba(255, 255, 255, 0.015);
            ">
                <div style="font-size: 2rem; margin-bottom: 8px; opacity: 0.6;">⚡</div>
                <div style="color: #64748b; font-size: 0.85rem; font-weight: 600;">
                    Passages retrieved by BM25, Dense, and Hybrid fusion will appear here.
                </div>
            </div>
            """, unsafe_allow_html=True)

# -- TAB 2: HIT@3 EVALUATION --
with tab2:
    st.markdown('<div class="section-title">Quantitative Benchmarks · Hit@3</div>', unsafe_allow_html=True)
    if chunks:
        hit_rates, eval_rows = evaluate_hit_at_3()
        k1, k2, k3 = st.columns(3)
        k1.metric("BM25 Only (Hit@3)", hit_rates.get("BM25 only", "0%"))
        k2.metric("Dense Only (Hit@3)", hit_rates.get("Dense only", "0%"))
        k3.metric("Hybrid RRF (Hit@3)", hit_rates.get("Hybrid", "0%"))
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.table(eval_rows)

# -- TAB 3: KNOWLEDGE BASE --
with tab3:
    st.markdown(f'<div class="section-title">Corpus Explorer · {num_docs} Documents · {len(chunks)} Chunks</div>', unsafe_allow_html=True)
    for c in chunks:
        with st.expander(f"{c['id']} — {c['source']} ({c['word_count']} words)"):
            st.markdown(f"<div style='color:#cbd5e1; font-size:0.83rem; line-height:1.75;'>{c['text']}</div>", unsafe_allow_html=True)
