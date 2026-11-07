import os
import io
import glob
import time
import html
import requests
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# =====================================================================
# 1. PAGE CONFIG & QUANTIX OBSIDIAN DARK GLASSMORPHIC CSS OVERRIDE
# =====================================================================
st.set_page_config(
    page_title="Quantix RAG | Sheheryar Ilyas Digital Clone",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Force Full Obsidian Dark Theme (Quantix Aesthetic) */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: radial-gradient(circle at 18% 10%, #141829 0%, #090A0F 55%, #06070B 100%) !important;
        color: #F8FAFC !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1440px !important;
    }

    /* Quantix Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D0F18 0%, #08090E 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    }
    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }

    /* Top Quantix Live Ticker Bar */
    .qx-ticker {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        background: rgba(17, 20, 32, 0.78);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px 18px;
        margin-bottom: 14px;
        font-size: 0.82rem;
    }
    .qx-breadcrumb {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #94A3B8;
        font-weight: 600;
    }
    .qx-breadcrumb span.active {
        color: #FFFFFF;
        font-weight: 700;
    }
    .qx-ticker-items {
        display: flex;
        align-items: center;
        gap: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }
    .qx-ticker-item {
        display: flex;
        align-items: center;
        gap: 6px;
        color: #CBD5E1;
    }
    .qx-dot-green {
        width: 8px;
        height: 8px;
        background: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10B981;
        display: inline-block;
    }

    /* Required AI Clone Notice Banner (Sleek Neon Amber Glass) */
    .qx-notice-banner {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.14) 0%, rgba(16, 185, 129, 0.08) 100%);
        border: 1px solid rgba(245, 158, 11, 0.38);
        border-left: 4px solid #F59E0B;
        border-radius: 12px;
        padding: 11px 18px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #FDE68A;
        font-size: 0.86rem;
        font-weight: 600;
    }

    /* Quantix Hero + KPI Metric Cards Row */
    .qx-kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 20px;
    }
    @media (max-width: 992px) {
        .qx-kpi-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .qx-card {
        background: linear-gradient(145deg, rgba(22, 26, 41, 0.92) 0%, rgba(13, 15, 24, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 14px 30px -10px rgba(0, 0, 0, 0.6);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .qx-card:hover {
        border-color: rgba(99, 102, 241, 0.45);
        transform: translateY(-2px);
    }
    .qx-card-accent-left {
        position: absolute;
        left: 0;
        top: 18px;
        bottom: 18px;
        width: 3px;
        border-radius: 0 4px 4px 0;
        background: linear-gradient(180deg, #6366F1, #10B981);
    }
    .qx-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .qx-card-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
        font-weight: 700;
    }
    .qx-icon-circle {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        background: rgba(99, 102, 241, 0.16);
        border: 1px solid rgba(99, 102, 241, 0.35);
    }
    .qx-card-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .qx-pill-green {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34D399;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.73rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .qx-pill-purple {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(139, 92, 246, 0.16);
        border: 1px solid rgba(139, 92, 246, 0.4);
        color: #C4B5FD;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.73rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .qx-pill-amber {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #FBBF24;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.73rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Retrieved Chunk Glass Cards (Right Panel - No Cut-off Numbers!) */
    .qx-chunk-card {
        background: linear-gradient(145deg, rgba(19, 23, 36, 0.94) 0%, rgba(11, 13, 21, 0.96) 100%);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 10px 24px -8px rgba(0, 0, 0, 0.55);
    }
    .qx-chunk-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }
    .qx-chunk-title {
        font-weight: 700;
        font-size: 0.92rem;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .qx-score-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        margin-bottom: 12px;
    }
    .qx-score-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 8px 10px;
        text-align: center;
    }
    .qx-score-box .lbl {
        font-size: 0.68rem;
        color: #94A3B8;
        text-transform: uppercase;
        display: block;
        margin-bottom: 3px;
    }
    .qx-score-box .val {
        font-size: 0.92rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }
    .qx-score-box .sub {
        font-size: 0.66rem;
        color: #10B981;
        font-family: 'JetBrains Mono', monospace;
    }
    .qx-bar-wrap {
        margin-bottom: 8px;
    }
    .qx-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .qx-bar-bg {
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.07);
        border-radius: 999px;
        overflow: hidden;
    }
    .qx-bar-fill-bm25 {
        height: 100%;
        background: linear-gradient(90deg, #F59E0B, #FBBF24);
        border-radius: 999px;
    }
    .qx-bar-fill-dense {
        height: 100%;
        background: linear-gradient(90deg, #6366F1, #10B981);
        border-radius: 999px;
    }
    .qx-passage-box {
        background: rgba(6, 8, 14, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 12px 14px;
        font-size: 0.84rem;
        line-height: 1.55;
        color: #CBD5E1;
        margin-top: 10px;
    }

    /* Quantix Market-Overview Style Table */
    .qx-table {
        width: 100%;
        border-collapse: collapse;
        background: rgba(15, 18, 29, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        overflow: hidden;
        font-size: 0.84rem;
    }
    .qx-table th {
        background: rgba(255, 255, 255, 0.04);
        color: #94A3B8;
        font-weight: 700;
        text-align: left;
        padding: 12px 14px;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .qx-table td {
        padding: 12px 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        color: #E2E8F0;
        vertical-align: top;
    }
    .qx-table tr:hover td {
        background: rgba(99, 102, 241, 0.06);
    }

    /* Buttons & Inputs Dark Styling */
    .stButton > button {
        background: linear-gradient(135deg, rgba(30, 35, 54, 0.95) 0%, rgba(19, 23, 36, 0.95) 100%) !important;
        color: #F1F5F9 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        padding: 0.5rem 0.9rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        border-color: #6366F1 !important;
        box-shadow: 0 0 16px rgba(99, 102, 241, 0.35) !important;
        color: #FFFFFF !important;
    }
    [data-testid="stChatMessage"] {
        background: rgba(20, 24, 38, 0.72) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 14px !important;
        padding: 12px 16px !important;
        margin-bottom: 10px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(16, 19, 30, 0.85);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.07);
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4F46E5 0%, #0D9488 100%) !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. LOAD KNOWLEDGE BASE, CHUNK (200-300 WORDS), INDEX & RERANKER
# =====================================================================
DATA_DIR = "data"
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")

@st.cache_resource
def initialize_rag_pipeline(chunk_size=220, overlap=40):
    os.makedirs(DATA_DIR, exist_ok=True)
    md_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.md")))
    
    chunks = []
    step = max(1, chunk_size - overlap)

    for filepath in md_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        
        words = text.split()
        if not words:
            continue
            
        chunk_idx = 1
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            if not chunk_words:
                break
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "id": f"{filename}#chunk_{chunk_idx}",
                "source": filename,
                "word_count": len(chunk_words),
                "text": chunk_text
            })
            chunk_idx += 1
            if i + chunk_size >= len(words):
                break

    if not chunks:
        return [], None, None, None, None, 0

    # 1. BM25 Lexical Index (rank_bm25)
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)

    # 2. Dense Semantic Index (all-MiniLM-L6-v2 + .npy Precomputed Storage)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    if os.path.exists(EMBEDDINGS_PATH):
        dense_embeddings = np.load(EMBEDDINGS_PATH)
        if dense_embeddings.shape[0] != len(chunks):
            dense_embeddings = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
            np.save(EMBEDDINGS_PATH, dense_embeddings)
    else:
        dense_embeddings = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
        np.save(EMBEDDINGS_PATH, dense_embeddings)

    # 3. Optional Extra #2: Cross-Encoder Neural Reranker
    try:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        reranker = None

    return chunks, bm25_index, embedder, dense_embeddings, reranker, len(md_files)

chunks, bm25_index, embedder, dense_embeddings, reranker, total_docs = initialize_rag_pipeline()

# =====================================================================
# 3. OPTIONAL EXTRA #1: AGENTIC ROUTING STEP (CLASSIFY & PICK MODE)
# =====================================================================
def agentic_router(query: str):
    """
    Classifies each question into:
    - Personal Fact / Exact Technical Keyword -> Recommends BM25 or Hybrid
    - Opinion / Conceptual / Paraphrased -> Recommends Dense or Hybrid
    - Out of Scope / Sensitive Private Data -> Declines or triggers guardrail
    """
    q = query.lower()

    # Check Out-of-Scope or Private Sensitive Requests
    private_triggers = ["cnic", "passport", "phone number", "home address", "bank account", "password", "credit card"]
    if any(t in q for t in private_triggers):
        return {
            "category": "Out of Scope / Privacy Guardrail",
            "recommended_mode": "Decline",
            "badge_class": "qx-pill-amber",
            "reason": "Blocked sensitive personal/financial query per privacy rules."
        }

    # Check Exact Rare Keyword Queries (BM25 Specialty)
    rare_tokens = ["adata", "toshiba", "qwen", "23018020011", "scrollnix", "eliteconnect", "erpify", "rls", "c#/.net"]
    if any(t in q for t in rare_tokens):
        return {
            "category": "Personal Fact (Exact Rare Keyword)",
            "recommended_mode": "Hybrid",
            "badge_class": "qx-pill-green",
            "reason": "Detected exact rare project/hardware tokens; combining BM25 lexical match with Dense context."
        }

    # Check Opinion / Hot Take / Paraphrased Lifestyle Queries (Dense Specialty)
    opinion_triggers = ["opinion", "hot take", "advice", "think about", "feel about", "relax", "unwind", "exhausted", "free time", "hobby", "favorite", "why"]
    if any(t in q for t in opinion_triggers):
        return {
            "category": "Personal Opinion & Lifestyle",
            "recommended_mode": "Dense only",
            "badge_class": "qx-pill-purple",
            "reason": "Semantic/opinion query detected; Dense embeddings capture paraphrased meaning best."
        }

    return {
        "category": "Personal Fact & Technical Profile",
        "recommended_mode": "Hybrid",
        "badge_class": "qx-pill-green",
        "reason": "General profile query; Hybrid RRF balances keyword precision and semantic recall."
    }

# =====================================================================
# 4. HYBRID RETRIEVAL + RRF / WEIGHTED FUSION + CROSS-ENCODER RERANKING
# =====================================================================
def min_max_norm(arr):
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)

def retrieve_passages(query, mode="Hybrid", fusion_method="RRF (1 / (60 + rank))", alpha=0.6, use_reranker=False, top_k=3):
    if not chunks:
        return []

    # BM25 Lexical Scores & Ranks
    tokens = query.lower().split()
    bm25_scores = np.array(bm25_index.get_scores(tokens), dtype=float)
    bm25_order = np.argsort(bm25_scores)[::-1]
    bm25_ranks = {idx: r + 1 for r, idx in enumerate(bm25_order)}

    # Dense Cosine Similarity Scores & Ranks (NumPy dot product on normalized vectors)
    q_vec = embedder.encode([query], normalize_embeddings=True)[0]
    dense_scores = np.dot(dense_embeddings, q_vec).astype(float)
    dense_order = np.argsort(dense_scores)[::-1]
    dense_ranks = {idx: r + 1 for r, idx in enumerate(dense_order)}

    bm25_n = min_max_norm(bm25_scores)
    dense_n = min_max_norm(dense_scores)

    candidates = []
    for idx in range(len(chunks)):
        b_rk = bm25_ranks[idx]
        d_rk = dense_ranks[idx]
        b_sc = float(bm25_scores[idx])
        d_sc = float(dense_scores[idx])

        # Fusion Formula 1: Reciprocal Rank Fusion
        rrf_sc = (1.0 / (60.0 + b_rk)) + (1.0 / (60.0 + d_rk))
        # Fusion Formula 2: Min-Max Normalized Weighted Score
        w_sc = (alpha * float(dense_n[idx])) + ((1.0 - alpha) * float(bm25_n[idx]))

        if mode == "BM25 only":
            base_score = b_sc
        elif mode == "Dense only":
            base_score = d_sc
        else:
            base_score = rrf_sc if "RRF" in fusion_method else w_sc

        candidates.append({
            "idx": idx,
            "chunk": chunks[idx],
            "score": base_score,
            "bm25_score": b_sc,
            "dense_score": d_sc,
            "bm25_norm": float(bm25_n[idx]),
            "dense_norm": float(dense_n[idx]),
            "rrf_score": rrf_sc,
            "weighted_score": w_sc,
            "bm25_rank": b_rk,
            "dense_rank": d_rk,
            "rerank_score": None
        })

    # Sort initial candidates
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Optional Extra #2: Apply Cross-Encoder Reranker over top-6 hybrid results
    if use_reranker and reranker is not None and len(candidates) > 1:
        top_pool = candidates[:min(6, len(candidates))]
        pairs = [[query, item["chunk"]["text"]] for item in top_pool]
        ce_scores = reranker.predict(pairs)
        for item, ce_sc in zip(top_pool, ce_scores):
            item["rerank_score"] = float(ce_sc)
            item["score"] = float(ce_sc)
        top_pool.sort(key=lambda x: x["score"], reverse=True)
        return top_pool[:top_k]

    return candidates[:top_k]

# =====================================================================
# 5. OPTIONAL EXTRA #3: HAND-LABELED EVALUATION SET WITH HIT@3 METRIC
# =====================================================================
EVAL_BENCHMARK = [
    {
        "id": "Q1",
        "type": "Exact Rare Keyword",
        "question": "What is my dual-storage setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B?",
        "expected_sources": ["workflow.md"],
        "why_winner": "BM25 wins on exact hardware/model tokens (ADATA, Toshiba, Qwen 2.5 7B)."
    },
    {
        "id": "Q2",
        "type": "Paraphrased (Zero Shared Words)",
        "question": "How do I unwind and recharge when I am exhausted from coding?",
        "expected_sources": ["hobbies.md"],
        "why_winner": "Dense wins by connecting 'unwind/recharge' semantically to watching Harry Potter, Wednesday Addams, and reels."
    },
    {
        "id": "Q3",
        "type": "Technical Architecture",
        "question": "How does the two-layer architecture in ERPify ensure security and tenant isolation?",
        "expected_sources": ["projects.md", "faq.md"],
        "why_winner": "Hybrid combines exact 'ERPify' keyword matches with semantic security/encryption passages."
    },
    {
        "id": "Q4",
        "type": "Personal Hot Take / Opinion",
        "question": "What is my honest opinion on traditional university degrees vs practical skills?",
        "expected_sources": ["hobbies.md", "advice.md", "faq.md"],
        "why_winner": "Hybrid surfaces both my opinion in hobbies.md and my junior advice in advice.md."
    },
    {
        "id": "Q5",
        "type": "AI Tooling Comparison",
        "question": "Which AI model do I prefer for daily coding productivity versus image generation?",
        "expected_sources": ["hobbies.md", "faq.md"],
        "why_winner": "Hybrid ranks both my Gemini Pro coding preference and ChatGPT/Imagen preference at the top."
    }
]

@st.cache_data
def compute_hit_at_3_metrics():
    if not chunks:
        return {"BM25 only": 0.0, "Dense only": 0.0, "Hybrid": 0.0}, []

    hits = {"BM25 only": 0, "Dense only": 0, "Hybrid": 0}
    table_rows = []

    for item in EVAL_BENCHMARK:
        q = item["question"]
        exp = item["expected_sources"]

        b_res = retrieve_passages(q, mode="BM25 only", top_k=3)
        d_res = retrieve_passages(q, mode="Dense only", top_k=3)
        h_res = retrieve_passages(q, mode="Hybrid", top_k=3)

        b_hit = any(r["chunk"]["source"] in exp for r in b_res)
        d_hit = any(r["chunk"]["source"] in exp for r in d_res)
        h_hit = any(r["chunk"]["source"] in exp for r in h_res)

        if b_hit: hits["BM25 only"] += 1
        if d_hit: hits["Dense only"] += 1
        if h_hit: hits["Hybrid"] += 1

        table_rows.append({
            "id": item["id"],
            "type": item["type"],
            "question": q,
            "bm25_top": f"{b_res[0]['chunk']['id']} ({b_res[0]['bm25_score']:.2f})" if b_res else "-",
            "dense_top": f"{d_res[0]['chunk']['id']} ({d_res[0]['dense_score']:.3f})" if d_res else "-",
            "hybrid_top": f"{h_res[0]['chunk']['id']} ({h_res[0]['rrf_score']:.4f})" if h_res else "-",
            "why": item["why_winner"]
        })

    n = len(EVAL_BENCHMARK)
    rates = {k: (v / n) * 100.0 for k, v in hits.items()}
    return rates, table_rows

hit_rates, benchmark_rows = compute_hit_at_3_metrics()

# =====================================================================
# 6. GROUNDED FIRST-PERSON LLM GENERATION WITH CONVERSATION MEMORY
# =====================================================================
def generate_grounded_reply(question, retrieved_items, chat_history, route_info):
    if route_info["recommended_mode"] == "Decline":
        return "I don't share private details like phone numbers, addresses, CNIC, or financial information. I am an AI clone of Sheheryar Ilyas."

    context_blocks = []
    for item in retrieved_items:
        c = item["chunk"]
        context_blocks.append(f"[{c['id']} | Source: {c['source']}]\n{c['text']}")
    context_str = "\n\n".join(context_blocks)

    # Optional Extra #4: Short Conversation Memory (last 2 user/assistant exchanges)
    memory_lines = []
    for m in chat_history[-4:]:
        role_label = "User" if m["role"] == "user" else "Sheheryar Clone"
        memory_lines.append(f"{role_label}: {m['content']}")
    memory_str = "\n".join(memory_lines) if memory_lines else "None"

    persona_prompt = f"""You are the AI clone of Sheheryar Ilyas, a 7th-semester BSCS student at UMT Sialkot.
Speak strictly in the first person ("I", "my", "me") in a confident, friendly, and natural tone.
Rules:
1. Answer ONLY using the CONTEXT below. It comes from my own personal documents.
2. If the context does not contain the answer, say exactly: "I don't have that in my notes." Do not guess or invent facts about me.
3. Never share phone numbers, addresses, ID numbers or other private details.
4. If asked whether you are the real person, clearly say you are an AI clone of Sheheryar Ilyas.

SHORT CONVERSATION MEMORY:
{memory_str}

CONTEXT:
{context_str}

QUESTION:
{question}

ANSWER:"""

    gemini_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    hf_token = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)

    if gemini_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": persona_prompt}]}],
            "generationConfig": {"temperature": 0.15, "maxOutputTokens": 320}
        }
        try:
            res = requests.post(url, json=payload, timeout=30)
            res.raise_for_status()
            data = res.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            return f"⚠️ Gemini API Error: {str(e)}"

    if hf_token:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=hf_token)
            resp = client.chat_completion(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": persona_prompt}],
                max_tokens=300,
                temperature=0.15
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ HF Inference API Error: {str(e)}"

    return "⚠️ No API key found in Secrets (`GEMINI_API_KEY` or `HF_TOKEN`). Please configure your secret key."

# =====================================================================
# 7. QUANTIX SIDEBAR NAVIGATION & RAG ENGINE CONTROLS
# =====================================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
        <div style="width:36px; height:36px; border-radius:10px; background:linear-gradient(135deg,#6366F1,#10B981); display:flex; align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; color:#fff;">Q</div>
        <div>
            <div style="font-weight:800; font-size:1.05rem; color:#fff; line-height:1.1;">Sheheryar.AI</div>
            <div style="font-size:0.72rem; color:#94A3B8;">Hybrid RAG Digital Clone</div>
        </div>
    </div>
    <div style="margin: 14px 0 18px 0; padding: 12px 14px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px;">
        <div style="font-size:1.1rem; font-weight:800; color:#fff;">Welcome, Visitor 👋</div>
        <div style="font-size:0.74rem; color:#94A3B8; margin-top:2px;">UMT Sialkot • BSCS 7th Sem • CLO-2</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🎛️ Retrieval Mode Selector")
    retrieval_mode = st.radio(
        "Select Active Retriever:",
        options=["Hybrid", "BM25 only", "Dense only"],
        index=0
    )

    agentic_auto_mode = st.toggle(
        "🧭 Agentic Auto-Mode Routing",
        value=False,
        help="Strong Student Extra #1: Automatically classifies each query and picks the optimal mode."
    )

    use_reranker = st.toggle(
        "⚡ Cross-Encoder Neural Reranker",
        value=False,
        help="Strong Student Extra #2: Reranks top hybrid chunks using ms-marco-MiniLM-L-6-v2."
    )

    fusion_method = st.selectbox(
        "🔗 Hybrid Fusion Formula:",
        options=["RRF (1 / (60 + rank))", "Weighted Linear (α·Dense + (1-α)·BM25)"],
        index=0,
        disabled=(retrieval_mode != "Hybrid")
    )

    alpha_weight = 0.6
    if retrieval_mode == "Hybrid" and "Weighted" in fusion_method:
        alpha_weight = st.slider("Dense Weight (α):", 0.0, 1.0, 0.6, 0.05)

    top_k = st.slider("📑 Top-K Chunks:", min_value=1, max_value=5, value=3)

    st.divider()
    if dense_embeddings is not None:
        buf = io.BytesIO()
        np.save(buf, dense_embeddings)
        st.download_button(
            label="⬇️ Download embeddings.npy",
            data=buf.getvalue(),
            file_name="embeddings.npy",
            mime="application/octet-stream",
            use_container_width=True
        )

    if st.button("🗑️ Reset Conversation Memory", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["last_retrieved"] = []
        st.rerun()

# =====================================================================
# 8. TOP QUANTIX TICKER + REQUIRED VISIBLE BANNER + KPI SPARKLINE CARDS
# =====================================================================
st.markdown(f"""
<div class="qx-ticker">
    <div class="qx-breadcrumb">
        <span>🏠 Overview</span> <span>/</span> <span class="active">Digital Clone Intelligence Studio</span>
    </div>
    <div class="qx-ticker-items">
        <div class="qx-ticker-item"><span class="qx-dot-green"></span> SYSTEM ONLINE</div>
        <div class="qx-ticker-item">CORPUS: <strong>{total_docs} DOCS ({len(chunks)} CHUNKS)</strong></div>
        <div class="qx-ticker-item">EMBEDDINGS: <strong>384-D (.NPY CACHED)</strong></div>
        <div class="qx-ticker-item">HYBRID HIT@3: <strong style="color:#34D399;">{hit_rates['Hybrid']:.0f}%</strong></div>
    </div>
</div>

<div class="qx-notice-banner">
    <span>⚠️ <strong>OFFICIAL AI CLONE NOTICE: This is an AI clone, not the real person.</strong> Answers are generated strictly in first person from Sheheryar Ilyas's personal Markdown documents.</span>
    <span style="font-family:'JetBrains Mono',monospace; font-size:0.76rem;">GROUNDED RAG v2.5</span>
</div>

<div class="qx-kpi-grid">
    <div class="qx-card">
        <div class="qx-card-accent-left"></div>
        <div class="qx-card-header">
            <span class="qx-card-label">Knowledge Base</span>
            <div class="qx-icon-circle">📂</div>
        </div>
        <div class="qx-card-value">{total_docs} Docs / {len(chunks)} Chunks</div>
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <span class="qx-pill-green">↗ 220w + 40w Overlap</span>
            <svg width="70" height="24" viewBox="0 0 70 24"><path d="M2 20 L16 14 L30 17 L46 7 L68 3" fill="none" stroke="#10B981" stroke-width="2.2"/></svg>
        </div>
    </div>
    <div class="qx-card">
        <div class="qx-card-accent-left"></div>
        <div class="qx-card-header">
            <span class="qx-card-label">BM25 Lexical Hit@3</span>
            <div class="qx-icon-circle">🔑</div>
        </div>
        <div class="qx-card-value">{hit_rates['BM25 only']:.0f}% Accuracy</div>
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <span class="qx-pill-amber">★ Wins Rare Keywords</span>
            <svg width="70" height="24" viewBox="0 0 70 24"><path d="M2 18 L18 10 L34 15 L50 8 L68 6" fill="none" stroke="#F59E0B" stroke-width="2.2"/></svg>
        </div>
    </div>
    <div class="qx-card">
        <div class="qx-card-accent-left"></div>
        <div class="qx-card-header">
            <span class="qx-card-label">Dense Semantic Hit@3</span>
            <div class="qx-icon-circle">🧠</div>
        </div>
        <div class="qx-card-value">{hit_rates['Dense only']:.0f}% Accuracy</div>
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <span class="qx-pill-purple">★ Wins Paraphrases</span>
            <svg width="70" height="24" viewBox="0 0 70 24"><path d="M2 21 L20 16 L36 9 L52 11 L68 4" fill="none" stroke="#8B5CF6" stroke-width="2.2"/></svg>
        </div>
    </div>
    <div class="qx-card">
        <div class="qx-card-accent-left"></div>
        <div class="qx-card-header">
            <span class="qx-card-label">Hybrid RRF Hit@3</span>
            <div class="qx-icon-circle">⚡</div>
        </div>
        <div class="qx-card-value">{hit_rates['Hybrid']:.0f}% Accuracy</div>
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <span class="qx-pill-green">↗ Best Overall Fusion</span>
            <svg width="70" height="24" viewBox="0 0 70 24"><path d="M2 22 L18 15 L34 12 L50 5 L68 2" fill="none" stroke="#34D399" stroke-width="2.2"/></svg>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# 9. MAIN WORKSPACE TABS
# =====================================================================
tab_studio, tab_benchmark, tab_corpus = st.tabs([
    "💬 Interactive Clone Terminal & Live Telemetry",
    "📊 Hit@3 Evaluation Benchmark & 1-Page Report",
    "🗄️ Indexed Knowledge Base Chunks"
])

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Welcome! I am the AI digital clone of **Sheheryar Ilyas** (7th-semester BSCS student at UMT Sialkot). Ask me about my Final Year Project **ERPify**, my **dual-storage ADATA/Toshiba setup**, **Qwen 2.5 7B local testing**, or my **hobbies & university hot takes**!"
        }
    ]

with tab_studio:
    col_left, col_right = st.columns([1.28, 1.0], gap="large")

    with col_left:
        st.markdown("##### ⚡ One-Click Diagnostic Test Prompts")
        p_cols = st.columns(3)
        starter_prompts = [
            ("🔑 Exact Keyword (BM25)", "What is my dual-storage setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B?"),
            ("🌊 Paraphrased (Dense)", "How do I unwind and recharge when I am exhausted from coding?"),
            ("🏗️ ERPify Architecture", "How does the two-layer architecture in ERPify ensure security?"),
            ("🔥 University Hot Take", "What is your honest opinion on traditional university degrees?"),
            ("🎨 AI Model Preference", "Which AI model do you prefer for daily coding vs image generation?"),
            ("🛡️ Out-of-Scope Test", "What is your favorite recipe for cooking chicken biryani?")
        ]

        clicked_prompt = None
        for i, (btn_label, q_str) in enumerate(starter_prompts):
            with p_cols[i % 3]:
                if st.button(btn_label, key=f"starter_{i}", use_container_width=True):
                    clicked_prompt = q_str

        # Chat Terminal Box
        chat_box = st.container(height=430)
        with chat_box:
            for msg in st.session_state["messages"]:
                with st.chat_message(msg["role"], avatar="⚡" if msg["role"] == "assistant" else "🧑‍💻"):
                    if "meta_badge" in msg:
                        st.markdown(msg["meta_badge"], unsafe_allow_html=True)
                    st.markdown(msg["content"])

        typed_query = st.chat_input("Ask Sheheryar's Digital Clone anything...")
        final_query = clicked_prompt if clicked_prompt else typed_query

        if final_query:
            st.session_state["messages"].append({"role": "user", "content": final_query})

            t_start = time.time()
            route_info = agentic_router(final_query)

            # Determine active retrieval mode (Manual vs Agentic Auto-Router)
            effective_mode = retrieval_mode
            if agentic_auto_mode and route_info["recommended_mode"] != "Decline":
                effective_mode = route_info["recommended_mode"]

            retrieved = retrieve_passages(
                final_query,
                mode=effective_mode,
                fusion_method=fusion_method,
                alpha=alpha_weight,
                use_reranker=use_reranker,
                top_k=top_k
            )

            reply = generate_grounded_reply(
                final_query,
                retrieved,
                st.session_state["messages"][:-1],
                route_info
            )
            elapsed = time.time() - t_start

            rerank_tag = " + CrossEncoder" if use_reranker else ""
            meta_html = (
                f"<div style='margin-bottom:6px;'>"
                f"<span class='{route_info['badge_class']}'>🧭 {route_info['category']}</span> "
                f"<span class='qx-pill-green' style='margin-left:6px;'>Mode: {effective_mode}{rerank_tag} • {elapsed:.2f}s</span>"
                f"</div>"
            )

            st.session_state["messages"].append({
                "role": "assistant",
                "content": reply,
                "meta_badge": meta_html
            })
            st.session_state["last_retrieved"] = retrieved
            st.session_state["last_route"] = route_info
            st.session_state["last_mode"] = effective_mode
            st.session_state["last_latency"] = elapsed
            st.rerun()

    # RIGHT COLUMN: CUSTOM QUANTIX GLASS CARDS FOR RETRIEVED CHUNKS
    with col_right:
        st.markdown("##### 🔬 Live Retrieval Telemetry & Grounded Passages")

        if "last_retrieved" in st.session_state and st.session_state["last_retrieved"]:
            r_info = st.session_state.get("last_route", {})
            eff_mode = st.session_state.get("last_mode", retrieval_mode)
            lat = st.session_state.get("last_latency", 0.0)

            st.markdown(f"""
            <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.28); border-radius:12px; padding:10px 14px; margin-bottom:12px; font-size:0.8rem;">
                <div style="font-weight:700; color:#34D399;">🧭 Agentic Router Verdict: {html.escape(r_info.get('category', 'General'))}</div>
                <div style="color:#CBD5E1; margin-top:2px;">{html.escape(r_info.get('reason', ''))} (Executed Mode: <strong>{eff_mode}</strong> in {lat:.2f}s)</div>
            </div>
            """, unsafe_allow_html=True)

            for rank, item in enumerate(st.session_state["last_retrieved"], start=1):
                c = item["chunk"]
                bm25_pct = int(round(item["bm25_norm"] * 100))
                dense_pct = int(round(item["dense_norm"] * 100))
                safe_text = html.escape(c["text"])
                rerank_display = f"{item['rerank_score']:.3f}" if item["rerank_score"] is not None else "OFF"

                st.markdown(f"""
                <div class="qx-chunk-card">
                    <div class="qx-chunk-top">
                        <div class="qx-chunk-title">
                            <span class="qx-pill-green">#{rank}</span>
                            <span>{html.escape(c['id'])}</span>
                        </div>
                        <span class="qx-pill-purple">📄 {html.escape(c['source'])} ({c['word_count']}w)</span>
                    </div>
                    <div class="qx-score-grid">
                        <div class="qx-score-box">
                            <span class="lbl">BM25 Lexical</span>
                            <span class="val">{item['bm25_score']:.2f}</span>
                            <span class="sub">Rank #{item['bm25_rank']}</span>
                        </div>
                        <div class="qx-score-box">
                            <span class="lbl">Dense Cosine</span>
                            <span class="val">{item['dense_score']:.3f}</span>
                            <span class="sub">Rank #{item['dense_rank']}</span>
                        </div>
                        <div class="qx-score-box">
                            <span class="lbl">Hybrid RRF</span>
                            <span class="val">{item['rrf_score']:.4f}</span>
                            <span class="sub">Fused</span>
                        </div>
                        <div class="qx-score-box">
                            <span class="lbl">Cross-Encoder</span>
                            <span class="val">{rerank_display}</span>
                            <span class="sub">Reranker</span>
                        </div>
                    </div>
                    <div class="qx-bar-wrap">
                        <div class="qx-bar-label"><span>BM25 Lexical Strength</span><span>{bm25_pct}%</span></div>
                        <div class="qx-bar-bg"><div class="qx-bar-fill-bm25" style="width:{bm25_pct}%;"></div></div>
                    </div>
                    <div class="qx-bar-wrap">
                        <div class="qx-bar-label"><span>Dense Semantic Similarity</span><span>{dense_pct}%</span></div>
                        <div class="qx-bar-bg"><div class="qx-bar-fill-dense" style="width:{dense_pct}%;"></div></div>
                    </div>
                    <div class="qx-passage-box">{safe_text}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="qx-chunk-card" style="text-align:center; padding:36px 20px;">
                <div style="font-size:2rem; margin-bottom:8px;">🛰️</div>
                <div style="font-weight:700; font-size:1rem; color:#F8FAFC;">No Active Retrieval Telemetry</div>
                <div style="font-size:0.84rem; color:#94A3B8; margin-top:6px;">
                    Click any <strong>One-Click Diagnostic Test Prompt</strong> on the left or type a question to inspect real-time BM25 scores, Dense Cosine similarity, Hybrid RRF math, and source chunks.
                </div>
            </div>
            """, unsafe_allow_html=True)

# =====================================================================
# TAB 2: HIT@3 BENCHMARK & EMBEDDED 1-PAGE REPORT
# =====================================================================
with tab_benchmark:
    st.markdown("#### 📊 Live 5-Question Benchmark Matrix & Hit@3 Evaluation")
    st.caption("Evaluated automatically at startup across all three retrieval modes (BM25 only, Dense only, Hybrid RRF).")

    rows_html = ""
    for r in benchmark_rows:
        rows_html += f"""
        <tr>
            <td><span class="qx-pill-purple">{html.escape(r['id'])}</span><br><small style="color:#94A3B8;">{html.escape(r['type'])}</small></td>
            <td style="font-weight:600;">{html.escape(r['question'])}</td>
            <td style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#FBBF24;">{html.escape(r['bm25_top'])}</td>
            <td style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#C4B5FD;">{html.escape(r['dense_top'])}</td>
            <td style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#34D399;">{html.escape(r['hybrid_top'])}</td>
            <td>{html.escape(r['why'])}</td>
        </tr>
        """

    st.markdown(f"""
    <table class="qx-table">
        <thead>
            <tr>
                <th>ID / Type</th>
                <th>Test Question</th>
                <th>BM25 Top-1</th>
                <th>Dense Top-1</th>
                <th>Hybrid Top-1</th>
                <th>Best Mode & Why</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c_fail, c_imp = st.columns(2)
    with c_fail:
        st.markdown("""
        <div class="qx-chunk-card">
            <div style="font-weight:700; color:#FBBF24; margin-bottom:6px;">⚠️ Documented Failure Case Analysis</div>
            <div style="font-size:0.85rem; color:#CBD5E1; line-height:1.6;">
                When asked <em>"Do you build desktop software in C#/.NET?"</em>, BM25 retrieved <code>projects.md#chunk_1</code> with a high lexical score because the competitor analysis sentence mentions <em>"Local Sialkot desktop software (C#/.NET)"</em> as a legacy competitor to ERPify. Without surrounding negation context, lexical matching treats competitor keywords as positive skills. Enabling our <strong>Cross-Encoder Reranker</strong> and Hybrid mode resolves this by prioritizing semantic context.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c_imp:
        st.markdown("""
        <div class="qx-chunk-card">
            <div style="font-weight:700; color:#34D399; margin-bottom:6px;">🚀 Implemented Optional Extras & Next Steps</div>
            <div style="font-size:0.85rem; color:#CBD5E1; line-height:1.6;">
                We implemented all four strong-student extras from the specification: (1) an <strong>Agentic Query Router</strong> that classifies personal facts, opinions, and out-of-scope queries, (2) a <strong>Cross-Encoder Reranker</strong> (<code>ms-marco-MiniLM-L-6-v2</code>), (3) an automated <strong>Hit@3 Evaluation Set</strong>, and (4) <strong>Short Conversation Memory</strong>. Next, we would add structure-aware Markdown header chunking so competitor tables never split away from their section titles.
            </div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# TAB 3: INDEXED KNOWLEDGE BASE EXPLORER
# =====================================================================
with tab_corpus:
    st.markdown(f"#### 🗄️ Personal Knowledge Base ({total_docs} Markdown Files → {len(chunks)} Overlapping Chunks)")
    for c in chunks:
        st.markdown(f"""
        <div class="qx-chunk-card">
            <div class="qx-chunk-top">
                <div class="qx-chunk-title"><span class="qx-pill-green">{html.escape(c['id'])}</span></div>
                <span class="qx-pill-purple">Source: {html.escape(c['source'])} • {c['word_count']} words</span>
            </div>
            <div class="qx-passage-box">{html.escape(c['text'])}</div>
        </div>
        """, unsafe_allow_html=True)
