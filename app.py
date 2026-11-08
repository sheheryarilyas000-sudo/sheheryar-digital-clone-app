import os
import io
import glob
import requests
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Sheheryar Ilyas - AI Digital Clone",
    page_icon="🤖",
    layout="wide"
)

# =========================================================================
# PREMIUM MAC-STYLE DARK UI — CSS INJECTION
# =========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── App Background — deep space gradient ── */
.stApp {
    background: radial-gradient(ellipse at top left, #1a1035 0%, #0d0d1a 40%, #0a0a14 100%);
    min-height: 100vh;
}

/* ── Hide Streamlit default decorations ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

/* ── TITLE — gradient chrome text ── */
h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 50%, #a1ffce 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -0.5px;
    margin-bottom: 0 !important;
}

h2, h3 {
    color: #c4b5fd !important;
    font-weight: 600 !important;
    letter-spacing: -0.3px;
}

/* ── AI CLONE BANNER — premium gradient pill ── */
[data-testid="stAlert"] {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(217,70,239,0.15)) !important;
    border: 1px solid rgba(239,68,68,0.4) !important;
    border-radius: 14px !important;
    color: #fca5a5 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    backdrop-filter: blur(20px) !important;
    padding: 12px 18px !important;
    box-shadow: 0 4px 20px rgba(239,68,68,0.1) !important;
}

/* ── SIDEBAR — frosted glass panel ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
    backdrop-filter: blur(30px) !important;
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #d4d4d8 !important;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #a78bfa !important;
    font-size: 14px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* ── TABS — pill navigation ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.08);
    gap: 2px;
}

[data-testid="stTabs"] button[role="tab"] {
    border-radius: 9px !important;
    color: #71717a !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important;
}

/* ── BUTTONS — 3D glass effect ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(79,70,229,0.2)) !important;
    color: #c4b5fd !important;
    border: 1px solid rgba(124,58,237,0.4) !important;
    border-radius: 10px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    backdrop-filter: blur(10px) !important;
    text-align: center !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(124,58,237,0.5), rgba(79,70,229,0.5)) !important;
    border-color: rgba(167,139,250,0.7) !important;
    color: #ffffff !important;
    transform: translateY(-1px) scale(1.01) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.35) !important;
}

.stButton > button:active {
    transform: translateY(0px) scale(0.99) !important;
}

/* ── CHAT MESSAGES — glass cards ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    margin-bottom: 10px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06) !important;
    transition: box-shadow 0.2s ease !important;
}

[data-testid="stChatMessage"]:hover {
    box-shadow: 0 8px 32px rgba(124,58,237,0.15), inset 0 1px 0 rgba(255,255,255,0.08) !important;
}

/* ── CHAT INPUT — premium input field ── */
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    border-radius: 14px !important;
    color: #f4f4f5 !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s ease !important;
    backdrop-filter: blur(20px) !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(167,139,250,0.7) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
}

/* ── EXPANDERS — chunk cards ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    margin-bottom: 10px !important;
    backdrop-filter: blur(20px) !important;
    overflow: hidden !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2) !important;
}

[data-testid="stExpander"]:hover {
    border-color: rgba(124,58,237,0.3) !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.1) !important;
}

[data-testid="stExpander"] summary {
    color: #a78bfa !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 14px 16px !important;
}

[data-testid="stExpander"] summary:hover {
    color: #c4b5fd !important;
}

/* ── METRICS — premium stat cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    text-align: center !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    transition: transform 0.2s ease !important;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
}

[data-testid="stMetricValue"] {
    background: linear-gradient(135deg, #34d399, #60a5fa) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
}

[data-testid="stMetricLabel"] {
    color: #71717a !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

/* ── INFO BOX — blue glass ── */
[data-testid="stInfo"] {
    background: rgba(59,130,246,0.08) !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-left: 3px solid #3b82f6 !important;
    border-radius: 12px !important;
    color: #93c5fd !important;
    backdrop-filter: blur(10px) !important;
}

/* ── RADIO — custom styled ── */
[data-testid="stRadio"] > div {
    gap: 6px !important;
}

[data-testid="stRadio"] label {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    color: #d4d4d8 !important;
    font-size: 13px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(124,58,237,0.2) !important;
    border-color: rgba(124,58,237,0.5) !important;
    color: #c4b5fd !important;
}

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #d4d4d8 !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #7c3aed !important;
    border-color: #a78bfa !important;
}

/* ── TABLE — dark glass ── */
table {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

thead tr th {
    background: rgba(124,58,237,0.15) !important;
    color: #c4b5fd !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    padding: 12px !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
}

tbody tr td {
    color: #d4d4d8 !important;
    font-size: 13px !important;
    border-color: rgba(255,255,255,0.05) !important;
    padding: 10px 12px !important;
}

tbody tr:hover td {
    background: rgba(124,58,237,0.06) !important;
}

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    margin: 16px 0 !important;
}

/* ── CAPTIONS & SMALL TEXT ── */
.stCaption, small, [data-testid="stCaptionContainer"] {
    color: #52525b !important;
    font-size: 12px !important;
}

/* ── CODE / BACKTICK ── */
code {
    background: rgba(124,58,237,0.15) !important;
    color: #a78bfa !important;
    border-radius: 5px !important;
    padding: 1px 6px !important;
    font-size: 12px !important;
}

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.2)) !important;
    border-color: rgba(16,185,129,0.4) !important;
    color: #6ee7b7 !important;
}

[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, rgba(16,185,129,0.4), rgba(5,150,105,0.4)) !important;
    color: #ffffff !important;
}

/* ── CHECKBOX ── */
[data-testid="stCheckbox"] label {
    color: #d4d4d8 !important;
    font-size: 13px !important;
}

/* ── CUSTOM SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(124,58,237,0.4);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(167,139,250,0.6); }

/* ── COLUMN GAP FIX ── */
[data-testid="stHorizontalBlock"] { gap: 1.5rem !important; }

/* ── GENERAL TEXT ── */
p, li, span, div {
    color: #d4d4d8;
    line-height: 1.6;
}

/* ── Subheader glow line ── */
h3::after {
    content: '';
    display: block;
    width: 40px;
    height: 2px;
    background: linear-gradient(90deg, #7c3aed, transparent);
    margin-top: 4px;
    border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Premium Header Card ──
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(79,70,229,0.08));
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(30px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06);
    display: flex;
    align-items: center;
    gap: 16px;
">
    <div style="font-size: 48px; filter: drop-shadow(0 0 12px rgba(167,139,250,0.5));">🤖</div>
    <div>
        <div style="font-size: 1.7rem; font-weight: 800;
            background: linear-gradient(135deg, #e0c3fc, #8ec5fc, #a1ffce);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Sheheryar Ilyas — Digital Clone
        </div>
        <div style="font-size: 13px; color: #71717a; margin-top: 2px; font-weight: 400;">
            Hybrid RAG · BM25 + Dense Embeddings · Gemini-Powered
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.warning("⚠️ **This is an AI clone, not the real person.** It answers in first person strictly using Sheheryar Ilyas's personal documents.")

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
# OPTIONAL EXTRA #1: AGENTIC ROUTING STEP
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
# RETRIEVAL + HYBRID FUSION (RRF & Weighted) + RERANKER
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
# GENERATION (First Person, Grounded, Zero Key Leak, Multi-Model Fallback)
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
            for attempt in range(2):
                try:
                    r = requests.post(url, headers=headers, json=payload, timeout=25)
                    if r.status_code == 200:
                        data = r.json()
                        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    elif r.status_code in (429, 500, 502, 503, 504):
                        import time
                        time.sleep(1.2)
                        continue
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
# OPTIONAL EXTRA #3: HAND-LABELED EVALUATION SET WITH HIT@3
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
        q = item["query"]
        rel = item["relevant_files"]

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
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.markdown("### ⚙️ Retrieval Settings")
    retrieval_mode = st.radio(
        "Mode Selector:",
        options=["Hybrid", "BM25 only", "Dense only"],
        index=0
    )
    fusion_type = st.selectbox(
        "Hybrid Fusion Method:",
        options=["RRF", "Weighted"],
        index=0,
        disabled=(retrieval_mode != "Hybrid")
    )
    alpha = 0.5
    if retrieval_mode == "Hybrid" and fusion_type == "Weighted":
        alpha = st.slider("Dense Weight (alpha):", 0.0, 1.0, 0.5, 0.05)

    top_k = st.slider("Top-K Chunks:", 1, 5, 3)

    st.divider()
    st.markdown("### ✨ Strong Student Extras")
    use_agentic_router = st.checkbox(
        "Auto-Override with Agentic Router",
        value=False,
        help="Agentic Router overrides your manual mode selection."
    )
    use_reranker = st.checkbox("Enable Cross-Encoder Reranker", value=False)

    st.divider()
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
    border-radius:12px;padding:14px;font-size:13px;color:#71717a;">
        <div style="color:#a78bfa;font-weight:600;margin-bottom:8px;">📊 Index Stats</div>
        <div>📁 Documents: <span style="color:#c4b5fd;font-weight:600;">{num_docs}</span></div>
        <div>🧩 Chunks: <span style="color:#c4b5fd;font-weight:600;">{len(chunks)}</span></div>
        <div>🔢 Embedding Dim: <span style="color:#c4b5fd;font-weight:600;">384</span></div>
        <div>🤖 Model: <span style="color:#c4b5fd;font-weight:600;">all-MiniLM-L6-v2</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if embeddings is not None:
        buf = io.BytesIO()
        np.save(buf, embeddings)
        st.download_button(
            "⬇️ Download embeddings.npy",
            data=buf.getvalue(),
            file_name="embeddings.npy",
            mime="application/octet-stream"
        )

    if st.button("🗑️ Clear Chat Memory", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["last_retrieved"] = []
        st.rerun()

# =========================================================================
# TABS
# =========================================================================
tab1, tab2, tab3 = st.tabs([
    "💬 Chat with Clone",
    "📊 Hit@3 Evaluation",
    "📂 Knowledge Base"
])

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ── TAB 1: CHAT ──
with tab1:
    col_chat, col_chunks = st.columns([1.2, 1.0], gap="large")

    with col_chat:
        st.subheader("Ask a Question")
        st.caption("Click a quick-test button or type your own question below:")

        c1, c2 = st.columns(2)
        q_samples = [
            ("Q1: Exact Keyword · BM25", "What is my dual-storage setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B?"),
            ("Q2: Paraphrased · Dense", "How do I recharge and relax when I am exhausted from coding?"),
            ("Q3: ERPify Architecture", "How does the Two-Layer Architecture in ERPify handle custom AI modules and security?"),
            ("Q4: University Opinion", "What is my honest opinion on traditional university degrees versus practical skills?"),
            ("Q5: SQE Testing", "How do I perform software quality engineering and security testing on mobile apps?"),
            ("Q6: Out-of-Scope Test", "What is your favorite recipe for cooking chicken biryani?")
        ]

        selected_sample = None
        for i, (btn_lbl, q_txt) in enumerate(q_samples):
            with (c1 if i % 2 == 0 else c2):
                if st.button(btn_lbl, key=f"q_btn_{i}", use_container_width=True):
                    selected_sample = q_txt

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                if "meta" in msg:
                    st.caption(msg["meta"])
                st.write(msg["content"])

        user_query = st.chat_input("Ask Sheheryar's clone anything...")
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

            meta_text = f"🧭 Router: {cat} · Mode: {active_mode} · Reranker: {'ON ✅' if use_reranker else 'OFF'}"
            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "meta": meta_text
            })
            st.session_state["last_retrieved"] = retrieved_list
            st.session_state["last_route_info"] = (cat, active_mode, recommended_mode, route_reason)
            st.rerun()

    with col_chunks:
        st.subheader("📑 Retrieved Chunks & Scores")

        if "last_route_info" in st.session_state:
            cat, m_used, rec_m, r_reason = st.session_state["last_route_info"]
            st.info(f"**Mode:** `{m_used}` · **Intent:** `{cat}` · Recommended: `{rec_m}`\n\n_{r_reason}_")

        if "last_retrieved" in st.session_state and st.session_state["last_retrieved"]:
            for rank, item in enumerate(st.session_state["last_retrieved"], start=1):
                c = item["chunk"]
                # Score badge color
                score_color = "#34d399" if rank == 1 else "#60a5fa" if rank == 2 else "#a78bfa"
                with st.expander(f"#{rank} · {c['id']} · Score: {item['score']:.4f}", expanded=(rank == 1)):
                    st.markdown(f"""
                    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;">
                        <span style="background:rgba(124,58,237,0.2);color:#c4b5fd;border-radius:6px;padding:3px 10px;font-size:12px;font-weight:600;">
                            📄 {c['source']}
                        </span>
                        <span style="background:rgba(255,255,255,0.05);color:#71717a;border-radius:6px;padding:3px 10px;font-size:12px;">
                            {c['word_count']} words
                        </span>
                        <span style="background:rgba(52,211,153,0.1);color:#34d399;border-radius:6px;padding:3px 10px;font-size:12px;">
                            Rank #{rank}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**BM25:** `{item['bm25_score']:.2f}` _(#{item['bm25_rank']})_")
                        st.markdown(f"**RRF:** `{item['rrf_score']:.4f}`")
                    with col_b:
                        st.markdown(f"**Dense:** `{item['dense_score']:.4f}` _(#{item['dense_rank']})_")
                        st.markdown(f"**Weighted:** `{item['weighted_score']:.4f}`")

                    if item["rerank_score"] is not None:
                        st.markdown(f"**Cross-Encoder:** `{item['rerank_score']:.4f}` 🔁")

                    st.markdown("---")
                    st.markdown(f"<div style='color:#d4d4d8;font-size:13px;line-height:1.7'>{c['text']}</div>",
                                unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:40px;color:#3f3f46;">
                <div style="font-size:48px;margin-bottom:12px;">🔍</div>
                <div style="font-size:14px;">Ask a question to see retrieved chunks here</div>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 2: HIT@3 EVALUATION ──
with tab2:
    st.subheader("📊 Hand-Labeled Evaluation Set (Hit@3)")
    if chunks:
        hit_rates, eval_rows = evaluate_hit_at_3()
        k1, k2, k3 = st.columns(3)
        k1.metric("BM25 Only · Hit@3", hit_rates.get("BM25 only", "0%"))
        k2.metric("Dense Only · Hit@3", hit_rates.get("Dense only", "0%"))
        k3.metric("Hybrid RRF · Hit@3", hit_rates.get("Hybrid", "0%"))
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.table(eval_rows)

# ── TAB 3: KNOWLEDGE BASE ──
with tab3:
    st.subheader(f"📂 Knowledge Base — {num_docs} Documents · {len(chunks)} Chunks")
    for c in chunks:
        with st.expander(f"{c['id']} · {c['source']} · {c['word_count']} words"):
            st.markdown(f"<div style='color:#d4d4d8;font-size:13px;line-height:1.7'>{c['text']}</div>",
                        unsafe_allow_html=True)
