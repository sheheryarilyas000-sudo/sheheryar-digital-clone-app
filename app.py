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
# 1. PAGE CONFIG & APPLE CUPERTINO 3D FROSTED GLASS DESIGN SYSTEM
# =====================================================================
st.set_page_config(
    page_title="Sheheryar Ilyas — AI Digital Clone",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    /* Apple Alabaster Surface & High-Contrast Typography */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: linear-gradient(180deg, #F5F5F7 0%, #ECECF1 100%) !important;
        color: #1D1D1F !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif !important;
    }
    [data-testid="stHeader"] {
        background: rgba(245, 245, 247, 0.75) !important;
        backdrop-filter: blur(12px) !important;
    }
    .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1360px !important;
    }

    /* Guarantee High Contrast Across All Native Streamlit Widgets */
    p, span, label, div, h1, h2, h3, h4, h5, h6 {
        color: #1D1D1F;
    }

    /* Apple Frosted Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.88) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(0, 0, 0, 0.08) !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #1D1D1F !important;
    }

    /* Fix Selectbox & Input Contrast */
    [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #D2D2D7 !important;
        border-radius: 10px !important;
        color: #1D1D1F !important;
    }
    [data-baseweb="select"] span {
        color: #1D1D1F !important;
        font-weight: 500 !important;
    }

    /* Apple 3D Hero Header Card */
    .apple-hero {
        background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
        border: 1px solid rgba(0, 0, 0, 0.07);
        border-radius: 22px;
        padding: 22px 28px;
        margin-bottom: 18px;
        box-shadow:
            0 12px 32px -8px rgba(0, 0, 0, 0.06),
            0 2px 6px rgba(0, 0, 0, 0.03),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
    }
    .apple-hero-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .apple-avatar {
        width: 54px;
        height: 54px;
        border-radius: 16px;
        background: linear-gradient(135deg, #0071E3 0%, #00C7BE 100%);
        color: #FFFFFF !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
        font-weight: 800;
        box-shadow: 0 8px 18px rgba(0, 113, 227, 0.28);
    }
    .apple-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #1D1D1F !important;
        letter-spacing: -0.4px;
        margin: 0;
        line-height: 1.2;
    }
    .apple-subtitle {
        font-size: 0.88rem;
        color: #515154 !important;
        margin-top: 3px;
        font-weight: 500;
    }

    /* Required AI Clone Disclaimer Pill */
    .apple-disclaimer {
        background: #FFF8E6;
        border: 1px solid #F5D07A;
        color: #7A4F01 !important;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 2px 6px rgba(245, 158, 11, 0.08);
    }

    /* Apple Segmented Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(229, 229, 234, 0.75) !important;
        padding: 5px !important;
        border-radius: 14px !important;
        gap: 6px !important;
        border: none !important;
        margin-bottom: 16px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #515154 !important;
        border-radius: 10px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #1D1D1F !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0,0,0,0.04) !important;
    }

    /* Tactile 3D Prompt Buttons (Full Width, Never Truncated) */
    .stButton > button {
        background: #FFFFFF !important;
        color: #1D1D1F !important;
        border: 1px solid rgba(0, 0, 0, 0.09) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        padding: 0.55rem 0.9rem !important;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.04), inset 0 1px 0 #FFFFFF !important;
        transition: all 0.18s ease !important;
        text-align: left !important;
        white-space: normal !important;
        height: auto !important;
        min-height: 42px !important;
    }
    .stButton > button:hover {
        border-color: #0071E3 !important;
        color: #0071E3 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(0, 113, 227, 0.12) !important;
    }

    /* Chat Messages High-Contrast Apple Bubbles */
    [data-testid="stChatMessage"] {
        background: #FFFFFF !important;
        border: 1px solid rgba(0, 0, 0, 0.07) !important;
        border-radius: 16px !important;
        padding: 14px 18px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03) !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] div {
        color: #1D1D1F !important;
        font-size: 0.92rem !important;
        line-height: 1.55 !important;
    }

    /* 3D Apple Evidence Card (Right Panel) */
    .apple-card {
        background: #FFFFFF;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 18px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow:
            0 10px 24px -6px rgba(0, 0, 0, 0.05),
            0 2px 6px rgba(0, 0, 0, 0.02);
    }
    .apple-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
        padding-bottom: 10px;
        margin-bottom: 12px;
        border-bottom: 1px solid #F0F0F5;
    }
    .apple-badge-blue {
        background: #E8F2FF;
        color: #0066CC !important;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .apple-badge-green {
        background: #E6F9F0;
        color: #057A55 !important;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .apple-badge-gray {
        background: #F2F2F7;
        color: #3A3A3C !important;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* 4-Column Score Strip Inside Chunk Card */
    .apple-score-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        margin-bottom: 12px;
    }
    .apple-score-cell {
        background: #F9F9FB;
        border: 1px solid #E5E5EA;
        border-radius: 12px;
        padding: 8px 10px;
        text-align: center;
    }
    .apple-score-cell .lbl {
        font-size: 0.68rem;
        font-weight: 700;
        color: #6E6E73 !important;
        text-transform: uppercase;
        display: block;
    }
    .apple-score-cell .val {
        font-size: 0.94rem;
        font-weight: 800;
        color: #1D1D1F !important;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 2px;
        display: block;
    }
    .apple-score-cell .rk {
        font-size: 0.66rem;
        color: #0071E3 !important;
        font-weight: 600;
    }

    /* Clean Passage Box */
    .apple-passage {
        background: #F5F5F7;
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 0.85rem;
        line-height: 1.6;
        color: #2C2C2E !important;
        border-left: 3px solid #0071E3;
    }

    /* Clean Apple Benchmark Table */
    .apple-table {
        width: 100%;
        border-collapse: collapse;
        background: #FFFFFF;
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #E5E5EA;
        font-size: 0.85rem;
    }
    .apple-table th {
        background: #F5F5F7;
        color: #515154 !important;
        font-weight: 700;
        text-align: left;
        padding: 12px 14px;
        border-bottom: 1px solid #E5E5EA;
        font-size: 0.75rem;
        text-transform: uppercase;
    }
    .apple-table td {
        padding: 12px 14px;
        border-bottom: 1px solid #F0F0F5;
        color: #1D1D1F !important;
        vertical-align: top;
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
    q = query.lower()

    private_triggers = ["cnic", "passport", "phone number", "home address", "bank account", "password", "credit card"]
    if any(t in q for t in private_triggers):
        return {
            "category": "Out of Scope / Privacy Guardrail",
            "recommended_mode": "Decline",
            "reason": "Blocked sensitive personal/financial query per privacy rules."
        }

    rare_tokens = ["adata", "toshiba", "qwen", "23018020011", "scrollnix", "eliteconnect", "erpify", "rls", "c#/.net"]
    if any(t in q for t in rare_tokens):
        return {
            "category": "Personal Fact (Exact Keyword)",
            "recommended_mode": "Hybrid",
            "reason": "Detected exact hardware/project terms; combining BM25 lexical match with Dense context."
        }

    opinion_triggers = ["opinion", "hot take", "advice", "think about", "feel about", "relax", "unwind", "exhausted", "free time", "hobby", "favorite"]
    if any(t in q for t in opinion_triggers):
        return {
            "category": "Opinion & Lifestyle (Semantic)",
            "recommended_mode": "Dense only",
            "reason": "Paraphrased/opinion query detected; Dense embeddings capture conceptual intent."
        }

    return {
        "category": "General Profile Inquiry",
        "recommended_mode": "Hybrid",
        "reason": "Balanced Hybrid RRF retrieval across all 6 personal documents."
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

    tokens = query.lower().split()
    bm25_scores = np.array(bm25_index.get_scores(tokens), dtype=float)
    bm25_order = np.argsort(bm25_scores)[::-1]
    bm25_ranks = {idx: r + 1 for r, idx in enumerate(bm25_order)}

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

        rrf_sc = (1.0 / (60.0 + b_rk)) + (1.0 / (60.0 + d_rk))
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

    candidates.sort(key=lambda x: x["score"], reverse=True)

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
        "type": "Paraphrased (No Shared Words)",
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

        if any(r["chunk"]["source"] in exp for r in b_res): hits["BM25 only"] += 1
        if any(r["chunk"]["source"] in exp for r in d_res): hits["Dense only"] += 1
        if any(r["chunk"]["source"] in exp for r in h_res): hits["Hybrid"] += 1

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
# 7. CLEAN APPLE SIDEBAR (RETRIEVAL & STRONG-STUDENT CONTROLS)
# =====================================================================
with st.sidebar:
    st.markdown("### 🎛️ RAG Engine Controls")
    st.caption("Configure retrieval mode, fusion formula, and strong-student extras.")

    retrieval_mode = st.radio(
        "Retrieval Mode:",
        options=["Hybrid", "BM25 only", "Dense only"],
        index=0
    )

    fusion_method = st.selectbox(
        "Hybrid Fusion Formula:",
        options=["RRF (1 / (60 + rank))", "Weighted Linear (Min-Max)"],
        index=0,
        disabled=(retrieval_mode != "Hybrid")
    )

    alpha_weight = 0.6
    if retrieval_mode == "Hybrid" and "Weighted" in fusion_method:
        alpha_weight = st.slider("Dense Weight (α):", 0.0, 1.0, 0.6, 0.05)

    top_k = st.slider("Top-K Chunks:", min_value=1, max_value=5, value=3)

    st.divider()
    st.markdown("### ✨ Bonus Features")
    agentic_auto_mode = st.toggle(
        "🧭 Agentic Auto-Mode Router",
        value=False,
        help="Automatically classifies each question and picks the best retrieval mode."
    )
    use_reranker = st.toggle(
        "⚡ Cross-Encoder Reranker",
        value=False,
        help="Reranks top passages using ms-marco-MiniLM-L-6-v2."
    )

    st.divider()
    st.caption(f"📚 **Indexed:** {total_docs} Docs ({len(chunks)} Overlapping Chunks) • **Hit@3:** {hit_rates['Hybrid']:.0f}%")

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

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["last_retrieved"] = []
        st.rerun()

# =====================================================================
# 8. MINIMAL APPLE 3D HERO HEADER & REQUIRED VISIBLE BANNER
# =====================================================================
st.markdown(f"""
<div class="apple-hero">
    <div class="apple-hero-left">
        <div class="apple-avatar">SI</div>
        <div>
            <div class="apple-title">Sheheryar Ilyas — Digital Clone</div>
            <div class="apple-subtitle">BSCS (7th Semester) @ UMT Sialkot • Hybrid RAG (BM25 + Dense Embeddings + RRF)</div>
        </div>
    </div>
    <div class="apple-disclaimer">
        ⚠️ <strong>This is an AI clone, not the real person.</strong> Grounded in {total_docs} personal documents.
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# 9. ORGANIZED 3-TAB APPLE WORKSPACE
# =====================================================================
tab_chat, tab_eval, tab_kb = st.tabs([
    "💬 Ask My Clone & Retrieved Passages",
    "📊 5-Question Benchmark & Hit@3 Report",
    "📂 Personal Knowledge Base (6 Docs)"
])

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi! I am the AI digital clone of **Sheheryar Ilyas**. Ask me anything about my Final Year Project (**ERPify**), my **ADATA/Toshiba dual-storage & Qwen 2.5 7B setup**, my **UMT Sialkot** academics, or my **hobbies & opinions**!"
        }
    ]

with tab_chat:
    col_chat, col_inspect = st.columns([1.15, 1.0], gap="large")

    with col_chat:
        st.markdown("##### ⚡ Sample Test Questions (Click any card to test)")
        b_col1, b_col2 = st.columns(2)
        sample_prompts = [
            ("🔑 Exact Keyword (BM25 Win)", "What is my dual-storage setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B?"),
            ("🌊 Paraphrased (Dense Win)", "How do I unwind and recharge when I am exhausted from coding?"),
            ("🏗️ ERPify FYP Architecture", "How does the two-layer architecture in ERPify ensure security?"),
            ("🎓 University Hot Take", "What is your honest opinion on traditional university degrees?"),
            ("🤖 AI Model Preference", "Which AI model do you prefer for daily coding vs image generation?"),
            ("🛡️ Out-of-Scope Test", "What is your favorite recipe for cooking chicken biryani?")
        ]

        clicked_q = None
        for idx, (label, q_text) in enumerate(sample_prompts):
            with (b_col1 if idx % 2 == 0 else b_col2):
                if st.button(label, key=f"sample_{idx}", use_container_width=True):
                    clicked_q = q_text

        chat_box = st.container(height=420)
        with chat_box:
            for msg in st.session_state["messages"]:
                with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "assistant" else "🙋‍♂️"):
                    if "badge" in msg:
                        st.markdown(msg["badge"], unsafe_allow_html=True)
                    st.markdown(msg["content"])

        typed_q = st.chat_input("Ask a question about Sheheryar...")
        active_q = clicked_q if clicked_q else typed_q

        if active_q:
            st.session_state["messages"].append({"role": "user", "content": active_q})

            t0 = time.time()
            route_info = agentic_router(active_q)

            effective_mode = retrieval_mode
            if agentic_auto_mode and route_info["recommended_mode"] != "Decline":
                effective_mode = route_info["recommended_mode"]

            retrieved = retrieve_passages(
                active_q,
                mode=effective_mode,
                fusion_method=fusion_method,
                alpha=alpha_weight,
                use_reranker=use_reranker,
                top_k=top_k
            )

            answer = generate_grounded_reply(
                active_q,
                retrieved,
                st.session_state["messages"][:-1],
                route_info
            )
            dt = time.time() - t0

            rerank_lbl = " + CrossEncoder" if use_reranker else ""
            badge_html = (
                f"<div style='margin-bottom:6px;'>"
                f"<span class='apple-badge-blue'>🧭 {html.escape(route_info['category'])}</span> "
                f"<span class='apple-badge-green'>Mode: {html.escape(effective_mode)}{rerank_lbl} ({dt:.2f}s)</span>"
                f"</div>"
            )

            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "badge": badge_html
            })
            st.session_state["last_retrieved"] = retrieved
            st.session_state["last_route"] = route_info
            st.session_state["last_mode"] = effective_mode
            st.rerun()

    # RIGHT PANEL: CLEAN APPLE 3D EVIDENCE CARDS
    with col_inspect:
        st.markdown("##### 📑 Retrieved Passages, Sources & Scores")

        if "last_retrieved" in st.session_state and st.session_state["last_retrieved"]:
            r_info = st.session_state.get("last_route", {})
            eff_mode = st.session_state.get("last_mode", retrieval_mode)

            st.markdown(f"""
            <div class="apple-card" style="padding:12px 16px; background:#F0F7FF; border-color:#BFE0FF;">
                <div style="font-size:0.82rem; font-weight:700; color:#0055B3;">
                    🧭 Intent Router: {html.escape(r_info.get('category', 'General'))} — Active Mode: {html.escape(eff_mode)}
                </div>
                <div style="font-size:0.78rem; color:#3A3A3C; margin-top:2px;">
                    {html.escape(r_info.get('reason', ''))}
                </div>
            </div>
            """, unsafe_allow_html=True)

            for rank, item in enumerate(st.session_state["last_retrieved"], start=1):
                c = item["chunk"]
                ce_str = f"{item['rerank_score']:.3f}" if item["rerank_score"] is not None else "—"

                st.markdown(f"""
                <div class="apple-card">
                    <div class="apple-card-header">
                        <div>
                            <span class="apple-badge-blue">Rank #{rank}</span>
                            <strong style="margin-left:8px; font-size:0.9rem; color:#1D1D1F;">{html.escape(c['id'])}</strong>
                        </div>
                        <span class="apple-badge-gray">📄 {html.escape(c['source'])} ({c['word_count']} words)</span>
                    </div>
                    <div class="apple-score-strip">
                        <div class="apple-score-cell">
                            <span class="lbl">BM25 Score</span>
                            <span class="val">{item['bm25_score']:.2f}</span>
                            <span class="rk">Rank #{item['bm25_rank']}</span>
                        </div>
                        <div class="apple-score-cell">
                            <span class="lbl">Dense Cosine</span>
                            <span class="val">{item['dense_score']:.3f}</span>
                            <span class="rk">Rank #{item['dense_rank']}</span>
                        </div>
                        <div class="apple-score-cell">
                            <span class="lbl">Hybrid RRF</span>
                            <span class="val">{item['rrf_score']:.4f}</span>
                            <span class="rk">Fused</span>
                        </div>
                        <div class="apple-score-cell">
                            <span class="lbl">Reranker</span>
                            <span class="val">{ce_str}</span>
                            <span class="rk">Cross-Enc</span>
                        </div>
                    </div>
                    <div class="apple-passage">{html.escape(c['text'])}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="apple-card" style="text-align:center; padding:45px 24px;">
                <div style="font-size:2.2rem; margin-bottom:10px;">📑</div>
                <div style="font-weight:700; font-size:1.05rem; color:#1D1D1F;">Retrieved Chunks & Scores Panel</div>
                <div style="font-size:0.86rem; color:#6E6E73; margin-top:6px; max-width:380px; margin-left:auto; margin-right:auto;">
                    Select any sample question on the left or ask your own to inspect the top-k retrieved chunks, their source files, and BM25 / Dense / Hybrid scores.
                </div>
            </div>
            """, unsafe_allow_html=True)

# =====================================================================
# TAB 2: BENCHMARK EVALUATION & HIT@3 METRICS
# =====================================================================
with tab_eval:
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("BM25 Lexical Hit@3", f"{hit_rates['BM25 only']:.0f}%", "Exact Rare Keywords")
    m_col2.metric("Dense Semantic Hit@3", f"{hit_rates['Dense only']:.0f}%", "Paraphrased Queries")
    m_col3.metric("Hybrid Fusion Hit@3", f"{hit_rates['Hybrid']:.0f}%", "Combined Best Rank")

    st.markdown("#### 📊 5-Question Evaluation Table (BM25 vs. Dense vs. Hybrid)")
    rows_html = ""
    for r in benchmark_rows:
        rows_html += f"""
        <tr>
            <td><strong>{html.escape(r['id'])}</strong><br><span class="apple-badge-gray">{html.escape(r['type'])}</span></td>
            <td style="font-weight:600;">{html.escape(r['question'])}</td>
            <td style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;">{html.escape(r['bm25_top'])}</td>
            <td style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;">{html.escape(r['dense_top'])}</td>
            <td style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:#057A55; font-weight:700;">{html.escape(r['hybrid_top'])}</td>
            <td>{html.escape(r['why'])}</td>
        </tr>
        """

    st.markdown(f"""
    <table class="apple-table">
        <thead>
            <tr>
                <th>Question ID</th>
                <th>Test Question</th>
                <th>BM25 Result</th>
                <th>Dense Result</th>
                <th>Hybrid Result</th>
                <th>Best Mode & Why</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="apple-card">
            <div style="font-weight:700; font-size:0.95rem; margin-bottom:6px;">⚠️ Failure Case & Root Cause</div>
            <div style="font-size:0.86rem; color:#3A3A3C; line-height:1.6;">
                When asked <em>"Do you build desktop software in C#/.NET?"</em>, BM25 ranked <code>projects.md#chunk_1</code> first because our ERPify competitor section mentions <em>"Local Sialkot desktop software (C#/.NET)"</em> as a legacy competitor. Lexical search matched the exact tokens without realizing the passage was contrasting ERPify against desktop apps. Combining Dense retrieval and Cross-Encoder reranking resolves this ambiguity.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="apple-card">
            <div style="font-weight:700; font-size:0.95rem; margin-bottom:6px;">🚀 Implemented Extras & Next Improvement</div>
            <div style="font-size:0.86rem; color:#3A3A3C; line-height:1.6;">
                Includes all four strong-student extras: (1) <strong>Agentic Query Router</strong>, (2) <strong>Cross-Encoder Reranker</strong> (<code>ms-marco-MiniLM-L-6-v2</code>), (3) <strong>Hit@3 Evaluation Benchmark</strong>, and (4) <strong>Short Conversation Memory</strong>. Next, we would add Markdown-header-aware semantic chunking.
            </div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# TAB 3: KNOWLEDGE BASE EXPLORER
# =====================================================================
with tab_kb:
    st.markdown(f"#### 📂 Personal Knowledge Base ({total_docs} Files → {len(chunks)} Overlapping Chunks)")
    for c in chunks:
        st.markdown(f"""
        <div class="apple-card">
            <div class="apple-card-header">
                <span class="apple-badge-blue">{html.escape(c['id'])}</span>
                <span class="apple-badge-gray">Source: {html.escape(c['source'])} • {c['word_count']} words</span>
            </div>
            <div class="apple-passage">{html.escape(c['text'])}</div>
        </div>
        """, unsafe_allow_html=True)
