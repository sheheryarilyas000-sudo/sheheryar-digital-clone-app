import os
import io
import glob
import time
import requests
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM PRO UI STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sheheryar Ilyas | AI Digital Clone",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Container Tweaks */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0f766e 100%);
        color: #f8fafc;
        padding: 1.6rem 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-sub {
        font-size: 0.98rem;
        color: #cbd5e1;
        margin-top: 0.35rem;
        margin-bottom: 0.9rem;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .tech-badge {
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #e2e8f0;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    /* Required AI Clone Disclaimer Banner */
    .clone-banner {
        background: linear-gradient(90deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        padding: 0.75rem 1.25rem;
        border-radius: 10px;
        border-left: 5px solid #f59e0b;
        font-weight: 600;
        font-size: 0.92rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    /* Retrieved Chunk Cards */
    .chunk-card {
        background: rgba(248, 250, 252, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .router-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        background: #0d9488;
        color: white;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. DATA LOADING, OVERLAPPING CHUNKING & HYBRID INDEXING
# ---------------------------------------------------------
DATA_DIR = "data"
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")

@st.cache_resource
def load_and_index_knowledge_base(chunk_size=220, overlap=40):
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
        return [], None, None, None, 0

    # BM25 Index
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)

    # Dense Embeddings (Precompute & Save as .npy)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    if os.path.exists(EMBEDDINGS_PATH):
        dense_embeddings = np.load(EMBEDDINGS_PATH)
        if dense_embeddings.shape[0] != len(chunks):
            dense_embeddings = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
            np.save(EMBEDDINGS_PATH, dense_embeddings)
    else:
        dense_embeddings = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
        np.save(EMBEDDINGS_PATH, dense_embeddings)

    return chunks, bm25_index, embedder, dense_embeddings, len(md_files)

chunks, bm25_index, embedder, dense_embeddings, total_docs = load_and_index_knowledge_base()

# ---------------------------------------------------------
# 3. AGENTIC QUERY ROUTER & HYBRID RETRIEVAL ENGINE
# ---------------------------------------------------------
def agentic_query_router(query: str):
    """Classifies user intent for transparency and routing feedback."""
    q = query.lower()
    if any(w in q for w in ["erpify", "fyp", "architecture", "supabase", "sandbox", "eliteconnect", "scrollnix", "project", "code", "stack"]):
        return "🚀 Technical Project & Architecture", "Prioritizing exact technical terms + architectural context"
    elif any(w in q for w in ["cgpa", "semester", "umt", "university", "internship", "healsend", "certificate", "degree", "study", "junior"]):
        return "🎓 Academic & Career Profile", "Balancing factual bio records with personal study philosophy"
    elif any(w in q for w in ["hobby", "free time", "relax", "unwind", "movie", "harry potter", "wednesday", "sport", "badminton", "gemini", "laptop", "adata"]):
        return "🎬 Lifestyle, Gear & Hot Takes", "Combining semantic intent with specific tool/media keywords"
    elif any(w in q for w in ["phone", "cnic", "password", "address", "bank", "real person"]):
        return "🛡️ Privacy & Identity Guardrail", "Enforcing strict clone persona and privacy rules"
    else:
        return "🧠 General Personal Inquiry", "Running balanced Hybrid RAG across all 6 personal documents"

def min_max_normalize(arr):
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v - min_v < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return (arr - min_v) / (max_v - min_v)

def retrieve_chunks(query, mode="Hybrid", fusion_type="RRF (Reciprocal Rank)", alpha=0.5, top_k=3):
    if not chunks:
        return []

    # 1. BM25 Keyword Scoring
    tokenized_query = query.lower().split()
    bm25_scores = np.array(bm25_index.get_scores(tokenized_query), dtype=float)
    bm25_ranks = np.argsort(bm25_scores)[::-1]
    bm25_rank_map = {idx: rank + 1 for rank, idx in enumerate(bm25_ranks)}

    # 2. Dense Semantic Scoring (Cosine Similarity via NumPy)
    query_vec = embedder.encode([query], normalize_embeddings=True)[0]
    dense_scores = np.dot(dense_embeddings, query_vec).astype(float)
    dense_ranks = np.argsort(dense_scores)[::-1]
    dense_rank_map = {idx: rank + 1 for rank, idx in enumerate(dense_ranks)}

    # Normalized scores for Weighted Fusion & visual bars
    bm25_norm = min_max_normalize(bm25_scores)
    dense_norm = min_max_normalize(dense_scores)

    results = []
    for idx in range(len(chunks)):
        b_score = float(bm25_scores[idx])
        d_score = float(dense_scores[idx])
        b_rank = bm25_rank_map[idx]
        d_rank = dense_rank_map[idx]

        # Fusion Method A: Reciprocal Rank Fusion (1 / (60 + rank))
        rrf_score = (1.0 / (60.0 + b_rank)) + (1.0 / (60.0 + d_rank))
        # Fusion Method B: Min-Max Normalized Weighted Score
        weighted_score = (alpha * float(dense_norm[idx])) + ((1.0 - alpha) * float(bm25_norm[idx]))

        if mode == "BM25 only":
            primary_score = b_score
        elif mode == "Dense only":
            primary_score = d_score
        else:
            primary_score = rrf_score if "RRF" in fusion_type else weighted_score

        results.append({
            "chunk": chunks[idx],
            "score": primary_score,
            "bm25_score": b_score,
            "dense_score": d_score,
            "bm25_norm": float(bm25_norm[idx]),
            "dense_norm": float(dense_norm[idx]),
            "rrf_score": rrf_score,
            "weighted_score": weighted_score,
            "bm25_rank": b_rank,
            "dense_rank": d_rank
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# ---------------------------------------------------------
# 4. GROUNDED FIRST-PERSON GENERATION WITH MEMORY
# ---------------------------------------------------------
def generate_clone_response(question, retrieved_items, chat_history):
    context_blocks = []
    for item in retrieved_items:
        c = item["chunk"]
        context_blocks.append(f"[{c['id']} | Source: {c['source']}]\n{c['text']}")
    context_str = "\n\n".join(context_blocks)

    # Short conversation memory (last 2 turns)
    history_str = ""
    if chat_history:
        recent = chat_history[-4:]
        history_lines = [f"{m['role'].upper()}: {m['content']}" for m in recent]
        history_str = "\n".join(history_lines)

    persona_prompt = f"""You are the AI clone of Sheheryar Ilyas, a 7th-semester BSCS student at UMT Sialkot.
Speak in the first person ("I", "my"), in a confident, friendly, and clear professional tone.
Rules:
1. Answer ONLY using the CONTEXT below. It comes from my own documents.
2. If the context does not contain the answer, say exactly: "I don't have that in my notes." Do not guess or invent facts about me.
3. Never share phone numbers, addresses, ID numbers or other private details.
4. If asked whether you are the real person, clearly state that you are an AI clone of Sheheryar Ilyas.

RECENT CONVERSATION MEMORY:
{history_str if history_str else "No prior turns."}

CONTEXT:
{context_str}

QUESTION:
{question}

ANSWER:"""

    gemini_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not gemini_key:
        return "⚠️ Please add `GEMINI_API_KEY` in Streamlit Secrets to enable LLM generation."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
    payload = {
        "contents": [{"parts": [{"text": persona_prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 350}
    }
    
    try:
        res = requests.post(url, json=payload, timeout=30)
        res.raise_for_status()
        data = res.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"⚠️ Generation API Error: {str(e)}"

# ---------------------------------------------------------
# 5. SIDEBAR: CONTROL CENTER & .NPY DOWNLOAD
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ RAG Control Center")
    st.caption("Tune the Hybrid Retrieval & Fusion pipeline in real time.")

    retrieval_mode = st.radio(
        "🔍 Retrieval Mode:",
        options=["Hybrid", "BM25 only", "Dense only"],
        index=0,
        help="Switch between lexical (BM25), semantic (Dense), or combined (Hybrid) search."
    )

    fusion_type = st.selectbox(
        "⚡ Hybrid Fusion Formula:",
        options=["RRF (Reciprocal Rank)", "Weighted Linear (Min-Max)"],
        index=0,
        disabled=(retrieval_mode != "Hybrid"),
        help="RRF uses 1/(60+rank). Weighted combines normalized BM25 & Dense scores."
    )

    alpha_val = 0.6
    if retrieval_mode == "Hybrid" and "Weighted" in fusion_type:
        alpha_val = st.slider("Dense Weight (α):", 0.0, 1.0, 0.6, 0.05)

    top_k = st.slider("📑 Top-K Chunks:", min_value=1, max_value=5, value=3)

    st.divider()
    st.markdown("### 📊 Knowledge Base Telemetry")
    c1, c2 = st.columns(2)
    c1.metric("Documents", f"{total_docs} files")
    c2.metric("Total Chunks", f"{len(chunks)}")
    st.caption("Model: `all-MiniLM-L6-v2` (384-d) + `BM25Okapi`")

    # Download precomputed embeddings.npy button
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

    if st.button("🗑️ Clear Chat Memory", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["last_retrieved"] = []
        st.rerun()

# ---------------------------------------------------------
# 6. HERO BANNER & REQUIRED DISCLAIMER
# ---------------------------------------------------------
st.markdown("""
<div class="hero-card">
    <div class="hero-title">⚡ Sheheryar Ilyas — Interactive Digital Clone</div>
    <div class="hero-sub">BS Computer Science (7th Sem) @ UMT Sialkot • Software Quality Engineering • AI Builder (ERPify & EliteConnect AI)</div>
    <div class="badge-row">
        <span class="tech-badge">🧠 Hybrid RAG Architecture</span>
        <span class="tech-badge">🔎 BM25 Lexical + MiniLM Dense</span>
        <span class="tech-badge">⚖️ RRF & Weighted Fusion</span>
        <span class="tech-badge">🧭 Agentic Intent Router</span>
        <span class="tech-badge">🔒 100% Grounded First-Person Persona</span>
    </div>
</div>

<div class="clone-banner">
    <span>⚠️ <strong>OFFICIAL NOTICE: This is an AI clone, not the real person.</strong> All responses are strictly grounded in Sheheryar's personal Markdown documents.</span>
    <span>CLO-2 • Agentic AI</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 7. MAIN TABS: LIVE CLONE STUDIO + BENCHMARK MATRIX + KB EXPLORER
# ---------------------------------------------------------
tab_chat, tab_eval, tab_kb = st.tabs([
    "💬 Live Clone Studio & X-Ray",
    "🧪 5-Question Mode Benchmark (Report Live View)",
    "📂 Knowledge Base Explorer"
])

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi! I'm the AI digital clone of **Sheheryar Ilyas**. Ask me anything about my Final Year Project (**ERPify**), my **UMT Sialkot** academic journey, my **dual-storage laptop & local LLM workflow**, or my **hobbies & hot takes**!"
        }
    ]

with tab_chat:
    col_chat, col_xray = st.columns([1.35, 1.0], gap="large")

    with col_chat:
        st.markdown("#### ⚡ Quick Starter Prompts (Click to Test)")
        q_cols = st.columns(2)
        preset_questions = [
            ("🔑 Exact Keyword (BM25 Win)", "What is my setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B?"),
            ("🌊 Paraphrased (Dense Win)", "How do I unwind and relax when I am exhausted from coding?"),
            ("🏗️ FYP Architecture (Hybrid)", "Explain the two-layer architecture and security in ERPify."),
            ("🔥 University Hot Take", "What is your honest opinion on traditional university degrees?"),
            ("🎨 AI Tools Preference", "Which AI model do you prefer for daily coding vs image generation?"),
            ("🛑 Out-of-Scope Guardrail Test", "What is your favorite cryptocurrency trading strategy and bank name?")
        ]

        triggered_question = None
        for idx, (label, q_text) in enumerate(preset_questions):
            with q_cols[idx % 2]:
                if st.button(label, key=f"preset_{idx}", use_container_width=True):
                    triggered_question = q_text

        st.divider()

        # Chat History Container
        chat_container = st.container(height=420)
        with chat_container:
            for msg in st.session_state["messages"]:
                with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "assistant" else "🙋‍♂️"):
                    if "route" in msg:
                        st.markdown(f"<span class='router-pill'>{msg['route']} | Mode: {msg.get('mode', 'Hybrid')}</span>", unsafe_allow_html=True)
                    st.markdown(msg["content"])

        user_input = st.chat_input("Ask Sheheryar's Clone anything...")
        active_query = triggered_question if triggered_question else user_input

        if active_query:
            st.session_state["messages"].append({"role": "user", "content": active_query})
            
            t0 = time.time()
            route_label, route_desc = agentic_query_router(active_query)
            retrieved = retrieve_chunks(
                active_query,
                mode=retrieval_mode,
                fusion_type=fusion_type,
                alpha=alpha_val,
                top_k=top_k
            )
            answer = generate_clone_response(active_query, retrieved, st.session_state["messages"][:-1])
            latency = time.time() - t0

            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "route": route_label,
                "mode": retrieval_mode
            })
            st.session_state["last_retrieved"] = retrieved
            st.session_state["last_query"] = active_query
            st.session_state["last_route"] = (route_label, route_desc)
            st.session_state["last_latency"] = latency
            st.session_state["last_mode"] = retrieval_mode
            st.rerun()

    # RIGHT PANEL: RAG X-RAY TELEMETRY & CHUNK INSPECTOR
    with col_xray:
        st.markdown("#### 🔬 Hybrid RAG X-Ray & Retrieved Passages")
        if "last_retrieved" in st.session_state and st.session_state["last_retrieved"]:
            r_label, r_desc = st.session_state.get("last_route", ("🧠 General", ""))
            lat = st.session_state.get("last_latency", 0.0)
            active_m = st.session_state.get("last_mode", retrieval_mode)

            st.info(f"**🧭 Agentic Router:** `{r_label}`  \n*{r_desc}* • **Mode:** `{active_m}` • **Latency:** `{lat:.2f}s`")

            for rank, item in enumerate(st.session_state["last_retrieved"], start=1):
                c = item["chunk"]
                with st.expander(f"🥇 Rank #{rank} — {c['id']}  |  Score: {item['score']:.4f}", expanded=(rank == 1)):
                    st.markdown(f"**📄 Source File:** `{c['source']}` | **🆔 Chunk ID:** `{c['id']}` | **📏 Length:** `{c['word_count']} words`")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("BM25 Score", f"{item['bm25_score']:.2f}", f"Rank #{item['bm25_rank']}")
                    m2.metric("Dense Cosine", f"{item['dense_score']:.4f}", f"Rank #{item['dense_rank']}")
                    m3.metric("Hybrid RRF", f"{item['rrf_score']:.4f}")

                    st.caption("Lexical (BM25) vs. Semantic (Dense) Normalized Strength:")
                    st.progress(min(1.0, max(0.0, item["bm25_norm"])), text=f"BM25 Normalized: {item['bm25_norm']:.2f}")
                    st.progress(min(1.0, max(0.0, item["dense_norm"])), text=f"Dense Normalized: {item['dense_norm']:.2f}")

                    st.markdown("**📜 Retrieved Passage Text:**")
                    st.info(c["text"])
        else:
            st.info("👈 Click any **Quick Starter Prompt** or type a question on the left to inspect live BM25 rankings, Dense Cosine scores, RRF fusion math, and retrieved source passages here.")

# ---------------------------------------------------------
# TAB 2: LIVE 5-QUESTION BENCHMARK MATRIX
# ---------------------------------------------------------
with tab_eval:
    st.markdown("### 🧪 Live Retriever Comparison Matrix (BM25 vs. Dense vs. Hybrid)")
    st.write("This interactive benchmark runs our 5 test questions across all three retrieval modes in real time to verify why BM25 and Dense fail differently and how Hybrid Fusion combines their strengths.")

    benchmark_qs = [
        ("Q1 (Exact Rare Keyword)", "What is my setup with ADATA and Qwen 2.5 7B?", "BM25 wins on exact hardware/model tokens (`ADATA`, `Qwen`)."),
        ("Q2 (Paraphrased / Zero Overlap)", "How do I relax when I am exhausted from coding?", "Dense wins by capturing semantic relaxation intent without shared words."),
        ("Q3 (Technical Architecture)", "How does the two-layer architecture in ERPify ensure security?", "Hybrid ranks both architecture and encryption chunks at the top."),
        ("Q4 (Personal Opinion)", "What is my honest opinion on traditional university degrees?", "Hybrid merges my hot take in `hobbies.md` with advice in `advice.md`."),
        ("Q5 (Tool Comparison)", "Which AI model do I prefer for image generation vs daily coding?", "Hybrid surfaces both my narrative preference and direct FAQ answer.")
    ]

    rows = []
    for q_type, q_str, verdict in benchmark_qs:
        b_top = retrieve_chunks(q_str, mode="BM25 only", top_k=1)[0]
        d_top = retrieve_chunks(q_str, mode="Dense only", top_k=1)[0]
        h_top = retrieve_chunks(q_str, mode="Hybrid", top_k=1)[0]
        rows.append({
            "Test Case": f"{q_type}: {q_str}",
            "BM25 Top-1 (Score)": f"{b_top['chunk']['id']} ({b_top['bm25_score']:.2f})",
            "Dense Top-1 (Cosine)": f"{d_top['chunk']['id']} ({d_top['dense_score']:.3f})",
            "Hybrid Top-1 (RRF)": f"{h_top['chunk']['id']} ({h_top['rrf_score']:.4f})",
            "Why Mode Wins": verdict
        })
    st.table(rows)

# ---------------------------------------------------------
# TAB 3: KNOWLEDGE BASE EXPLORER
# ---------------------------------------------------------
with tab_kb:
    st.markdown("### 📂 Indexed Knowledge Base Documents & Overlapping Chunks")
    st.write(f"Currently indexing **{total_docs} Markdown files** split into **{len(chunks)} overlapping chunks** (~220 words per chunk, 40-word overlap).")
    for c in chunks:
        with st.expander(f"📄 {c['id']} — Source: {c['source']} ({c['word_count']} words)"):
            st.write(c["text"])
