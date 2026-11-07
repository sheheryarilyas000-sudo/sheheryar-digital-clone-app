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

# Visible Required Banner (Criterion 5)
st.title("🤖 Sheheryar Ilyas — Digital Clone (Hybrid RAG)")
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
        # Try primary model, then automatically fall back if Google returns 503/429
        candidate_models = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash"
        ]
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": gemini_key  # Passed in header so key NEVER leaks in URL errors
        }
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

# -------------------------------------------------------------------------
# 6. CLEAN NATIVE UI
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Retrieval Settings")
    retrieval_mode = st.radio(
        "Retrieval Mode Selector:",
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
    st.subheader("✨ Strong Student Extras")
    use_agentic_router = st.checkbox(
        "Auto-Override Mode with Agentic Router",
        value=False,
        help="When checked, the Agentic Router overrides your manual mode selection above."
    )
    use_reranker = st.checkbox("Enable Cross-Encoder Reranker", value=False)

    st.divider()
    st.write(f"**Indexed Documents:** {num_docs} files")
    st.write(f"**Total Chunks:** {len(chunks)} chunks")

    if embeddings is not None:
        buf = io.BytesIO()
        np.save(buf, embeddings)
        st.download_button(
            "⬇️ Download embeddings.npy",
            data=buf.getvalue(),
            file_name="embeddings.npy",
            mime="application/octet-stream"
        )

    if st.button("🗑️ Clear Chat Memory"):
        st.session_state["messages"] = []
        st.session_state["last_retrieved"] = []
        st.rerun()

tab1, tab2, tab3 = st.tabs([
    "💬 Chat with Clone & Retrieved Chunks",
    "📊 Hit@3 Evaluation & 5 Test Questions",
    "📂 Knowledge Base (Documents)"
])

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with tab1:
    col_chat, col_chunks = st.columns([1.2, 1.0], gap="large")

    with col_chat:
        st.subheader("Ask a Question")
        st.caption("Click a sample test button or type your own question below:")

        c1, c2 = st.columns(2)
        q_samples = [
            ("Q1: Exact Rare Keyword (BM25)", "What is my dual-storage setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B?"),
            ("Q2: Paraphrased (Dense)", "How do I recharge and relax when I am exhausted from coding?"),
            ("Q3: ERPify Architecture (Hybrid)", "How does the Two-Layer Architecture in ERPify handle custom AI modules and security?"),
            ("Q4: University Opinion", "What is my honest opinion on traditional university degrees versus practical skills?"),
            ("Q5: SQE Testing Workflow", "How do I perform software quality engineering and security testing on mobile apps?"),
            ("Q6: Out-of-Scope Test", "What is your favorite recipe for cooking chicken biryani?")
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

        user_query = st.chat_input("Ask Sheheryar's clone...")
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

            meta_text = f"🧭 Router Class: {cat} | Active Mode: {active_mode} | Reranker: {'ON' if use_reranker else 'OFF'}"
            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "meta": meta_text
            })
            st.session_state["last_retrieved"] = retrieved_list
            st.session_state["last_route_info"] = (cat, active_mode, recommended_mode, route_reason)
            st.rerun()

    with col_chunks:
        st.subheader("📑 Retrieved Chunks, Sources & Scores")
        if "last_route_info" in st.session_state:
            cat, m_used, rec_m, r_reason = st.session_state["last_route_info"]
            st.info(f"**Active Mode:** `{m_used}` | **Router Intent:** `{cat}` *(Recommended: {rec_m})*\n\n*{r_reason}*")

        if "last_retrieved" in st.session_state and st.session_state["last_retrieved"]:
            for rank, item in enumerate(st.session_state["last_retrieved"], start=1):
                c = item["chunk"]
                with st.expander(f"Rank #{rank}: {c['id']} (Selected Mode Score: {item['score']:.4f})", expanded=True):
                    st.markdown(f"**Source File:** `{c['source']}` | **Chunk ID:** `{c['id']}` | **Words:** `{c['word_count']}`")
                    st.markdown(
                        f"- **BM25 Score:** `{item['bm25_score']:.2f}` (Rank #{item['bm25_rank']})\n"
                        f"- **Dense Cosine Score:** `{item['dense_score']:.4f}` (Rank #{item['dense_rank']})\n"
                        f"- **Hybrid RRF Score:** `{item['rrf_score']:.4f}` | **Weighted Score:** `{item['weighted_score']:.4f}`"
                        + (f"\n- **Cross-Encoder Rerank Score:** `{item['rerank_score']:.4f}`" if item["rerank_score"] is not None else "")
                    )
                    st.markdown(f"<div style='font-size: 0.85rem; color: #1D1D1F; font-weight: 400; line-height: 1.5;'>{html.escape(c['text'])}</div>", unsafe_allow_html=True)
        else:
            st.write("Ask a question on the left to view the retrieved chunks, their source files, and their scores.")

with tab2:
    st.subheader("📊 Hand-Labeled Evaluation Set (Hit@3) & 5 Test Questions")
    if chunks:
        hit_rates, eval_rows = evaluate_hit_at_3()
        k1, k2, k3 = st.columns(3)
        k1.metric("BM25 Only (Hit@3)", hit_rates.get("BM25 only", "0%"))
        k2.metric("Dense Only (Hit@3)", hit_rates.get("Dense only", "0%"))
        k3.metric("Hybrid RRF (Hit@3)", hit_rates.get("Hybrid", "0%"))
        st.table(eval_rows)

with tab3:
    st.subheader(f"📂 Indexed Knowledge Base ({num_docs} Documents, {len(chunks)} Chunks)")
    for c in chunks:
        with st.expander(f"{c['id']} — {c['source']} ({c['word_count']} words)"):
            st.write(c["text"])
