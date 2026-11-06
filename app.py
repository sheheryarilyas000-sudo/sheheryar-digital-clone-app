import os
import glob
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# Page Configuration
st.set_page_config(
    page_title="Sheheryar Ilyas - Digital Clone",
    page_icon="🤖",
    layout="wide"
)

# 6. Required Visible Banner
st.title("🤖 Sheheryar Ilyas — Digital Clone (Hybrid RAG)")
st.warning("⚠️ **This is an AI clone, not the real person.** It answers strictly using Sheheryar's personal knowledge base documents.")

DATA_DIR = "data"
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")

# 1. Load and Chunk Documents (200 to 300 words with overlap, keeping file name and chunk ID)
@st.cache_resource
def load_and_index_knowledge_base():
    os.makedirs(DATA_DIR, exist_ok=True)
    md_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.md")))
    
    chunks = []
    chunk_size = 220  # Target 200-300 words
    overlap = 40      # Small overlap

    for filepath in md_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        
        words = text.split()
        if not words:
            continue
            
        step = chunk_size - overlap
        chunk_idx = 1
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            if not chunk_words:
                break
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "id": f"{filename}#chunk_{chunk_idx}",
                "source": filename,
                "text": chunk_text
            })
            chunk_idx += 1
            if i + chunk_size >= len(words):
                break

    if not chunks:
        return [], None, None, None

    # 2. BM25 Retriever Index
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)

    # 3. Dense Retriever (sentence-transformers + numpy cosine similarity + .npy caching)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    if os.path.exists(EMBEDDINGS_PATH):
        dense_embeddings = np.load(EMBEDDINGS_PATH)
        # Recompute if document chunks changed
        if dense_embeddings.shape[0] != len(chunks):
            dense_embeddings = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
            np.save(EMBEDDINGS_PATH, dense_embeddings)
    else:
        dense_embeddings = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
        np.save(EMBEDDINGS_PATH, dense_embeddings)

    return chunks, bm25_index, embedder, dense_embeddings

chunks, bm25_index, embedder, dense_embeddings = load_and_index_knowledge_base()

# 3 & 4. Retrieve & Hybrid Fusion (Reciprocal Rank Fusion: score = sum of 1 / (60 + rank))
def retrieve_chunks(query, mode="Hybrid", top_k=3):
    if not chunks:
        return []

    # BM25 Ranking
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranks = np.argsort(bm25_scores)[::-1]
    bm25_rank_map = {idx: rank + 1 for rank, idx in enumerate(bm25_ranks)}

    # Dense Cosine Similarity using NumPy
    query_vec = embedder.encode([query], normalize_embeddings=True)[0]
    dense_scores = np.dot(dense_embeddings, query_vec)
    dense_ranks = np.argsort(dense_scores)[::-1]
    dense_rank_map = {idx: rank + 1 for rank, idx in enumerate(dense_ranks)}

    results = []
    for idx in range(len(chunks)):
        b_score = float(bm25_scores[idx])
        d_score = float(dense_scores[idx])
        b_rank = bm25_rank_map[idx]
        d_rank = dense_rank_map[idx]

        # RRF Formula: 1 / (60 + rank)
        rrf_score = (1.0 / (60 + b_rank)) + (1.0 / (60 + d_rank))

        if mode == "BM25 only":
            primary_score = b_score
        elif mode == "Dense only":
            primary_score = d_score
        else:  # Hybrid (RRF)
            primary_score = rrf_score

        results.append({
            "chunk": chunks[idx],
            "score": primary_score,
            "bm25_score": b_score,
            "dense_score": d_score,
            "rrf_score": rrf_score,
            "bm25_rank": b_rank,
            "dense_rank": d_rank
        })

    # Sort descending by selected mode's score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# 5. Grounded First-Person Generation
def generate_clone_response(question, retrieved_items):
    context_blocks = []
    for item in retrieved_items:
        c = item["chunk"]
        context_blocks.append(f"[{c['id']} | Source: {c['source']}]\n{c['text']}")
    context_str = "\n\n".join(context_blocks)

    persona_prompt = f"""You are the AI clone of Sheheryar Ilyas, a 7th-semester BSCS student at UMT Sialkot.
Speak in the first person ("I", "my"), in a clear, natural, and professional tone.
Rules:
1. Answer ONLY using the CONTEXT below. It comes from my own documents.
2. If the context does not contain the answer, say exactly: "I don't have that in my notes." Do not guess or invent facts about me.
3. Never share phone numbers, addresses, ID numbers or other private details.
4. If asked whether you are the real person, say you are an AI clone of Sheheryar Ilyas.

CONTEXT:
{context_str}

QUESTION:
{question}

ANSWER:"""

    hf_token = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)
    
    if not hf_token:
        return "⚠️ API Key (`HF_TOKEN`) is not configured in Secrets. Showing retrieved context in the right panel."

    try:
        client = InferenceClient(token=hf_token)
        response = client.chat_completion(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": persona_prompt}],
            max_tokens=300,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Generation API Error: {str(e)}"

# 6. UI Layout
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("💬 Ask My Clone")
    retrieval_mode = st.radio(
        "Select Retrieval Mode:",
        options=["Hybrid", "BM25 only", "Dense only"],
        index=0,
        horizontal=True
    )
    top_k = st.slider("Number of Chunks to Retrieve (top-k):", min_value=1, max_value=5, value=3)
    
    user_question = st.text_input(
        "Your Question:",
        placeholder="e.g., What is the architecture of your FYP ERPify?"
    )
    
    ask_btn = st.button("Ask Clone", type="primary")

    if ask_btn and user_question.strip():
        if not chunks:
            st.error("No `.md` documents found inside the `data/` folder! Please make sure your 6 markdown files are inside `data/`.")
        else:
            with st.spinner("Retrieving passages and generating response..."):
                retrieved = retrieve_chunks(user_question, mode=retrieval_mode, top_k=top_k)
                answer = generate_clone_response(user_question, retrieved)
            
            st.markdown("### 🗣️ Clone's Answer")
            st.success(answer)
            st.session_state["last_retrieved"] = retrieved
            st.session_state["last_mode"] = retrieval_mode

with col2:
    st.subheader("📑 Retrieved Chunks & Scores")
    if "last_retrieved" in st.session_state and st.session_state["last_retrieved"]:
        st.caption(f"Showing top results for mode: **{st.session_state.get('last_mode', 'Hybrid')}**")
        for rank, item in enumerate(st.session_state["last_retrieved"], start=1):
            c = item["chunk"]
            with st.expander(f"Rank #{rank}: {c['id']} (Score: {item['score']:.4f})", expanded=True):
                st.markdown(f"**Source File:** `{c['source']}` | **Chunk ID:** `{c['id']}`")
                st.markdown(
                    f"**Mode Score:** `{item['score']:.4f}`  \n"
                    f"*(BM25 Score: `{item['bm25_score']:.2f}` [Rank #{item['bm25_rank']}] | "
                    f"Dense Cosine: `{item['dense_score']:.4f}` [Rank #{item['dense_rank']}] | "
                    f"RRF Score: `{item['rrf_score']:.4f}`)*"
                )
                st.write(c["text"])
    else:
        st.info("Submit a question on the left to inspect the retrieved chunks, sources, and scores here.")