---
title: Sheheryar Digital Clone
emoji: 🤖
colorFrom: green
colorTo: pink
sdk: static
pinned: false
---

# 🤖 Sheheryar Ilyas — Digital Clone (Hybrid RAG)

> **⚠️ Live App Link:** [Click Here to Open Live Streamlit App](PASTE_YOUR_STREAMLIT_URL_HERE_LATER)

---

## 📄 One-Page Assignment Report

### 1. Knowledge Base Chosen and Why
I utilized 6 distinct Markdown documents (`bio.md`, `projects.md`, `hobbies.md`, `faq.md`, `advice.md`, and `workflow.md`) to build a comprehensive and personal knowledge base[cite: 18, 22]. This specific mix covers factual biography details (BSCS 7th semester at UMT Sialkot, 3.18 CGPA, SQE skills), deep architectural documentation (my Final Year Project **ERPify**, **EliteConnect AI**, **Scrollnix**), my hardware/software workflow (dual-storage ADATA SSD + Toshiba HDD setup, local LLM testing with Qwen 2.5 7B), personal hobbies (Harry Potter, Wednesday Addams, badminton), and 20 first-person FAQs[cite: 16, 18]. This ensures the clone has rich context for both exact keyword matching (BM25) and semantic understanding (Dense retrieval)[cite: 17, 18].

### 2. Retrieval & Hybrid Fusion Methods
* **Implemented Fusion Method:** I implemented **Reciprocal Rank Fusion (RRF)** using the formula $\text{score} = \sum \frac{1}{60 + \text{rank}}$, which merges the rankings from `rank_bm25` and `sentence-transformers` (`all-MiniLM-L6-v2`) without requiring score normalization[cite: 18].
* **Alternative Method:** Another standard approach is **Weighted Linear Combination** ($\alpha \cdot \text{dense} + (1 - \alpha) \cdot \text{bm25}$) after applying Min-Max normalization to both score distributions[cite: 18].

### 3. Test Questions Evaluation (BM25 vs. Dense vs. Hybrid)

| Question | BM25 Result | Dense Result | Hybrid Result | Best Mode and Why |
| :--- | :--- | :--- | :--- | :--- |
| **Q1 (Exact Rare Keyword):** What is my setup with ADATA and Qwen 2.5 7B? | Retrieved `workflow.md` chunk #1 at Rank 1 (matched exact terms `ADATA` and `Qwen`). | Missed exact hardware chunk at Rank 1; ranked general AI chunks higher. | Ranked `workflow.md` chunk #1 at the top. | **BM25**, because it excels at matching exact rare tokens and model/hardware names[cite: 17, 18]. |
| **Q2 (Paraphrased / No Shared Words):** How do I relax when I am exhausted from coding? | Failed to match relevant context (0 keyword overlap with "free time" or "unwind"). | Retrieved `hobbies.md` (matched semantic meaning of relaxing with watching movies and reels). | Ranked `hobbies.md` in top chunks alongside `faq.md`. | **Dense**, because semantic embeddings capture intent even when zero exact words overlap[cite: 17, 18]. |
| **Q3:** How does the two-layer architecture in ERPify ensure security? | Retrieved `projects.md` based on `ERPify` and `architecture`. | Retrieved `projects.md` focusing on encryption and sandboxing concepts. | Placed the exact security & architecture chunks of `projects.md` at Rank 1 and 2. | **Hybrid**, as it combined the exact project title match with semantic security concepts. |
| **Q4:** What is my honest opinion on traditional university degrees? | Matched "university" across `bio.md` and `hobbies.md`. | Matched opinionated passages in `advice.md` and `hobbies.md`. | Surfaced both `hobbies.md` and `advice.md` at the very top. | **Hybrid**, because it gathered both my hot take and junior advice together. |
| **Q5:** Which AI model do I prefer for image generation vs daily coding? | Matched `Gemini Pro` and `Imagen` in `hobbies.md`. | Matched tech preference passages in `hobbies.md` and `faq.md`. | Ranked `hobbies.md` Rank 1 and `faq.md` Rank 2 with highest confidence. | **Hybrid**, providing both the narrative context and direct FAQ answer. |

### 4. Failure Case Analysis
When I tested the clone with the question *"Do you develop desktop applications in C#/.NET?"*, the retriever pulled the competitor analysis section from `projects.md`, which mentions *"Local Sialkot desktop software (C#/.NET)"* as a competitor to ERPify. Because those keywords appeared in my document, BM25 scored that chunk highly, and the LLM initially got slightly confused before clarifying that ERPify is actually a cloud-based Next.js SaaS. This happened because chunking split the competitor comparison away from the main project intro, causing keyword overlap without full surrounding negation context.

### 5. Future Improvement
Next, I would add an **Agentic Query Router** and a **Cross-Encoder Reranker** (`ms-marco-MiniLM-L-6-v2`) over the top-10 hybrid chunks so that the system first classifies whether a query is a personal fact, technical architecture question, or out-of-scope, and then re-scores the retrieved passages for deeper contextual accuracy[cite: 21].