---
title: Sheheryar Digital Clone
emoji: 🤖
colorFrom: green
colorTo: pink
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: false
---

# 🤖 Sheheryar Ilyas — Digital Clone (Hybrid RAG)

> 🔗 **Live Streamlit App:** [Open Live Digital Clone](https://sheheryar-digital-clone-app-jzyfgbjn7j29gmrg2zetvs.streamlit.app/)
>
> ⚠️ **Notice:** This is an AI clone, not the real person. It answers strictly using Sheheryar Ilyas's personal documents.

---

## 📄 Assignment Report — CLO-2

### 1. Knowledge Base Chosen and Why

I built my knowledge base from **12 Markdown documents** covering every dimension of my personal and academic life:

| File | Type | Why Included |
|------|------|--------------|
| `bio.md` | Factual/CV | Education (BSCS 7th sem, UMT Sialkot, 3.18 CGPA), skills, experience |
| `faq.md` | First-person FAQ | 20+ Q&As answered in my own voice — makes the clone sound most like me |
| `hobbies.md` | Personality/Opinion | Harry Potter, Wednesday Addams, badminton, hot takes |
| `erpify_architecture.md` | Long-form | Deep technical write-up of my FYP (ERPify) two-layer architecture |
| `projects.md` | Long-form | EliteConnect AI, Scrollnix, ERPify project details |
| `advice.md` | Opinion | Advice I would give junior students |
| `workflow.md` | How-I-work | Dual-storage ADATA SSD + Toshiba HDD setup, Qwen 2.5 7B local LLM |
| `sqe_testing_notes.md` | Study Notes | SQE internship at BigBrains, testing methodologies |
| `cs_coursework_notes.md` | Study Notes | Core CS subjects and labs |
| `career_goals_values.md` | Values | Long-term goals and personal philosophy |
| `academic_projects_readme.md` | README | GitHub project documentation |
| `content_strategy_healsend.md` | Long-form | Content strategy work at Healsend internship |

This mix ensures BM25 has rare, exact keywords to win on (ADATA, Qwen 2.5 7B, ERPify, BigBrains) and Dense retrieval has semantically rich, plain-language passages to shine on paraphrased queries.

---

### 2. Retrieval & Hybrid Fusion Methods

**Implemented Fusion Method — Reciprocal Rank Fusion (RRF):**

$$\text{score} = \frac{1}{60 + \text{rank}_{BM25}} + \frac{1}{60 + \text{rank}_{Dense}}$$

RRF merges rankings from `rank_bm25` (BM25Okapi) and `sentence-transformers` (all-MiniLM-L6-v2 cosine similarity) without requiring score normalization, making it robust when the two score scales differ greatly.

**Alternative Method — Weighted Linear Combination:**

$$\text{score} = \alpha \cdot \text{dense\_norm} + (1 - \alpha) \cdot \text{bm25\_norm}$$

After Min-Max normalizing both score distributions to [0, 1], a tunable alpha weight blends them. This is also implemented in the app and selectable via the sidebar slider.

---

### 3. Five Test Questions — BM25 vs Dense vs Hybrid

**Overall Hit@3 Scores (from live app evaluation):**

| Mode | Hit@3 Score |
|------|-------------|
| BM25 Only | 80% (4/5) |
| Dense Only | 100% (5/5) |
| Hybrid RRF | 100% (5/5) |

| Question | BM25 Top-1 Result | Dense Top-1 Result | Hybrid Top-1 Result | Best Mode and Why |
| :--- | :--- | :--- | :--- | :--- |
| **Q1 (Exact Rare Keyword):** What is my dual-storage setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B? | `workflow.md#chunk_1` score=19.80 ✅ HIT | `workflow.md#chunk_1` score=0.294 ✅ HIT | `workflow.md#chunk_1` score=0.0328 ✅ HIT | **BM25** — highest raw score (19.80) confirms exact rare token matching wins on hardware/model names like ADATA, Toshiba, Qwen 2.5 7B |
| **Q2 (Paraphrased / No Shared Words):** How do I recharge and relax when I am exhausted from coding? | `sqe_testing_notes.md#chunk_1` score=7.60 ❌ MISS — no keyword overlap with hobbies | `hobbies.md#chunk_1` score=0.234 ✅ HIT — semantically matched "relax/recharge" to watching movies and scrolling reels | `hobbies.md#chunk_1` score=0.0318 ✅ HIT | **Dense** — only retriever that found the right chunk; BM25 completely missed because "recharge" and "movies" share zero words with the answer |
| **Q3 (FYP Architecture):** How does the Two-Layer Architecture in ERPify handle custom AI modules and security? | `faq.md#chunk_1` score=10.61 ✅ HIT | `erpify_architecture.md#chunk_1` score=0.658 ✅ HIT | `erpify_architecture.md#chunk_1` score=0.0325 ✅ HIT | **Hybrid** — Dense retrieved the deeper architecture doc while BM25 caught the FAQ mention; combined result is most complete |
| **Q4 (Personal Opinion):** What is my honest opinion on traditional university degrees versus practical skills? | `faq.md#chunk_3` score=8.20 ✅ HIT | `hobbies.md#chunk_2` score=0.621 ✅ HIT | `faq.md#chunk_3` score=0.0323 ✅ HIT | **Hybrid** — ranks both my opinion in `hobbies.md` and my structured junior advice in `faq.md` together for a complete answer |
| **Q5 (SQE & Testing):** How do I perform software quality engineering and security testing on mobile apps? | `sqe_testing_notes.md#chunk_1` score=11.45 ✅ HIT | `sqe_testing_notes.md#chunk_1` score=0.644 ✅ HIT | `sqe_testing_notes.md#chunk_1` score=0.0328 ✅ HIT | **Hybrid** — all modes agree; Hybrid wins by combining exact SQE keyword strength with semantic testing methodology understanding |

---

### 4. Failure Case Analysis

**Question tested:** *"How do I recharge and relax when I am exhausted from coding?"* — tested in **BM25 only** mode.

**What happened:** BM25 retrieved `sqe_testing_notes.md#chunk_1` (score: 7.60) as its top result — a complete **MISS**. The correct answer (watching movies, scrolling reels, playing badminton) is in `hobbies.md`. The LLM received the wrong chunk and produced an answer about software testing workflows instead of my actual hobbies. This failure is confirmed in the live Hit@3 evaluation: BM25 scored 80% (4/5) while Dense scored 100% (5/5) — Q2 is the exact question BM25 failed on.

**Why it happened:** This is the fundamental BM25 weakness. The question uses emotional language ("recharge", "relax", "exhausted") that shares **zero exact words** with the answer in `hobbies.md` ("movies", "reels", "badminton", "Harry Potter"). BM25 latched onto `sqe_testing_notes.md` by coincidence — that document contains statistically rare words that scored higher than the actually relevant document. Dense retrieval correctly solved this because embeddings capture semantic intent even with zero shared vocabulary.

---

### 5. One Improvement I Would Make Next

I would implement **conversation-aware query rewriting**. Currently, follow-up questions like *"What about its security?"* (after asking about ERPify) are sent to the retriever as bare phrases with no context. A small LLM call that rewrites the follow-up into a standalone question (*"How does ERPify handle security?"*) before retrieval would dramatically improve multi-turn accuracy without changing any other component of the pipeline.
