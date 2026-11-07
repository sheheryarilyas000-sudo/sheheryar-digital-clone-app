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

> **⚠️ This is an AI clone, not the real person.** It answers strictly using Sheheryar Ilyas's personal documents.

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

| Question | BM25 Result | Dense Result | Hybrid Result | Best Mode and Why |
| :--- | :--- | :--- | :--- | :--- |
| **Q1 (Exact Rare Keyword):** What is my setup with ADATA SSD, Toshiba HDD, and Qwen 2.5 7B? | ✅ Retrieved `workflow.md` chunk #1 at Rank 1 — exact token matches on `ADATA`, `Toshiba`, `Qwen 2.5 7B` | ❌ Missed hardware chunk at Rank 1; pushed generic AI productivity chunks higher | ✅ Ranked `workflow.md` chunk #1 at top | **BM25** — excels at exact rare token matching; these hardware/model names are unique strings that embeddings don't distinguish well |
| **Q2 (Paraphrased / No Shared Words):** How do I recharge and relax when I am exhausted from coding? | ❌ Zero keyword overlap with "free time", "unwind", or "movies" — returned unrelated chunks | ✅ Retrieved `hobbies.md` — semantically matched "relax/recharge" to watching movies and scrolling reels | ✅ Ranked `hobbies.md` at top via RRF | **Dense** — semantic embeddings capture intent when zero exact words overlap; this is the clearest win for dense retrieval |
| **Q3 (Technical Architecture):** How does the two-layer architecture in ERPify handle security? | ✅ Retrieved `projects.md` and `erpify_architecture.md` via `ERPify` and `architecture` keywords | ✅ Retrieved security/sandboxing passages semantically | ✅ Placed exact security + architecture chunks at Rank 1 and 2 | **Hybrid** — combined exact project-name match with semantic security concepts; neither retriever alone gets both |
| **Q4 (Personal Opinion):** What is my honest opinion on traditional university degrees vs practical skills? | ⚠️ Matched "university" across `bio.md` and `hobbies.md` but missed nuanced opinion | ✅ Matched opinionated passages in `advice.md` and `hobbies.md` | ✅ Surfaced both `hobbies.md` and `advice.md` at the top | **Hybrid** — gathered both my hot take (hobbies.md) and my structured junior advice (advice.md) together |
| **Q5 (SQE / Testing):** How do I perform software quality engineering and security testing on mobile apps? | ✅ Strong match on `SQE`, `BigBrains`, `testing` in `sqe_testing_notes.md` | ✅ Matched testing methodology passages semantically | ✅ Ranked `sqe_testing_notes.md` Rank 1 and `bio.md` Rank 2 | **Hybrid** — combined exact SQE keyword match with semantic testing methodology; highest confidence result |

---

### 4. Failure Case Analysis

**Question tested:** *"Do you develop desktop applications in C#/.NET?"*

**What happened:** The retriever pulled the competitor analysis section from `projects.md`, which mentions *"Local Sialkot desktop software (C#/.NET)"* as a competitor to ERPify. BM25 scored this chunk highly because the keywords matched, and the LLM initially produced a confused answer before clarifying that ERPify is actually a cloud-based Next.js SaaS — not a C#/.NET app.

**Why it happened:** Chunking split the competitor comparison away from the surrounding context that negates it. The chunk contained the words "C#/.NET desktop" without the negating sentence "ERPify replaces these." This caused keyword overlap without the full context, leading to a misleading retrieval. This is a classic BM25 failure — it matches tokens, not meaning.

---

### 5. One Improvement I Would Make Next

I would implement **conversation-aware query rewriting**. Currently, follow-up questions like *"What about its security?"* (after asking about ERPify) are sent to the retriever as bare phrases with no context. A small LLM call that rewrites the follow-up into a standalone question (*"How does ERPify handle security?"*) before retrieval would dramatically improve multi-turn accuracy without changing any other component of the pipeline.
