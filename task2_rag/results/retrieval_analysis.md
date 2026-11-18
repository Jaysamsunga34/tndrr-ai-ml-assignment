# Task 2: RAG Retrieval Analysis & Theoretical Discussion

## 1. Document Metadata
- **Title:** NIST AI 100-1: Artificial Intelligence Risk Management Framework (AI RMF 1.0)
- **Source URL:** https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
- **Publication Date / Version:** January 2023 (Version 1.0)
- **Total Pages:** 48 pages (extracted via PyMuPDF / `fitz`)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim dense vectors, 256-token max sequence length)
- **Vector Index:** FAISS `IndexFlatIP` over L2-normalized embeddings (exact Cosine Similarity)
- **Test Set:** 15 original questions grounded verbatim in the NIST AI RMF 1.0 text (marked for human review in `task2_rag/evaluation/ground_truth_qa.json`).

---

## 2. Chunk-Size Experiment Results

Both configurations held the embedding model (`all-MiniLM-L6-v2`), overlap (`50 tokens`), top-K (`K=5`), and the 15-question evaluation suite strictly constant.

| Configuration | Chunk Size | Overlap | Total Chunks | Recall@1 | Recall@5 | MRR | Avg Query Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A** | 256 tokens | 50 tokens | 115 | 0.6667 (10/15) | **0.9333 (14/15)** | **0.7889** | 3.53 ms |
| **Config B** | 512 tokens | 50 tokens | 68 | 0.6667 (10/15) | 0.8667 (13/15) | 0.7500 | 3.17 ms |

### Why the Two Chunk Sizes Produced Different Retrieval Performance
1. **Encoder Max-Sequence Truncation:** `all-MiniLM-L6-v2` has a hard maximum sequence window of **256 WordPiece tokens**. When fed 512-token chunks (Config B), the tokenizer silently truncates the second half of every chunk. Any factual answer residing in tokens 257–512 becomes completely invisible to the dense vector representation.
2. **Semantic Dilution (Topic Averaging):** Smaller chunks (256 tokens) isolate single concepts, definitions, or figure captions, yielding sharper, higher-magnitude cosine similarities with focused user queries. Larger chunks (512 tokens) merge multiple unrelated subsections (e.g., figure captions + body paragraphs), diluting the pooled embedding vector.
3. **Latency vs. Precision Trade-Off:** Config B produced 41% fewer vectors (68 vs. 115), resulting in marginally faster flat search latency (3.17 ms vs. 3.53 ms), but at the cost of a 6.67 percentage-point drop in Recall@5 and lower MRR.

---

## 3. Genuine Retrieval Failure Analysis

### Failure Case 1: Cross-Page Fragmentation & Semantic Redundancy (`q05`)
- **User Question:** *"What are the key characteristics of trustworthy AI systems listed in the NIST AI RMF?"*
- **Correct Source Passage (Physical Page 8):** *"valid and reliable, safe, secure and resilient, accountable and transparent, explainable and interpretable, privacy enhanced, and fair with their harmful biases managed."*
- **Outcome:** Failed in **both** Config A (256 tokens) and Config B (512 tokens).
- **Actual Top-5 Retrieved Chunks (Config A):**
  1. Page 17 — Section 3 intro (*"3. AI Risks and Trustworthiness For AI systems to be trustworthy..."*)
  2. Page 5 — List of Figures (*"Fig. 1 Examples of potential harms related to AI systems..."*)
  3. Page 6 — OECD reference / adaptability overview
  4. Page 24 — Section 4 (*"Effectiveness of the AI RMF..."*)
  5. Page 7 — Executive Summary opening paragraph
- **Why Retrieval Failed:** Page-by-page extraction split a single sentence across a page boundary: the lead-in phrase (*"...characteristics of trustworthy AI:"*) appeared at the bottom of Page 7, while the actual comma-separated enumeration started at the top of Page 8. Without the header phrase `"characteristics of trustworthy AI"` inside the Page 8 chunk, dense retrieval favored generic section introductions on Pages 17 and 5 that repeated the query's phrasing verbatim.
- **Potential Improvement:** Concatenate document pages into a continuous text stream with character-to-page span mapping before chunking so page boundaries never sever lead-in clauses from enumerations, and prepend hierarchical section headers to each chunk (Contextual Retrieval).

### Failure Case 2: Exact Lexical Citation Lost to Dilution & Truncation (`q13`)
- **User Question:** *"Which 2022 OECD publication was Figure 2 (Lifecycle and Key Dimensions of an AI System) modified from?"*
- **Correct Source Passage (Physical Page 15):** *"Modified from OECD (2022) OECD Framework for the Classification of AI systems — OECD Digital Economy Papers."*
- **Outcome:** Retrieved at Rank 2 in Config A (256 tokens), but **failed entirely** in Config B (512 tokens).
- **Actual Top-5 Retrieved Chunks (Config B):**
  1. Page 5 — List of Figures (*"Fig. 1... Fig. 2..."*)
  2. Page 40 — Appendix A (*"Descriptions of AI Actor Tasks from Figures 2 and 3..."*)
  3. Page 2 — Commercial entity disclaimer
  4. Page 27 — GOVERN overview
  5. Page 7 — Social responsibility (ISO 26000:2010)
- **Why Retrieval Failed:** In Config B (512 tokens), the brief Figure 2 caption on Page 15 was merged into a long chunk dominated by prose about environmental groups and civil society actors. Because `all-MiniLM-L6-v2` both averages out the caption's signal and truncates past 256 tokens, the dense embedding lost specificity for the exact proper nouns (`OECD`, `2022`, `Figure 2`). Meanwhile, Appendix A and the List of Figures explicitly mentioned "Figures 2" in shorter, less diluted chunks.
- **Potential Improvement:** Deploy Hybrid Retrieval combining sparse **BM25** (which heavily weights rare exact tokens like `OECD`, `2022`, and `Fig. 2`) with dense vector search via Reciprocal Rank Fusion (RRF), plus a Cross-Encoder reranker.

---

## 4. Technical Explanations

### What BM25 Contributes Beyond Embedding-Based Search
Dense bi-encoder embeddings compress text into fixed-dimensional vectors that excel at capturing **semantic similarity and paraphrases** (e.g., matching *"stopping a dangerous model"* to *"cease development and deployment safely"*). However, dense compression frequently smooths over rare exact lexical identifiers: acronyms (`TEVV`), subcategory codes (`MEASURE 2.6`, `MANAGE 4.1`), standard numbers (`ISO 26000:2010`), and author/year citations (`OECD 2022`). **BM25** (Best Matching 25) is a probabilistic sparse retrieval function based on Term Frequency (with saturation) and Inverse Document Frequency (IDF), normalized by document length. Because rare technical identifiers carry very high IDF scores, BM25 guarantees exact lexical recall where dense models suffer from semantic drift. Combining both via Hybrid Search (e.g., Reciprocal Rank Fusion) yields strong semantic and exact-match coverage.

### What a Reranker Does
In a standard bi-encoder retrieval step, the query and each chunk are encoded **independently** into separate vectors and compared with a single dot product (cosine similarity). This is extremely fast (milliseconds over millions of vectors) but prevents token-level interaction between the query and passage. A **Cross-Encoder Reranker** (e.g., `BAAI/bge-reranker-base` or `ms-marco-MiniLM-L-6-v2`) takes a top-$N$ candidate set (e.g., top 20 chunks from Hybrid Search) and feeds each `(query, chunk)` pair **jointly** through a Transformer encoder with full self-attention across all query and document tokens. This enables fine-grained verification of whether the chunk actually *answers* the question rather than merely sharing topic vocabulary (eliminating false positives like List-of-Figures entries), significantly boosting Recall@1 and MRR at modest added latency.
