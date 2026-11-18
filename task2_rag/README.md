# Task 2: RAG From Scratch (NIST AI RMF 1.0)

## Overview
Implements a complete Retrieval-Augmented Generation pipeline **without LangChain or LlamaIndex** over the 48-page public **NIST AI 100-1: Artificial Intelligence Risk Management Framework (AI RMF 1.0)** (January 2023, `https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf`).

## Pipeline Components
1. **PDF Extraction & Cleaning (`PyMuPDF`):** Page-indexed extraction with hyphenated line-break repair.
2. **Token-Aware Chunking:** Sliding-window chunker comparing **256 tokens** (115 chunks) vs. **512 tokens** (68 chunks), both with 50-token overlap.
3. **Dense Indexing:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim L2-normalized) indexed with `faiss.IndexFlatIP` (exact cosine similarity).
4. **Grounded Generation:** Top-K context assembly with `[Page X]` citations passed to local `Qwen2.5-0.5B-Instruct`.

## Key Files
- `scripts/download_doc.py` — Downloads NIST AI 100-1 PDF
- `scripts/rag_pipeline.py` — Extractor, chunker, FAISS index, context builder
- `scripts/evaluate_rag.py` — Evaluates Recall@1, Recall@5, MRR, and latency across 256 vs. 512 chunks
- `scripts/generate_rag_answers.py` — End-to-end cited answer generation
- `evaluation/ground_truth_qa.json` — 15 verbatim-grounded questions & physical page numbers *(marked for personal review/rewriting)*
- `results/chunk_comparison.csv`, `retrieval_metrics.json`, `failure_analysis.csv`, `retrieval_analysis.md`, `rag_answers.json`
