#!/usr/bin/env python3
"""
rag_pipeline.py - Complete RAG from scratch without any orchestration frameworks.
Custom implementation of:
  1. PDF text extraction & cleaning (PyMuPDF / fitz)
  2. Token-aware chunking (configurable chunk_size and overlap)
  3. Embedding generation (sentence-transformers: all-MiniLM-L6-v2)
  4. FAISS / NumPy cosine similarity indexing
  5. Top-K retrieval and context assembly
  6. Answer generation with source citations
"""

import os
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Lazy imports for ML packages to avoid slow startup
fitz = None
SentenceTransformer = None
faiss = None

def _lazy_load_deps():
    global fitz, SentenceTransformer, faiss
    if fitz is None:
        import fitz as _fitz
        fitz = _fitz
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as _ST
        SentenceTransformer = _ST
    if faiss is None:
        import faiss as _faiss
        faiss = _faiss

DOCUMENT_METADATA = {
    "title": "NIST AI 100-1: Artificial Intelligence Risk Management Framework (AI RMF 1.0)",
    "source_url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
    "publication_date": "January 2023",
    "version": "1.0",
    "expected_pages": 48
}

class DocumentExtractor:
    """Extracts and cleans page-by-page text from PDF documents."""

    @staticmethod
    def extract_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
        _lazy_load_deps()
        doc = fitz.open(pdf_path)
        pages = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            raw_text = page.get_text("text")

            # Basic cleaning: join hyphenated line breaks, collapse extraneous whitespace
            cleaned_text = re.sub(r"-\s*\n\s*", "", raw_text)
            cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
            cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text).strip()

            pages.append({
                "doc_name": Path(pdf_path).name,
                "page_num": page_idx + 1,  # 1-indexed
                "text": cleaned_text,
                "char_count": len(cleaned_text)
            })
        return pages

class TokenChunker:
    """Token-aware chunking with sliding window and specified overlap."""

    def __init__(self, chunk_size: int = 256, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = max(1, chunk_size - overlap)

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        chunk_counter = 0

        for page_info in pages:
            text = page_info["text"]
            if not text.strip():
                continue

            # Word-level proxy for tokenization (approx 1 token ~= 0.75 words, 1 word ~= 1.33 tokens)
            # To ensure standard token alignment, we split into whitespace words
            words = text.split()
            if not words:
                continue

            # Use token multiplier estimate: 1 word ~ 1.3 tokens -> chunk_size_words = chunk_size / 1.3
            words_per_chunk = max(20, int(self.chunk_size * 0.75))
            words_overlap = max(5, int(self.overlap * 0.75))
            step = max(1, words_per_chunk - words_overlap)

            for start_idx in range(0, len(words), step):
                chunk_words = words[start_idx : start_idx + words_per_chunk]
                if not chunk_words:
                    break
                chunk_text = " ".join(chunk_words)

                chunks.append({
                    "chunk_id": f"chunk_{self.chunk_size}_{chunk_counter}",
                    "doc_name": page_info["doc_name"],
                    "page_num": page_info["page_num"],
                    "chunk_index": chunk_counter,
                    "text": chunk_text,
                    "word_count": len(chunk_words),
                    "approx_token_count": int(len(chunk_words) * 1.33),
                    "chunk_size_config": self.chunk_size,
                    "overlap_config": self.overlap
                })
                chunk_counter += 1

                if start_idx + words_per_chunk >= len(words):
                    break

        return chunks

class RAGIndex:
    """Manages dense embeddings, FAISS indexing, and cosine similarity retrieval."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        _lazy_load_deps()
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.chunks: List[Dict[str, Any]] = []
        self.index = None
        self.dimension = None

    def build_index(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        print(f"Embedding {len(texts)} chunks with {self.model_name}...")
        embeddings = self.encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        self.dimension = embeddings.shape[1]

        # FAISS IndexFlatIP (Inner Product on normalized vectors = Cosine Similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        print(f"FAISS index built: {self.index.ntotal} vectors of dimension {self.dimension}.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vec = self.encoder.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)

        scores, indices = self.index.search(query_vec, top_k)
        retrieved = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1 or idx >= len(self.chunks):
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            chunk["rank"] = rank + 1
            retrieved.append(chunk)
        return retrieved

def assemble_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved passages into grounded context string with citations."""
    context_blocks = []
    for chunk in retrieved_chunks:
        block = (
            f"[Source: {chunk['doc_name']}, Page: {chunk['page_num']}, Chunk: {chunk['chunk_id']}]\n"
            f"{chunk['text']}"
        )
        context_blocks.append(block)
    return "\n\n".join(context_blocks)

def main():
    pdf_path = "/Users/jayadityashah/Downloads/tndrr-ai-ml-assignment/task2_rag/documents/NIST.AI.100-1.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} does not exist. Run download_doc.py first.")
        return

    print("Extracting document...")
    pages = DocumentExtractor.extract_from_pdf(pdf_path)
    print(f"Extracted {len(pages)} pages.")

    # Smoke test Config A (256 tokens)
    chunker_a = TokenChunker(chunk_size=256, overlap=50)
    chunks_a = chunker_a.chunk_pages(pages)
    print(f"Config A (256 tokens, 50 overlap): {len(chunks_a)} chunks created.")

    # Smoke test Config B (512 tokens)
    chunker_b = TokenChunker(chunk_size=512, overlap=50)
    chunks_b = chunker_b.chunk_pages(pages)
    print(f"Config B (512 tokens, 50 overlap): {len(chunks_b)} chunks created.")

if __name__ == "__main__":
    main()
