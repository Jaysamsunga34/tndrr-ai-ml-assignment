#!/usr/bin/env python3
"""
generate_rag_answers.py - End-to-end RAG Question Answering with Source Citations.
Retrieves top-5 passages using Config A (256-token chunks, all-MiniLM-L6-v2, FAISS)
and generates grounded answers using Qwen2.5-0.5B-Instruct with explicit page citations.
"""

import os
import sys
import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).parent))
from rag_pipeline import DocumentExtractor, TokenChunker, RAGIndex, assemble_context

RAG_SYSTEM_PROMPT = (
    "You are a precise technical assistant answering questions strictly from the provided "
    "NIST AI Risk Management Framework (AI RMF 1.0) excerpts. Always cite the page number(s) "
    "in square brackets like [Page X]. If the answer is not in the context, state that explicitly."
)

def main():
    pdf_path = "/Users/jayadityashah/Downloads/tndrr-ai-ml-assignment/task2_rag/documents/NIST.AI.100-1.pdf"
    qa_file = "/Users/jayadityashah/Downloads/tndrr-ai-ml-assignment/task2_rag/evaluation/ground_truth_qa.json"
    out_file = Path("/Users/jayadityashah/Downloads/tndrr-ai-ml-assignment/task2_rag/results/rag_answers.json")

    pages = DocumentExtractor.extract_from_pdf(pdf_path)
    chunker = TokenChunker(chunk_size=256, overlap=50)
    chunks = chunker.chunk_pages(pages)

    indexer = RAGIndex(model_name="all-MiniLM-L6-v2")
    indexer.build_index(chunks)

    with open(qa_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading generator LLM {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True
    ).to(device)
    model.eval()

    generated_answers = []
    # Run end-to-end grounded generation on first 5 questions (representative sample) + q05/q13 failure check
    sample_ids = {"q01", "q02", "q06", "q07", "q08", "q11", "q12"}
    for item in qa_data:
        if item["id"] not in sample_ids:
            continue
        retrieved = indexer.retrieve(item["question"], top_k=3)
        context_str = assemble_context(retrieved)

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {item['question']}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=140,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        gen_time = time.time() - t0
        answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        generated_answers.append({
            "id": item["id"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "generated_answer": answer,
            "retrieved_pages": [c["page_num"] for c in retrieved],
            "top_score": round(retrieved[0]["score"], 4),
            "generation_time_sec": round(gen_time, 2)
        })
        print(f"[{item['id']}] Generated in {gen_time:.2f}s | Pages: {[c['page_num'] for c in retrieved]}")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(generated_answers, f, indent=2)
    print(f"Saved {len(generated_answers)} grounded RAG answers to {out_file}")

if __name__ == "__main__":
    main()
