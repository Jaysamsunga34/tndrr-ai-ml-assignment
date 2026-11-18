# Tndrr AI/ML Internship Assignment

Reproducible, framework-free implementation and empirical evaluation covering **LoRA fine-tuning**, **from-scratch RAG**, and **local quantized LLM inference optimization**.

## Repository Structure

```text
tndrr-ai-ml-assignment/
├── task1_finetuning/       # LoRA fine-tuning (Qwen2.5-0.5B-Instruct) + 7B VRAM analysis
├── task2_rag/              # From-scratch RAG over NIST AI RMF 1.0 (48 pages)
├── task3_optimization/     # Q4_K_M vs Q8_0 llama.cpp Metal benchmark + 24 GB serving analysis
├── report/                 # report.pdf (4 pages) & report/build_report.py
├── validate_experiments.py # Automated verification of all required metrics & artifacts
├── VIVA_PREPARATION.md     # 18 technical interview questions linked to observed results
└── requirements.txt
```

## Summary of Measured Results

| Task | Key Experiment | Measured Result | Results Location |
| :--- | :--- | :--- | :--- |
| **Task 1** | `Qwen2.5-0.5B-Instruct` LoRA (`r=8, alpha=16`, 0.22% trainable) on 68 held-out leak-free tests | Exact JSON Match: `0.00% -> 32.35%`; Category: `20.59% -> 75.00%`; Priority: `51.47% -> 77.94%`; Intent: `0.00% -> 51.47%`; JSON Validity: `100.0%` | `task1_finetuning/results/` |
| **Task 2** | Manual RAG (`PyMuPDF` + `all-MiniLM-L6-v2` + `FAISS IndexFlatIP`) over NIST AI RMF 1.0 (48 pp, 15 QA) | 256-tok chunks: **Recall@5 = 93.33%**, MRR = `0.7889` vs. 512-tok chunks: Recall@5 = `86.67%`, MRR = `0.7500` | `task2_rag/results/` |
| **Task 3** | `Qwen2.5-1.5B-Instruct` (`llama.cpp` Metal, 10 prompts × 3 reps) `Q4_K_M` vs. `Q8_0` | Decode: **107.78 vs. 71.83 tok/s (1.50×)**; Prefill: `1,707 vs. 1,736 tok/s`; Peak RSS: **1,403 vs. 2,141 MB (-34.5%)** | `task3_optimization/results/` |

## Setup & Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing All Experiments

```bash
# Task 1: Dataset generation, LoRA training, and held-out Base vs. Fine-Tuned evaluation
python task1_finetuning/scripts/generate_dataset.py
python task1_finetuning/scripts/train.py
python task1_finetuning/scripts/evaluate.py

# Task 2: Download NIST AI RMF 1.0, run 256 vs. 512 chunk eval, generate cited RAG answers
python task2_rag/scripts/download_doc.py
python task2_rag/scripts/evaluate_rag.py
python task2_rag/scripts/generate_rag_answers.py

# Task 3: Download Q4_K_M & Q8_0 GGUFs and benchmark 10 prompts x 3 repetitions
python task3_optimization/scripts/download_models.py
python task3_optimization/scripts/benchmark_quantization.py

# Validate all assignment artifacts & regenerate 4-page report
python validate_experiments.py
python report/build_report.py
```

## Hardware & Software Used

- **Hardware:** Apple M5 (arm64, 24 GB Unified Memory, Metal GPU / MPS)
- **Software:** Python 3.11.15, PyTorch 2.15.0 (MPS), Transformers 5.23.0, PEFT 0.21.3, Sentence-Transformers 6.3.0, FAISS-CPU 1.16.0, PyMuPDF 2.1, `llama-cpp-python` 0.3.37 (Metal), ReportLab 5.1.0.
