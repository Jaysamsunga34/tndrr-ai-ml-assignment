# Task 3: Local LLM Inference & Quantization Optimization

## Overview
Benchmarks identical architecture **`Qwen2.5-1.5B-Instruct`** under **Q4_K_M** vs. **Q8_0** GGUF quantizations using `llama.cpp` (`llama-cpp-python 0.3.37` with Apple M5 Metal GPU offload, `n_gpu_layers=-1`, `n_ctx=2048`, `temp=0.0`, `max_tokens=256`).

## Methodology
- **10 Standardized Prompts** (`prompts/benchmark_prompts.json`): factual explanation, logical reasoning, JSON extraction, constrained poem generation, summarization, Python coding, classification, multi-step instructions, arithmetic, and engineering persona critique.
- **3 Repetitions per Prompt per Configuration** with KV cache and native C-level `llama_perf_context()` timers reset before every run to separate **prefill throughput** from **decode throughput**.
- **Memory Metric:** Process Resident Set Size (RSS) in Apple Unified Memory via `psutil`.

## Key Files
- `scripts/download_models.py` — Downloads Q4_K_M and Q8_0 GGUFs from Hugging Face (`models/` excluded from Git)
- `scripts/benchmark_quantization.py` — Controlled throughput, latency, memory, and output benchmark
- `prompts/benchmark_prompts.json` — Standardized 10-prompt suite
- `results/benchmark_metrics.json`, `latency_comparison.csv`, `outputs_q4_vs_q8.json`, `quality_comparison.csv`, `theory_and_capacity_planning.md`
