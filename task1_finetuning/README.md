# Task 1: LoRA Fine-Tuning for Support Ticket → Structured JSON

## Overview
Fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` using LoRA (`r=8`, `alpha=16`, `dropout=0.05`, target modules `q_proj, k_proj, v_proj, o_proj`, `lr=2e-4`, 3 epochs, seed=42) to classify natural-language customer support requests into strict JSON (`category`, `priority`, `intent`).

## Data Integrity (Zero Leakage)
- **Splits:** 504 Train / 28 Validation / 68 Held-Out Test (including 14 adversarial edge cases).
- **Leakage Prevention:** Disjoint intent phrasing descriptions and disjoint conversational templates across splits; assistant-only cross-entropy loss masking (`-100` on prompt tokens).

## Reproducing Inference with Saved Adapter

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base, "task1_finetuning/results/lora_adapter")
model.eval()
```

## Key Files
- `scripts/generate_dataset.py` — Deterministic leak-free dataset builder
- `scripts/train.py` — LoRA fine-tuning with MPS/CUDA support
- `scripts/evaluate.py` — Identical-prompt greedy evaluation of Base vs. Fine-Tuned
- `scripts/gpu_memory_calculation.md` — 7B Full vs. LoRA vs. QLoRA VRAM breakdown
- `notebooks/task1_lora_finetuning.ipynb` — Cloud Colab/Kaggle runnable notebook
- `results/evaluation_summary.json`, `baseline_predictions.csv`, `finetuned_predictions.csv`, `failure_analysis.csv`, `detailed_failure_analysis.md`, `training_config.json`, `training_metrics.csv`, `lora_adapter/`
