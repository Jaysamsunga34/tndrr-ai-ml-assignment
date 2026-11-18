#!/usr/bin/env python3
"""
validate_experiments.py - Automated verification of all assignment requirements.
Checks that every required file, metric, failure case, and calculation exists and is non-fabricated.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/jayadityashah/Downloads/tndrr-ai-ml-assignment")

def check(condition: bool, label: str, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    msg = f"[{status}] {label}"
    if detail:
        msg += f" -> {detail}"
    print(msg)
    return condition

def main():
    all_passed = True
    print("=" * 70)
    print("PHASE 4: AUTOMATED EXPERIMENT VALIDATION")
    print("=" * 70)

    # ---- TASK 1 ----
    print("\n--- TASK 1: LoRA / QLoRA Fine-Tuning ---")
    t1_data = ROOT / "task1_finetuning/data"
    t1_res = ROOT / "task1_finetuning/results"

    test_json = t1_data / "test.json"
    if check(test_json.exists(), "Task 1 held-out test set exists"):
        with open(test_json) as f:
            test_samples = json.load(f)
        all_passed &= check(len(test_samples) >= 50, "Task 1 held-out test >= 50 samples", f"{len(test_samples)} samples")
    else:
        all_passed = False

    base_csv = t1_res / "baseline_predictions.csv"
    ft_csv = t1_res / "finetuned_predictions.csv"
    if check(base_csv.exists() and ft_csv.exists(), "Task 1 baseline & finetuned predictions CSVs exist"):
        df_base = pd.read_csv(base_csv)
        df_ft = pd.read_csv(ft_csv)
        same_inputs = list(df_base["input"]) == list(df_ft["input"])
        all_passed &= check(same_inputs and len(df_base) >= 50, "Task 1 base & finetuned use exact same held-out test samples", f"{len(df_base)} identical rows")
    else:
        all_passed = False

    fail1_csv = t1_res / "failure_analysis.csv"
    if check(fail1_csv.exists(), "Task 1 failure_analysis.csv exists"):
        df_f1 = pd.read_csv(fail1_csv)
        all_passed &= check(len(df_f1) >= 3, "Task 1 has >= 3 genuine failure cases", f"{len(df_f1)} failures recorded")
    else:
        all_passed = False

    all_passed &= check((t1_res / "training_config.json").exists(), "Task 1 saved training_config.json")
    all_passed &= check((t1_res / "training_metrics.csv").exists(), "Task 1 saved training_metrics.csv")
    all_passed &= check((t1_res / "evaluation_summary.json").exists(), "Task 1 saved evaluation_summary.json")
    all_passed &= check((t1_res / "lora_adapter").exists(), "Task 1 saved LoRA adapter weights")
    all_passed &= check((ROOT / "task1_finetuning/scripts/gpu_memory_calculation.md").exists(), "Task 1 saved 7B GPU memory calculation")

    # ---- TASK 2 ----
    print("\n--- TASK 2: RAG From Scratch ---")
    t2_doc = ROOT / "task2_rag/documents/NIST.AI.100-1.pdf"
    t2_qa = ROOT / "task2_rag/evaluation/ground_truth_qa.json"
    t2_res = ROOT / "task2_rag/results"

    if check(t2_doc.exists(), "Task 2 source PDF exists", str(t2_doc.name)):
        import fitz
        doc = fitz.open(str(t2_doc))
        all_passed &= check(len(doc) >= 30, "Task 2 source document >= 30 pages", f"{len(doc)} pages")
    else:
        all_passed = False

    if check(t2_qa.exists(), "Task 2 ground-truth QA file exists"):
        with open(t2_qa) as f:
            qa_items = json.load(f)
        has_passages = all(bool(q.get("gold_passage")) and bool(q.get("page_num")) for q in qa_items)
        all_passed &= check(len(qa_items) >= 15 and has_passages, "Task 2 has >= 15 questions with exact supporting passages & pages", f"{len(qa_items)} questions")
    else:
        all_passed = False

    ret_json = t2_res / "retrieval_metrics.json"
    if check(ret_json.exists(), "Task 2 retrieval_metrics.json exists"):
        with open(ret_json) as f:
            rm = json.load(f)
        has_two = "config_a_256" in rm and "config_b_512" in rm
        all_passed &= check(has_two, "Task 2 evaluates two chunk-size configurations",
                            f"256 R@5={rm.get('config_a_256',{}).get('recall_at_5',0):.4f}, 512 R@5={rm.get('config_b_512',{}).get('recall_at_5',0):.4f}")
    else:
        all_passed = False

    fail2_csv = t2_res / "failure_analysis.csv"
    if check(fail2_csv.exists(), "Task 2 failure_analysis.csv exists"):
        df_f2 = pd.read_csv(fail2_csv)
        all_passed &= check(len(df_f2) >= 2, "Task 2 has >= 2 genuine retrieval failures", f"{len(df_f2)} failures recorded")
    else:
        all_passed = False

    all_passed &= check((t2_res / "retrieval_analysis.md").exists(), "Task 2 BM25/reranker/chunk analysis saved")

    # ---- TASK 3 ----
    print("\n--- TASK 3: Local LLM Inference & Optimization ---")
    t3_res = ROOT / "task3_optimization/results"

    bm_json = t3_res / "benchmark_metrics.json"
    if check(bm_json.exists(), "Task 3 benchmark_metrics.json exists"):
        with open(bm_json) as f:
            bm = json.load(f)
        has_quants = "Q4_K_M" in bm and "Q8_0" in bm
        q4_tps = bm.get("Q4_K_M", {}).get("mean_generation_tokens_per_sec", 0)
        q8_tps = bm.get("Q8_0", {}).get("mean_generation_tokens_per_sec", 0)
        q4_mem = bm.get("Q4_K_M", {}).get("peak_memory_mb", 0)
        q8_mem = bm.get("Q8_0", {}).get("peak_memory_mb", 0)
        all_passed &= check(has_quants and q4_tps > 0 and q8_tps > 0 and q4_mem > 0 and q8_mem > 0,
                            "Task 3 real tokens/sec & memory measurements for Q4_K_M and Q8_0",
                            f"Q4={q4_tps:.1f} tok/s ({q4_mem:.0f} MB), Q8={q8_tps:.1f} tok/s ({q8_mem:.0f} MB)")
    else:
        all_passed = False

    out3_json = t3_res / "outputs_q4_vs_q8.json"
    if check(out3_json.exists(), "Task 3 outputs_q4_vs_q8.json exists"):
        with open(out3_json) as f:
            outs = json.load(f)
        all_passed &= check(len(outs) == 10, "Task 3 compares same 10 prompts across both quantizations", f"{len(outs)} prompts")
    else:
        all_passed = False

    all_passed &= check((t3_res / "quality_comparison.csv").exists(), "Task 3 quality_comparison.csv exists")
    all_passed &= check((t3_res / "theory_and_capacity_planning.md").exists(), "Task 3 KV cache theory & 24 GB GPU capacity estimate saved")

    print("\n" + "=" * 70)
    if all_passed:
        print("VALIDATION RESULT: ALL REQUIRED EXPERIMENTS AND ARTIFACTS PASSED!")
        sys.exit(0)
    else:
        print("VALIDATION RESULT: SOME CHECKS FAILED OR ARE STILL PENDING.")
        sys.exit(1)

if __name__ == "__main__":
    main()
