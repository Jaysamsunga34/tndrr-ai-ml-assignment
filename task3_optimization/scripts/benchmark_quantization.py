#!/usr/bin/env python3
"""
benchmark_quantization.py - Local LLM benchmark comparing Qwen2.5-1.5B-Instruct in Q4_K_M vs Q8_0.
Uses llama.cpp (via llama-cpp-python) with Metal GPU offloading (n_gpu_layers=-1) on Apple M5.
Uses native llama_perf_context() C-API timings to accurately separate:
  1. Prompt processing throughput (prefill tokens/sec)
  2. Token generation throughput (decode tokens/sec)
  3. Total inference latency (s)
  4. Process memory footprint (Unified Memory RSS in MB via psutil)
  5. Model loading time (s)
Runs 3 repetitions per prompt across all 10 standardized prompts and saves full text outputs.
"""

import os
import sys
import gc
import json
import time
import psutil
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import llama_cpp
from llama_cpp import Llama

def get_process_memory_mb() -> float:
    """Measure Resident Set Size (RSS) memory of the current process in MB (Apple Unified Memory)."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_model_benchmark(
    model_path: str,
    quant_label: str,
    prompts: List[Dict[str, Any]],
    n_ctx: int = 2048,
    n_threads: int = 4,
    n_gpu_layers: int = -1,
    repetitions: int = 3
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Benchmarking: {quant_label} ({Path(model_path).name})")
    print(f"=======================================================")

    gc.collect()
    mem_before = get_process_memory_mb()
    t_load_start = time.time()

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=False
    )
    load_time = time.time() - t_load_start
    mem_after_load = get_process_memory_mb()
    model_mem_delta = max(0.0, mem_after_load - mem_before)

    # Warmup call (excluded from benchmark averages)
    _ = llm.create_chat_completion(
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=8,
        temperature=0.0
    )
    mem_after_warmup = get_process_memory_mb()

    print(f"Model loaded in {load_time:.2f} s | RSS after load: {mem_after_load:.1f} MB (delta: {model_mem_delta:.1f} MB)")

    prompt_results = []
    all_gen_speeds = []
    all_prompt_speeds = []
    all_latencies = []
    peak_memory = max(mem_after_load, mem_after_warmup)

    for p_idx, p_item in enumerate(prompts):
        pid = p_item["id"]
        cat = p_item["category"]
        prompt_text = p_item["prompt"]

        rep_records = []
        last_output_text = ""

        for rep in range(repetitions):
            messages = [
                {"role": "system", "content": "You are a helpful and precise assistant."},
                {"role": "user", "content": prompt_text}
            ]

            # Reset KV cache & internal perf timers before each repetition for fair independent measurement
            llm.reset()
            llama_cpp.llama_perf_context_reset(llm._ctx.ctx)

            t0 = time.time()
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.0,
                repeat_penalty=1.05
            )
            total_time = time.time() - t0

            perf = llama_cpp.llama_perf_context(llm._ctx.ctx)
            n_p_eval = int(perf.n_p_eval)
            t_p_eval_ms = float(perf.t_p_eval_ms)
            n_eval = int(perf.n_eval)
            t_eval_ms = float(perf.t_eval_ms)

            usage = response.get("usage", {})
            prompt_tokens = n_p_eval if n_p_eval > 0 else usage.get("prompt_tokens", 0)
            completion_tokens = n_eval if n_eval > 0 else usage.get("completion_tokens", 0)
            output_text = response["choices"][0]["message"]["content"]
            last_output_text = output_text

            prompt_speed = (prompt_tokens / (t_p_eval_ms / 1000.0)) if t_p_eval_ms > 0 else 0.0
            gen_speed = (completion_tokens / (t_eval_ms / 1000.0)) if t_eval_ms > 0 else (completion_tokens / max(0.001, total_time))

            all_gen_speeds.append(gen_speed)
            if prompt_speed > 0:
                all_prompt_speeds.append(prompt_speed)
            all_latencies.append(total_time)

            current_mem = get_process_memory_mb()
            if current_mem > peak_memory:
                peak_memory = current_mem

            rep_records.append({
                "rep": rep + 1,
                "latency_sec": round(total_time, 4),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prompt_eval_ms": round(t_p_eval_ms, 2),
                "gen_eval_ms": round(t_eval_ms, 2),
                "prompt_speed_tok_sec": round(prompt_speed, 2),
                "gen_speed_tok_sec": round(gen_speed, 2)
            })

        avg_latency = float(np.mean([r["latency_sec"] for r in rep_records]))
        std_latency = float(np.std([r["latency_sec"] for r in rep_records]))
        avg_gen_speed = float(np.mean([r["gen_speed_tok_sec"] for r in rep_records]))
        std_gen_speed = float(np.std([r["gen_speed_tok_sec"] for r in rep_records]))
        avg_prompt_speed = float(np.mean([r["prompt_speed_tok_sec"] for r in rep_records]))

        prompt_results.append({
            "prompt_id": pid,
            "category": cat,
            "prompt": prompt_text,
            "output_text": last_output_text,
            "avg_latency_s": round(avg_latency, 4),
            "std_latency_s": round(std_latency, 4),
            "avg_prompt_speed_tok_s": round(avg_prompt_speed, 2),
            "avg_gen_speed_tok_s": round(avg_gen_speed, 2),
            "std_gen_speed_tok_s": round(std_gen_speed, 2),
            "repetitions": rep_records
        })
        print(
            f"[{p_idx+1}/{len(prompts)}] {pid:26s} | Prefill: {avg_prompt_speed:6.1f} tok/s | "
            f"Decode: {avg_gen_speed:5.1f} ± {std_gen_speed:3.1f} tok/s | Lat: {avg_latency:.2f}s"
        )

    overall_metrics = {
        "quantization": quant_label,
        "model_file": Path(model_path).name,
        "model_file_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2),
        "load_time_seconds": round(load_time, 3),
        "memory_metric": "Process RSS (Apple Unified Memory, MB)",
        "rss_before_load_mb": round(mem_before, 2),
        "rss_after_load_mb": round(mem_after_load, 2),
        "model_memory_delta_mb": round(model_mem_delta, 2),
        "peak_memory_mb": round(peak_memory, 2),
        "mean_prompt_tokens_per_sec": round(float(np.mean(all_prompt_speeds)), 2) if all_prompt_speeds else 0.0,
        "std_prompt_tokens_per_sec": round(float(np.std(all_prompt_speeds)), 2) if all_prompt_speeds else 0.0,
        "mean_generation_tokens_per_sec": round(float(np.mean(all_gen_speeds)), 2),
        "std_generation_tokens_per_sec": round(float(np.std(all_gen_speeds)), 2),
        "mean_latency_seconds": round(float(np.mean(all_latencies)), 3),
        "std_latency_seconds": round(float(np.std(all_latencies)), 3),
        "total_prompts_tested": len(prompts),
        "repetitions_per_prompt": repetitions
    }

    del llm
    gc.collect()
    return {
        "metrics": overall_metrics,
        "prompt_results": prompt_results
    }

def main():
    models_dir = Path("/Users/jayadityashah/Downloads/tndrr-ai-ml-assignment/task3_optimization/models")
    prompts_file = Path("/Users/jayadityashah/Downloads/tndrr-ai-ml-assignment/task3_optimization/prompts/benchmark_prompts.json")
    results_dir = Path("/Users/jayadityashah/Downloads/tndrr-ai-ml-assignment/task3_optimization/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    q4_path = models_dir / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    q8_path = models_dir / "qwen2.5-1.5b-instruct-q8_0.gguf"

    if not q4_path.exists() or not q8_path.exists():
        print("Model files not found. Run download_models.py first.")
        return

    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # Benchmark Q4_K_M
    res_q4 = run_model_benchmark(str(q4_path), "Q4_K_M", prompts, repetitions=3)

    # Benchmark Q8_0
    res_q8 = run_model_benchmark(str(q8_path), "Q8_0", prompts, repetitions=3)

    summary_metrics = {
        "hardware": "Apple M5 (arm64, 24 GB Unified Memory, Metal GPU offload)",
        "framework": "llama.cpp via llama-cpp-python 0.3.37 (n_gpu_layers=-1, n_ctx=2048, n_threads=4, temp=0.0)",
        "model_architecture": "Qwen2.5-1.5B-Instruct",
        "Q4_K_M": res_q4["metrics"],
        "Q8_0": res_q8["metrics"]
    }
    with open(results_dir / "benchmark_metrics.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)

    comparison_rows = []
    for p4, p8 in zip(res_q4["prompt_results"], res_q8["prompt_results"]):
        comparison_rows.append({
            "Prompt ID": p4["prompt_id"],
            "Category": p4["category"],
            "Q4 Prefill (tok/s)": p4["avg_prompt_speed_tok_s"],
            "Q8 Prefill (tok/s)": p8["avg_prompt_speed_tok_s"],
            "Q4 Decode (tok/s)": p4["avg_gen_speed_tok_s"],
            "Q8 Decode (tok/s)": p8["avg_gen_speed_tok_s"],
            "Q4 Latency (s)": p4["avg_latency_s"],
            "Q8 Latency (s)": p8["avg_latency_s"],
            "Decode Speedup (Q4/Q8)": round(p4["avg_gen_speed_tok_s"] / max(0.1, p8["avg_gen_speed_tok_s"]), 2)
        })
    df_lat = pd.DataFrame(comparison_rows)
    df_lat.to_csv(results_dir / "latency_comparison.csv", index=False)
    print("\n=== Latency and Throughput Comparison ===")
    print(df_lat.to_string(index=False))

    outputs_comparison = []
    for p4, p8 in zip(res_q4["prompt_results"], res_q8["prompt_results"]):
        outputs_comparison.append({
            "prompt_id": p4["prompt_id"],
            "category": p4["category"],
            "prompt": p4["prompt"],
            "output_q4_k_m": p4["output_text"].strip(),
            "output_q8_0": p8["output_text"].strip()
        })

    with open(results_dir / "outputs_q4_vs_q8.json", "w") as f:
        json.dump(outputs_comparison, f, indent=2)

    print(f"\nSaved full outputs to {results_dir / 'outputs_q4_vs_q8.json'}")
    print(f"Summary Q4_K_M: Decode={res_q4['metrics']['mean_generation_tokens_per_sec']} ± {res_q4['metrics']['std_generation_tokens_per_sec']} tok/s | Prefill={res_q4['metrics']['mean_prompt_tokens_per_sec']} tok/s | Peak RSS={res_q4['metrics']['peak_memory_mb']} MB")
    print(f"Summary Q8_0:   Decode={res_q8['metrics']['mean_generation_tokens_per_sec']} ± {res_q8['metrics']['std_generation_tokens_per_sec']} tok/s | Prefill={res_q8['metrics']['mean_prompt_tokens_per_sec']} tok/s | Peak RSS={res_q8['metrics']['peak_memory_mb']} MB")

if __name__ == "__main__":
    main()
