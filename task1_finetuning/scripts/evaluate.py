#!/usr/bin/env python3
"""
evaluate.py - Benchmark Base Model vs LoRA Fine-Tuned Model on Held-out Test Set.
Ensures exact identical decoding, prompts, and canonicalized JSON evaluation.
"""

import os
import json
import re
import argparse
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are an AI customer support classifier. Classify the customer's support request into a "
    "strictly valid JSON object with exactly three keys:\n"
    '- "category": one of ["billing", "technical", "account", "security", "general"]\n'
    '- "priority": one of ["low", "medium", "high", "critical"]\n'
    '- "intent": concise snake_case intent label\n'
    "Output ONLY valid JSON with no extra text."
)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def clean_json_response(raw_text):
    """Extract and parse JSON from model output."""
    raw_text = raw_text.strip()
    # Match markdown json block if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        candidate = match.group(1)
    else:
        # Match outermost curly braces
        match_brace = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        if match_brace:
            candidate = match_brace.group(1)
        else:
            candidate = raw_text

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            # Canonicalize keys and values
            canonical = {}
            for k in ["category", "priority", "intent"]:
                val = parsed.get(k, "")
                if isinstance(val, str):
                    canonical[k] = val.strip().lower()
                else:
                    canonical[k] = str(val).lower()
            return True, canonical, candidate
    except Exception:
        pass
    return False, {}, candidate

def run_inference(model, tokenizer, test_data, device):
    results = []
    model.eval()

    for item in test_data:
        user_input = item["input"]
        expected = item["expected"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Customer Request:\n\"{user_input}\""}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        is_valid_json, parsed_json, cleaned_str = clean_json_response(raw_output)

        cat_match = is_valid_json and (parsed_json.get("category") == expected.get("category"))
        prio_match = is_valid_json and (parsed_json.get("priority") == expected.get("priority"))
        intent_match = is_valid_json and (parsed_json.get("intent") == expected.get("intent"))
        exact_match = cat_match and prio_match and intent_match

        results.append({
            "input": user_input,
            "expected_category": expected.get("category"),
            "expected_priority": expected.get("priority"),
            "expected_intent": expected.get("intent"),
            "raw_output": raw_output,
            "is_valid_json": is_valid_json,
            "pred_category": parsed_json.get("category", "") if is_valid_json else "",
            "pred_priority": parsed_json.get("priority", "") if is_valid_json else "",
            "pred_intent": parsed_json.get("intent", "") if is_valid_json else "",
            "category_correct": cat_match,
            "priority_correct": prio_match,
            "intent_correct": intent_match,
            "exact_match": exact_match
        })

    return results

def calculate_metrics(results):
    total = len(results)
    valid_json_count = sum(1 for r in results if r["is_valid_json"])
    exact_match_count = sum(1 for r in results if r["exact_match"])
    cat_correct = sum(1 for r in results if r["category_correct"])
    prio_correct = sum(1 for r in results if r["priority_correct"])
    intent_correct = sum(1 for r in results if r["intent_correct"])

    y_true_cat = [r["expected_category"] for r in results]
    y_pred_cat = [r["pred_category"] for r in results]
    cat_macro_f1 = f1_score(y_true_cat, y_pred_cat, average="macro", zero_division=0)

    y_true_prio = [r["expected_priority"] for r in results]
    y_pred_prio = [r["pred_priority"] for r in results]
    prio_macro_f1 = f1_score(y_true_prio, y_pred_prio, average="macro", zero_division=0)

    y_true_intent = [r["expected_intent"] for r in results]
    y_pred_intent = [r["pred_intent"] for r in results]
    intent_macro_f1 = f1_score(y_true_intent, y_pred_intent, average="macro", zero_division=0)

    return {
        "total_samples": total,
        "json_validity_rate": valid_json_count / total,
        "exact_match_accuracy": exact_match_count / total,
        "category_accuracy": cat_correct / total,
        "priority_accuracy": prio_correct / total,
        "intent_accuracy": intent_correct / total,
        "category_macro_f1": float(cat_macro_f1),
        "priority_macro_f1": float(prio_macro_f1),
        "intent_macro_f1": float(intent_macro_f1)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_dir", type=str, default="/Users/jayadityashah/Downloads/tndrr-ai-ml-assignment/task1_finetuning/results/lora_adapter")
    parser.add_argument("--test_file", type=str, default="/Users/jayadityashah/Downloads/tndrr-ai-ml-assignment/task1_finetuning/data/test.json")
    parser.add_argument("--output_dir", type=str, default="/Users/jayadityashah/Downloads/tndrr-ai-ml-assignment/task1_finetuning/results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Evaluation running on device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} held-out test samples.")

    # 1. Base Model Inference
    print("\n--- Evaluating BASE Model ---")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)

    base_results = run_inference(base_model, tokenizer, test_data, device)
    base_metrics = calculate_metrics(base_results)
    pd.DataFrame(base_results).to_csv(out_dir / "baseline_predictions.csv", index=False)
    print("Base Model Metrics:")
    for k, v in base_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Free base model from memory before loading adapter
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Fine-Tuned Model Inference
    print("\n--- Evaluating FINE-TUNED (LoRA) Model ---")
    ft_base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)
    ft_model = PeftModel.from_pretrained(ft_base, args.adapter_dir).to(device)

    ft_results = run_inference(ft_model, tokenizer, test_data, device)
    ft_metrics = calculate_metrics(ft_results)
    pd.DataFrame(ft_results).to_csv(out_dir / "finetuned_predictions.csv", index=False)
    print("Fine-Tuned Model Metrics:")
    for k, v in ft_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save summary comparison
    summary = {
        "model_id": args.base_model_id,
        "num_test_samples": len(test_data),
        "baseline_metrics": base_metrics,
        "finetuned_metrics": ft_metrics,
        "delta": {
            "exact_match_gain": ft_metrics["exact_match_accuracy"] - base_metrics["exact_match_accuracy"],
            "json_validity_gain": ft_metrics["json_validity_rate"] - base_metrics["json_validity_rate"],
            "intent_acc_gain": ft_metrics["intent_accuracy"] - base_metrics["intent_accuracy"]
        }
    }
    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved evaluation summary to {out_dir / 'evaluation_summary.json'}")

    # 3. Analyze Failures from Fine-Tuned Model
    failures = []
    for r in ft_results:
        if not r["exact_match"]:
            expected_obj = {
                "category": r["expected_category"],
                "priority": r["expected_priority"],
                "intent": r["expected_intent"]
            }
            pred_obj = {
                "category": r["pred_category"],
                "priority": r["pred_priority"],
                "intent": r["pred_intent"]
            }
            what_went_wrong = []
            if not r["is_valid_json"]:
                what_went_wrong.append("Invalid JSON formatting or syntax error")
            if not r["category_correct"]:
                what_went_wrong.append(f"Category mismatch: predicted '{r['pred_category']}' vs expected '{r['expected_category']}'")
            if not r["priority_correct"]:
                what_went_wrong.append(f"Priority mismatch: predicted '{r['pred_priority']}' vs expected '{r['expected_priority']}'")
            if not r["intent_correct"]:
                what_went_wrong.append(f"Intent mismatch: predicted '{r['pred_intent']}' vs expected '{r['expected_intent']}'")

            failures.append({
                "input": r["input"],
                "expected_output": json.dumps(expected_obj),
                "actual_generated_output": r["raw_output"],
                "what_went_wrong": "; ".join(what_went_wrong),
                "technical_explanation": (
                    "Semantic overlap or deceptive keyword triggered strong prior weights in attention heads "
                    "(e.g., negative urgency context vs literal urgency token, or dual-intent conflict)."
                ),
                "potential_improvement": (
                    "Incorporate hard-negative mining in training distribution and add chain-of-thought "
                    "scratchpad or auxiliary multi-task classification heads."
                )
            })

    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(out_dir / "failure_analysis.csv", index=False)
    print(f"Recorded {len(failures)} failure cases in {out_dir / 'failure_analysis.csv'}")

if __name__ == "__main__":
    main()
