#!/usr/bin/env python3
"""
train.py - LoRA fine-tuning for Qwen2.5-0.5B-Instruct on customer support tickets to JSON.
Uses PEFT and Hugging Face Transformers with exact specified hyperparameters.
Masks prompt tokens (-100) so cross-entropy loss is computed strictly on the JSON target.
"""

import os
import json
import time
import argparse
import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

SYSTEM_PROMPT = (
    "You are an AI customer support classifier. Classify the customer's support request into a "
    "strictly valid JSON object with exactly three keys:\n"
    '- "category": one of ["billing", "technical", "account", "security", "general"]\n'
    '- "priority": one of ["low", "medium", "high", "critical"]\n'
    '- "intent": concise snake_case intent label\n'
    "Output ONLY valid JSON with no extra text."
)

def prepare_dataset(data_path, tokenizer, max_length=256):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = []
    for item in raw_data:
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Customer Request:\n\"{item['input']}\""}
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        answer_text = json.dumps(item["expected"]) + tokenizer.eos_token

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + answer_ids)[:max_length]
        # Mask prompt tokens with -100 so loss focuses on structured JSON generation
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
        attention_mask = [1] * len(input_ids)

        records.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        })

    return Dataset.from_list(records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_dir", type=str, default="/Users/jayadityashah/Downloads/tndrr-ai-ml-assignment/task1_finetuning/data")
    parser.add_argument("--output_dir", type=str, default="/Users/jayadityashah/Downloads/tndrr-ai-ml-assignment/task1_finetuning/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    adapter_path = output_path / "lora_adapter"

    print(f"Loading tokenizer & model: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True
    )

    # Verify target modules exist (all 4 attention projections for better structured classification)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    for tm in target_modules:
        assert tm in module_names, f"Target module {tm} not found in model!"
    print(f"Verified target modules in model: {target_modules}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none"
    )

    model = get_peft_model(model, peft_config)
    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(f"Trainable params: {trainable_params:,} / {all_param:,} ({100 * trainable_params / all_param:.4f}%)")

    config_record = {
        "model_id": args.model_id,
        "seed": args.seed,
        "device": device,
        "dtype": str(dtype),
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "per_device_batch_size": args.batch_size,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": args.batch_size * 2,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "trainable_params": trainable_params,
        "total_params": all_param,
        "trainable_percent": round(100 * trainable_params / all_param, 4),
        "max_seq_length": 256,
        "loss_masking": "assistant_response_only"
    }
    with open(output_path / "training_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    train_file = Path(args.data_dir) / "train.json"
    val_file = Path(args.data_dir) / "val.json"

    train_dataset = prepare_dataset(train_file, tokenizer)
    val_dataset = prepare_dataset(val_file, tokenizer)
    print(f"Prepared datasets -> Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=15,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=args.seed,
        fp16=False,
        bf16=(device == "cuda" and torch.cuda.is_bf16_supported()),
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", label_pad_token_id=-100)
    )

    print("Starting LoRA fine-tuning...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")

    # Save metrics
    log_history = trainer.state.log_history
    metrics_df = pd.DataFrame(log_history)
    metrics_df.to_csv(output_path / "training_metrics.csv", index=False)
    print(f"Saved training metrics to {output_path / 'training_metrics.csv'}")

    # Update config with runtime
    config_record["training_duration_seconds"] = round(elapsed, 2)
    with open(output_path / "training_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    # Save adapter
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Saved LoRA adapter to {adapter_path}")

if __name__ == "__main__":
    main()
