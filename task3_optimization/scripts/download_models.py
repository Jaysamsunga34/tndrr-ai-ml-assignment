#!/usr/bin/env python3
"""
download_models.py - Download Qwen2.5-1.5B-Instruct GGUFs (Q4_K_M and Q8_0)
for Task 3 local inference quantization benchmark.
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS_DIR = Path("/Users/jayadityashah/Downloads/tndrr-ai-ml-assignment/task3_optimization/models")
REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"

FILES = [
    "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-1.5b-instruct-q8_0.gguf"
]

def download_quantized_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_paths = {}

    for filename in FILES:
        target_path = MODELS_DIR / filename
        if target_path.exists() and target_path.stat().st_size > 100_000_000:
            print(f"File already exists: {target_path} ({target_path.stat().st_size / 1e6:.1f} MB)")
            downloaded_paths[filename] = str(target_path)
            continue

        print(f"Downloading {filename} from {REPO_ID}...")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False
        )
        print(f"Downloaded: {downloaded}")
        downloaded_paths[filename] = downloaded

    return downloaded_paths

if __name__ == "__main__":
    download_quantized_models()
