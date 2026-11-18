#!/usr/bin/env python3
"""
download_doc.py - Download authoritative NIST AI Risk Management Framework document.
NIST AI 100-1 (AI RMF 1.0) is a 48-page comprehensive document published by NIST in January 2023.
"""

import os
import requests
from pathlib import Path

DOC_URL = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"
DOC_DIR = Path("/Users/jayadityashah/Downloads/tndrr-ai-ml-assignment/task2_rag/documents")
DEST_FILE = DOC_DIR / "NIST.AI.100-1.pdf"

def download_nist_doc():
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    if DEST_FILE.exists() and DEST_FILE.stat().st_size > 100000:
        print(f"Document already exists at {DEST_FILE} ({DEST_FILE.stat().st_size} bytes)")
        return str(DEST_FILE)

    print(f"Downloading NIST AI RMF 1.0 from {DOC_URL}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    response = requests.get(DOC_URL, headers=headers, timeout=60, stream=True)
    response.raise_for_status()

    with open(DEST_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Downloaded successfully: {DEST_FILE} ({DEST_FILE.stat().st_size} bytes)")
    return str(DEST_FILE)

if __name__ == "__main__":
    download_nist_doc()
