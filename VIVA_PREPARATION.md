# Viva & Technical Interview Preparation Guide

This document contains 18 likely technical interview questions tied directly to our actual implementation, experimental measurements, and failure analyses across all three assignment tasks.

---

## Task 1: LoRA / QLoRA Fine-Tuning

### 1. Why did you choose LoRA instead of full fine-tuning?
Full fine-tuning updates every weight in the model, requiring gradients (2 B/param) and AdamW FP32 optimizer states (12 B/param: FP32 master weights + 1st and 2nd moments) for every parameter. For a 7B model, static memory alone reaches ~112 GB (~117.5 GB peak VRAM with activations), requiring multi-GPU A100 nodes. LoRA freezes the base weights $W_0$ and learns low-rank update matrices $\Delta W = B A$ (where $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$). In our Task 1 run on `Qwen2.5-0.5B-Instruct`, LoRA updated only **1,081,344 / 495,114,112 parameters (0.2184%)**, cutting optimizer/gradient memory by >99%, training in 3.87 minutes on Apple M5 MPS, and producing a compact, swappable adapter without catastrophic forgetting of base JSON syntax (100% JSON validity preserved).

### 2. Why does QLoRA reduce memory even further than LoRA?
Standard LoRA still stores the frozen base model in 16-bit precision (~14.0 GB for a 7B model). QLoRA compresses the frozen base weights into **4-bit NormalFloat (NF4)**—an information-theoretically optimal quantization data type for normally distributed neural network weights—plus **Double Quantization** (quantizing the block-level quantization constants themselves) and **Paged Optimizers** (offloading optimizer memory spikes to CPU RAM via CUDA unified memory). For a 7B model, this cuts base weight memory from ~14.0 GB to ~3.85 GB, bringing total peak training VRAM down to **~9.5 GB** (fitting on a free 16 GB Colab T4 GPU).

### 3. What do LoRA rank ($r$) and alpha ($\alpha$) mean, and why did you pick $r=8, \alpha=16$?
- **Rank ($r$)** sets the bottleneck dimension of the low-rank decomposition $\Delta W = B A$. Higher $r$ increases expressive capacity and trainable parameters linearly; lower $r$ regularizes against overfitting on narrow tasks.
- **Alpha ($\alpha$)** is a scaling constant applied to the LoRA output via $\frac{\alpha}{r} B A x$. Setting $\alpha = 2r$ (here $\frac{16}{8} = 2$) is standard practice because it keeps the effective update magnitude stable when tuning $r$.
- For our narrow 3-key structured JSON extraction task with 504 training examples, $r=8, \alpha=16$ across `q_proj, k_proj, v_proj, o_proj` provided sufficient capacity to boost Exact Match from 0% to 32.35% and Category Accuracy from 20.59% to 75.00% without memorizing noise.

### 4. Why did you mask prompt tokens with `-100` during training?
In standard causal LM fine-tuning, computing cross-entropy loss over the entire sequence forces the LoRA adapters to spend gradient updates predicting the fixed system prompt and user input text. In `task1_finetuning/scripts/train.py`, we set `labels = [-100] * len(prompt_ids) + answer_ids` so PyTorch ignores prompt tokens and computes gradients strictly on the structured JSON assistant tokens.

### 5. Why did the fine-tuned model improve overall, and why did it still fail on certain test cases?
- **Improvements:** On 68 held-out test examples, Exact JSON Match jumped from **0.00% → 32.35%**, Category Accuracy **20.59% → 75.00%**, Priority Accuracy **51.47% → 77.94%**, and Intent Accuracy **0.00% → 51.47%** (while keeping **100% JSON validity**). The base model could emit valid JSON but didn't know our internal 28-class snake_case intent taxonomy or domain priority rules.
- **Genuine Failures:** We observed four clear failure mechanisms: (1) **Prefix attention shortcut** on negated urgency (`"URGENT PLEASE READ: Actually I'm not in a hurry..."` → predicted `high` instead of `low`); (2) **Implicit criticality** without explicit `"EMERGENCY"` markers (`"actively exporting our entire customer table"` → predicted `medium`); (3) **OOV intent hallucination** (`"2fa_bypass"` instead of `"2fa_trouble"`) under unconstrained decoding; and (4) **Autoregressive key-order inconsistency** (`"category": "technical", "intent": "payment_failed"` because `category` was generated before `intent`).

### 6. How was training/test contamination and template leakage avoided?
In `task1_finetuning/scripts/generate_dataset.py`, we enforced three strict separation barriers:
1. **Disjoint intent descriptions:** Each of the 28 intents had 4 distinct phrasings; descriptions 0–2 went strictly to Train/Val, while description 3 was reserved exclusively for Held-out Test.
2. **Disjoint conversational framing templates:** `TRAIN_FRAMES`, `VAL_FRAMES`, and `TEST_FRAMES` share zero templates, and we removed literal label names (`"billing ticket"`) from inputs after diagnosing template-copying shortcut behavior in our first prototype.
3. **Adversarial hold-out:** 14 completely hand-written natural edge cases appear only in `test.json`, verified programmatically to have zero string overlap with `train.json`.

---

## Task 2: RAG From Scratch

### 7. How does cosine similarity work in your FAISS index?
Cosine similarity measures the angle between two vectors: $\cos(\theta) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$. In `task2_rag/scripts/rag_pipeline.py`, we pass `normalize_embeddings=True` when encoding both document chunks and queries with `all-MiniLM-L6-v2` so every vector has unit L2 norm ($\|u\|_2 = \|v\|_2 = 1$). Once L2-normalized, exact cosine similarity is mathematically identical to the inner product $u \cdot v$, which FAISS computes directly via `faiss.IndexFlatIP(384)`.

### 8. What are Recall@5, Recall@1, and MRR, and how did you define a retrieval hit?
- **Recall@K:** Fraction of test queries where at least one relevant chunk appears within the top-$K$ retrieved results.
- **MRR (Mean Reciprocal Rank):** $\frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$ (0 if not found in top-$K$), which rewards ranking the right passage at rank 1 rather than rank 5.
- **Hit Definition:** A retrieved chunk only counts as a hit if it contains the verbatim gold passage (or $\ge 55\%$ of its distinctive content terms after punctuation normalization) **and** falls within the matching physical page span of `NIST.AI.100-1.pdf`—preventing false positives from merely retrieving an unrelated chunk from the same PDF.

### 9. Why did 256-token chunks outperform 512-token chunks in your experiment?
Config A (256 tokens, 50 overlap) achieved **93.33% Recall@5** and **0.7889 MRR**, outperforming Config B (512 tokens, 50 overlap) at **86.67% Recall@5** and **0.7500 MRR**. Two technical factors drove this:
1. **Hard Encoder Truncation:** `all-MiniLM-L6-v2` has a maximum input window of **256 WordPiece tokens**. Any text in tokens 257–512 of Config B chunks is silently discarded before embedding.
2. **Semantic Dilution:** Larger 512-token chunks merge disparate paragraphs (e.g., merging the brief Figure 2 OECD caption on Page 15 with unrelated civil-society prose), averaging down the cosine similarity for targeted factual queries (`q13`).

### 10. Explain your two genuine retrieval failures and how you would fix them.
- **Failure 1 (`q05`, failed in both 256 & 512):** The lead-in phrase *"characteristics of trustworthy AI:"* sat at the bottom of Page 7 while the actual enumerated list started at the top of Page 8. Page-isolated extraction severed the heading from its list, so dense retrieval returned generic pages repeating `"trustworthy AI"` (Section 3 intro, List of Figures). *Fix:* Continuous cross-page chunking with character-to-page span mapping + prepending section headers (Contextual Retrieval).
- **Failure 2 (`q13`, Rank 2 in 256 but failed in 512):** Asking for the 2022 OECD citation in Figure 2 failed in Config B due to 256-token encoder truncation and semantic dilution of exact entities (`OECD`, `2022`, `Fig. 2`). *Fix:* Hybrid BM25 + dense retrieval (RRF) and aligning chunk size to the encoder's token limit.

### 11. What does BM25 add beyond dense vector retrieval?
Dense embeddings capture paraphrases and conceptual similarity well, but compress text into a fixed 384-dim vector that blurs rare exact identifiers—acronyms (`TEVV`), subcategory IDs (`MEASURE 2.6`, `MANAGE 4.1`), standard codes (`ISO 26000:2010`), and citations (`OECD 2022`). BM25 is a sparse lexical scorer combining saturating Term Frequency, Inverse Document Frequency (IDF), and document-length normalization. Rare exact terms carry high IDF weight, complementing dense recall via Hybrid Search (Reciprocal Rank Fusion).

### 12. What does a Cross-Encoder Reranker do?
Bi-encoders encode query and chunk separately (fast dot product, no cross-token attention). A **Cross-Encoder Reranker** takes the top-15 to top-20 retrieved candidates and processes `[CLS] query [SEP] chunk [SEP]` jointly through full self-attention layers. This fine-grained interaction filters out misleading topical overlaps (like List-of-Figures entries or boilerplate intros), boosting Recall@1 and MRR with minimal added latency.

---

## Task 3: Local LLM Inference & Optimization

### 13. What is model quantization, and how do Q4_K_M and Q8_0 differ?
Quantization maps 16-bit floating-point weights to lower-bit discrete levels using block-wise scaling factors. **Q8_0** stores weights at ~8.5 bits/weight (1.06 B/param), preserving near-lossless FP16 quality. **Q4_K_M** uses llama.cpp's K-quant mixed-precision scheme (~4.8 bits/weight, 0.60 B/param): most weights use 4-bit super-blocks while sensitive layers (attention output / feed-forward down-projections / embeddings) retain 6-bit precision. On `Qwen2.5-1.5B-Instruct`, Q4_K_M shrank file size by **41%** (1,806 MB → 1,066 MB) and runtime RSS memory by **34.5%** (2,141 MB → 1,403 MB).

### 14. Why did Q4_K_M speed up decode by 1.50× while prefill speed stayed almost identical?
This demonstrates the **hardware roofline model** directly on Apple M5 Metal:
- **Decode (Token Generation)** runs at batch size 1 per step, executing matrix-vector products (`GEMV`). It is strictly **memory-bandwidth bound**—every generated token streams the entire model's weights from Unified Memory. Reducing weight bytes from Q8_0 to Q4_K_M cut memory traffic directly, boosting decode from **71.83 tok/s to 107.78 tok/s (1.50×)**.
- **Prefill (Prompt Processing)** processes all prompt tokens in parallel via dense matrix-matrix products (`GEMM`), which is **compute-bound**. Consequently, prefill was nearly identical (**1,707.4 tok/s vs. 1,736.4 tok/s**), with Q8_0 slightly ahead by 1.7% due to lower dequantization overhead.

### 15. Why might Q4_K_M and Q8_0 outputs differ—and what did you actually observe?
Quantization introduces small weight rounding noise that slightly perturbs output token logits. Even at `temperature=0.0`, when two candidate tokens have close probabilities, a tiny logit shift flips the argmax token and diverges the remaining autoregressive trajectory. Across our 10 prompts: structured JSON (`p03`, `p07`) was **byte-for-byte identical**, `Q8_0` followed conciseness/code cleanliness better on `p01` and `p06`, `Q4_K_M` actually obeyed the 1-to-1 pairing instruction better on `p08`, and both quantizations failed the Wolf-Goat-Cabbage state-tracking puzzle (`p02`)—proving that core reasoning limits stemmed from 1.5B model capacity rather than 4-bit quantization.

### 16. What is the KV cache, and why does its memory scale with context and concurrency?
During causal autoregressive decoding, past tokens' Key and Value vectors never change. Caching them in memory avoids recomputing all previous tokens' projections at every step ($O(T^2)$ → $O(1)$ projection + cached attention per step). Its memory scales linearly with **both** sequence length $S$ and active batch size $B$:
$$M_{\text{KV}} = 2 \times L \times n_{\text{kv}} \times d_h \times S \times B \times p$$
Grouped-Query Attention (GQA, where $n_{\text{kv}} \ll n_{\text{heads}}$) is critical for serving concurrency: `Qwen2.5-7B` ($n_{\text{kv}}=4$) requires only **56 KB/token** in FP16, whereas an older MHA 7B model like Llama-2-7B ($n_{\text{kv}}=32$) requires **512 KB/token** (9.1× larger).

### 17. Can a single 24 GB GPU serve 20 simultaneous users, and what limits latency?
It depends critically on **architecture (GQA vs. MHA)** and **quantization**:
- **Qwen2.5-7B (GQA) in 4-bit (AWQ/Q4_K_M)** at 4,096 context fits comfortably: ~4.6 GB weights + ~4.48 GB FP16 KV cache (20 users × 4K × 56 KB) + ~2.2 GB runtime overhead = **~11.3 GB** (12.7 GB headroom).
- **Qwen2.5-7B in FP16** is marginal (**~21.9 GB**), risking preemption/OOM under prefill bursts.
- **Llama-2-7B (MHA) in 4-bit** fails outright (**~46.4 GB** peak due to 40 GB KV cache).
- **Latency caveat:** Fitting 20 users in VRAM does *not* guarantee low latency. Continuous batching improves aggregate tokens/sec, but per-user Inter-Token Latency (ITL) degrades as 20 concurrent KV caches saturate GPU memory bandwidth, and long incoming prompts cause Time-To-First-Token (TTFT) spikes unless mitigated with **chunked prefill**, **FP8 KV cache**, **prefix caching**, and `max_num_seqs` admission control.

### 18. What are the main limitations of these three experiments and your next engineering steps?
1. **Task 1:** Synthetic dataset lacks real-world typos/multi-turn ambiguity; next steps are curating real anonymized tickets, adding GBNF/JSON-Schema constrained decoding over the 28 canonical intents, and reordering JSON keys (`intent` before `category`).
2. **Task 2:** Single 48-page PDF and 15 questions (6.67% granularity per question) using dense-only retrieval; next steps are continuous cross-page chunking, Hybrid BM25+FAISS (RRF), and Cross-Encoder reranking evaluated over a larger multi-document benchmark.
3. **Task 3:** Benchmarked single-stream local inference on Apple Silicon Unified Memory (RSS proxy) rather than live multi-user discrete GPU load testing; next step is running `vllm bench serve` under Poisson arrival rates on a 24 GB L4/A10G to measure empirical P95 TTFT and ITL curves.
