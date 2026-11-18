# Task 3: Theoretical Foundations & 24 GB GPU Capacity Planning

---

## 1. KV Cache Mechanics in Autoregressive Inference

### What the KV Cache Stores
In a Transformer decoder, self-attention computes projections for every token position $t$:
$$Q_t = x_t W_Q, \quad K_t = x_t W_K, \quad V_t = x_t W_V$$
During autoregressive generation, predicting token $t+1$ requires computing attention scores between the new query vector $Q_t$ and the key/value vectors of **all preceding tokens** $K_{1:t}, V_{1:t}$ at every transformer layer $l \in \{1, \dots, L\}$. Because causal masking guarantees that keys and values of past tokens never change when future tokens are appended, the **KV cache** stores these computed $(K, V)$ tensors in VRAM/RAM for every layer and every active sequence.

### Why It Speeds Up Autoregressive Inference
- **Without a KV cache:** Generating the $t$-th token requires recomputing $K_1 \dots K_{t-1}$ and $V_1 \dots V_{t-1}$ from scratch, making total generation complexity $O(T^2)$ in FLOPs for a sequence of length $T$ (with $O(T^3)$ attention work overall).
- **With a KV cache:** Decoding is divided into two distinct phases:
  1. **Prefill (Prompt Processing):** Computes and caches $K, V$ for all prompt tokens in parallel (compute-bound, high matrix-matrix GEMM utilization).
  2. **Decode (Token Generation):** Computes only the single new token's $Q_t, K_t, V_t$, appends $(K_t, V_t)$ to the cache, and attends over the cached history (memory-bandwidth bound). This reduces per-token work from $O(t)$ forward passes to $O(1)$ projection plus $O(t)$ attention lookups.

### Why KV Cache Memory Scales with Context Length and Concurrent Sequences
For a Transformer using Grouped-Query Attention (GQA) with $L$ layers, $n_{\text{kv}}$ KV heads, head dimension $d_h$, precision $p$ bytes per element (e.g., 2 bytes for FP16/BF16), sequence length $S$, and $B$ concurrent active requests, the KV cache footprint is:
$$M_{\text{KV}} = 2 \times L \times n_{\text{kv}} \times d_h \times S \times B \times p$$
*(The factor of $2$ accounts for storing both Key and Value tensors.)*
Because each active request maintains its own independent conversation history, KV cache memory scales **linearly with both sequence length $S$ and batch concurrency $B$**. At long context windows or high concurrency, KV cache easily surpasses the size of the model weights themselves.

---

## 2. Weight Quantization & Engineering Trade-Offs

### How Quantization Saves Model Memory
Post-Training Quantization (PTQ) maps high-precision floating-point weights (FP16/BF16, 16 bits/weight) to lower-bit integer or non-uniform discrete representations (e.g., 8-bit `Q8_0` or 4-bit `Q4_K_M`). Block-wise quantization partitions weights into small groups (e.g., blocks of 32 or 256 weights) and stores a shared high-precision scaling factor (and optionally zero-point/min) per block:
- **FP16:** 16.0 bits/weight (2.00 bytes/param)
- **Q8_0:** ~8.5 bits/weight (~1.06 bytes/param) — $\approx 47\%$ memory reduction vs. FP16
- **Q4_K_M:** ~4.8 bits/weight (~0.60 bytes/param, mixed 4-bit/6-bit super-blocks protecting sensitive attention/output layers) — $\approx 70\%$ memory reduction vs. FP16

### Trade-Offs Between Quantization, Quality, Throughput, and Memory
1. **Throughput (Memory Bandwidth Bottleneck):** Single-sequence autoregressive decoding is heavily **memory-bandwidth bound**—at every generated token, the GPU/CPU must stream all model weights from HBM/RAM into compute registers. Halving weight size from Q8_0 to Q4_K_M roughly cuts memory traffic by $\sim 40\text{–}45\%$, directly translating to faster tokens/sec despite minor dequantization compute overhead.
2. **Output Quality & Perplexity:** Moving from FP16 to Q8_0 introduces virtually zero perplexity degradation ($<0.05\%$). Moving to Q4_K_M preserves nearly all instruction-following and factual quality on medium/large models, though smaller models ($\le 1.5\text{B}$) can exhibit slightly higher sensitivity on strict formatting constraints, multi-step arithmetic, or rare factual recall.
3. **Batch/Concurrency Headroom:** Lower weight footprint frees precious VRAM for larger KV caches, enabling higher concurrent request capacity on fixed hardware.

---

## 3. Capacity Planning: Serving 20 Simultaneous Users on a 24 GB GPU

### Workload & System Assumptions
- **Target Hardware:** Single 24 GB GPU (e.g., NVIDIA RTX 4090 / A10G / L4, ~24.0 GB VRAM, ~500–1008 GB/s memory bandwidth).
- **Serving Engine:** vLLM or llama.cpp server with **PagedAttention** and **Continuous (Iteration-Level) Batching**.
- **Concurrent Users ($B$):** 20 simultaneous active requests.
- **Max Context Length ($S_{\max}$):** 4,096 tokens per user (e.g., ~3,000 prompt tokens + ~1,000 generated tokens).
- **Typical Active Context ($S_{\text{avg}}$):** 2,048 tokens average across active requests.
- **KV Cache Precision ($p$):** FP16 (2 bytes/element), with optional FP8 (1 byte/element) fallback.
- **CUDA Runtime, PyTorch/vLLM Engine & Activation Overhead:** $\approx 2.2\text{ GB}$.
- **Available VRAM for Weights + KV Cache:** $24.0 - 2.2 = \mathbf{21.8\text{ GB}}$.

### Scenario A: 7B–8B Model with GQA (e.g., Llama-3-8B or Qwen2.5-7B-Instruct) in 4-bit (AWQ / Q4_K_M)
Using **Qwen2.5-7B-Instruct** architecture ($L=28$ layers, $n_{\text{kv}}=4$ GQA KV heads, $d_h=128$):
1. **Model Weights (4-bit Q4_K_M / AWQ):**
   $$M_{\text{weights}} \approx 7.6\text{B} \times 0.60\text{ bytes} \approx \mathbf{4.6\text{ GB}}$$
2. **KV Cache per Token (FP16):**
   $$M_{\text{KV/tok}} = 2 \times 28 \times 4 \times 128 \times 2\text{ bytes} = 57,344\text{ bytes} \approx 56\text{ KB/token}$$
3. **KV Cache for 20 Simultaneous Users at Full 4,096 Context:**
   $$M_{\text{KV, worst-case}} = 20 \times 4,096 \times 56\text{ KB} \approx \mathbf{4.48\text{ GB}}$$
   *(At average 2,048 context: $\approx 2.24\text{ GB}$.)*
4. **Total Peak VRAM Estimate:**
   $$M_{\text{total}} = 4.6\text{ GB (weights)} + 4.5\text{ GB (KV cache)} + 2.2\text{ GB (runtime/activations)} = \mathbf{11.3\text{ GB}}$$
- **Verdict (Scenario A):** **Fits comfortably** inside 24 GB VRAM (~12.7 GB headroom remaining), even if all 20 users simultaneously reach 4,096 tokens.

### Scenario B: 7B–8B Model in FP16 (Unquantized)
1. **Model Weights (FP16):** $7.6\text{B} \times 2\text{ bytes} \approx \mathbf{15.2\text{ GB}}$
2. **KV Cache (20 users $\times$ 4,096 tokens, FP16):** $\mathbf{4.48\text{ GB}}$
3. **Runtime & Activation Overhead:** $\mathbf{2.2\text{ GB}}$
4. **Total Peak VRAM:** $15.2 + 4.48 + 2.2 = \mathbf{21.88\text{ GB}}$
- **Verdict (Scenario B):** **Tight / Marginal fit.** Leaves $<2.1\text{ GB}$ headroom; any spike in prompt prefill activations (e.g., simultaneous 4K prefills) or requests exceeding 4K tokens will trigger request preemption/queueing or OOM errors.

### Scenario C: Older MHA 7B Model (No GQA, e.g., Llama-2-7B with $L=32, n_{\text{kv}}=32, d_h=128$) in 4-bit
1. **KV Cache per Token (FP16):** $2 \times 32 \times 32 \times 128 \times 2 = 524,288\text{ bytes} = 512\text{ KB/token}$ (9$\times$ larger than Qwen2.5-7B GQA!).
2. **KV Cache for 20 users $\times$ 4,096 tokens:** $20 \times 4,096 \times 512\text{ KB} = \mathbf{40.0\text{ GB}}$!
- **Verdict (Scenario C):** **Cannot fit on 24 GB GPU.** Even with 4-bit weights (~4.2 GB), the KV cache alone overflows 24 GB by over 20 GB.

### Throughput, Latency Trade-Offs, and Mitigation Levers
Fitting 20 concurrent requests into VRAM **does not guarantee low per-user latency**:
- **Inter-Token Latency (ITL) vs. System Throughput:** Continuous batching amortizes weight-loading memory bandwidth across all 20 requests, increasing *aggregate* system throughput (total tokens/sec across all users) substantially. However, as batch size increases, each decode iteration must read $20\times$ more KV cache and perform larger attention matrix-vector operations, so *per-user* token generation speed degrades and **Time-To-First-Token (TTFT)** spikes whenever long prompts enter prefill.
- **Required Changes if Workload Exceeds 24 GB (e.g., 14B+ model, 8K+ context, or MHA model):**
  1. **Use GQA/MQA architectures** (e.g., Qwen2.5, Llama-3, Mistral) to cut KV cache by $4\times\text{–}8\times$.
  2. **Quantize KV Cache to FP8 / INT8** (cuts KV memory by 50% with minimal accuracy impact).
  3. **Enable Prefix Caching** (shared system prompts stored once in VRAM rather than duplicated $20\times$).
  4. **Cap max context / max concurrent sequences** (`max_num_seqs`) with request queuing and chunked prefill to prevent TTFT stalls.
  5. **Scale horizontally or use Tensor Parallelism** across $2\times$ 24 GB GPUs if serving 14B–32B models at 20 concurrency.
