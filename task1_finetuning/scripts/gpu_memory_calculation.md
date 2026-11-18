# GPU Memory Calculation: 7B Model Fine-Tuning Analysis

This document provides a transparent, first-principles back-of-the-envelope memory calculation comparing **Full Fine-Tuning**, **LoRA**, and **QLoRA** for a standard 7-billion parameter language model (e.g., Llama 2/3 7B or Mistral 7B).

---

## 1. Baseline Assumptions and Workload Specification

- **Model Parameters ($N$):** $7.0 \times 10^9$ parameters (7 billion)
- **Precision of Base Model:**
  - Full Fine-Tuning: 16-bit (bfloat16/float16), 2 bytes/param
  - LoRA: 16-bit (bfloat16/float16), 2 bytes/param
  - QLoRA: 4-bit NormalFloat (NF4), 0.5 bytes/param + quantization block overhead (~0.05 bytes/param) $\approx 0.55\text{ bytes/param}$
- **Optimizer:** Standard AdamW (maintains FP32 master weights, 1st momentum $m_t$, 2nd momentum $v_t$)
  - FP32 copy of parameters: 4 bytes/param
  - First momentum ($m_t$, FP32): 4 bytes/param
  - Second momentum ($v_t$, FP32): 4 bytes/param
  - Total AdamW state per trainable parameter = 12 bytes/param (or 16 bytes/param if tracking FP32 gradients separately)
  - In QLoRA, paged AdamW 8-bit can reduce optimizer memory to 2 bytes/param, but we present standard 32-bit AdamW for conservative estimation and also report 8-bit optimizer savings.
- **LoRA Hyperparameters:**
  - Target modules: All linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
  - LoRA Rank ($r$): 16
  - Hidden dimension ($d$): 4096, Intermediate dimension ($d_{ffn}$): 11008, Layers ($L$): 32
  - Trainable parameters ($N_{\text{adapter}}$): $\approx 0.2\%\text{ to }0.4\%$ of base model $\approx 20 \times 10^6$ parameters (20M params)
- **Batch Size & Sequence Length:**
  - Micro-batch size per GPU ($B$): 2
  - Sequence length ($S$): 2048 tokens
  - Gradient Checkpointing (Activation Recomputation): **Enabled** (stores only layer input activations during forward pass, recomputing intermediate activations during backward pass).

---

## 2. Component-by-Component Memory Breakdown

### A. Model Weights ($M_{\text{weights}}$)
- **Full Fine-Tuning (16-bit):**
  $$M_{\text{weights}} = 7.0 \times 10^9 \times 2\text{ bytes} = 14.0\text{ GB}$$
- **LoRA (16-bit base + 16-bit adapters):**
  $$M_{\text{base}} = 7.0 \times 10^9 \times 2\text{ bytes} = 14.0\text{ GB}$$
  $$M_{\text{adapters}} = 20 \times 10^6 \times 2\text{ bytes} \approx 0.04\text{ GB}$$
  $$M_{\text{weights}} \approx 14.04\text{ GB}$$
- **QLoRA (4-bit NF4 base + 16-bit adapters):**
  $$M_{\text{base}} = 7.0 \times 10^9 \times 0.55\text{ bytes} \approx 3.85\text{ GB}$$
  $$M_{\text{adapters}} = 20 \times 10^6 \times 2\text{ bytes} \approx 0.04\text{ GB}$$
  $$M_{\text{weights}} \approx 3.89\text{ GB}$$

---

### B. Gradients ($M_{\text{grad}}$)
- **Full Fine-Tuning (16-bit gradients):**
  Gradients must be stored for all 7B parameters.
  $$M_{\text{grad}} = 7.0 \times 10^9 \times 2\text{ bytes} = 14.0\text{ GB}$$
  *(Note: If mixed precision optimizer computes FP32 gradients, this is $28.0\text{ GB}$. With FP16 gradient accumulation buffers, it is $14.0\text{ GB}$.)*
- **LoRA / QLoRA (16-bit gradients for adapters only):**
  Base model weights are frozen ($\nabla_{\theta_{\text{base}}} = 0$ is not stored).
  $$M_{\text{grad}} = 20 \times 10^6 \times 2\text{ bytes} \approx 0.04\text{ GB}$$

---

### C. Optimizer States ($M_{\text{opt}}$)
- **Full Fine-Tuning (AdamW, FP32 master weights + moments):**
  AdamW stores 1st moment ($4\text{ B}$), 2nd moment ($4\text{ B}$), and FP32 master copy of weights ($4\text{ B}$) per parameter:
  $$M_{\text{opt}} = 7.0 \times 10^9 \times 12\text{ bytes} = 84.0\text{ GB}$$
  *(With gradient buffer in optimizer: up to $112\text{ GB}$.)*
- **LoRA (AdamW for adapters only):**
  $$M_{\text{opt}} = 20 \times 10^6 \times 12\text{ bytes} \approx 0.24\text{ GB}$$
- **QLoRA (Paged AdamW 32-bit or 8-bit for adapters):**
  $$M_{\text{opt}} = 20 \times 10^6 \times 12\text{ bytes} \approx 0.24\text{ GB}\quad (\text{or } 0.04\text{ GB with 8-bit Adam})$$

---

### D. Activation Memory ($M_{\text{act}}$)
With **Gradient Checkpointing enabled**, intermediate activations are discarded and only the activations at the boundaries of transformer blocks are retained:
$$M_{\text{act}} \approx L \times B \times S \times d \times 2\text{ bytes}$$
For $L=32$, $B=2$, $S=2048$, $d=4096$:
$$M_{\text{act}} \approx 32 \times 2 \times 2048 \times 4096 \times 2 \approx 1,073,741,824\text{ bytes} \approx 1.07\text{ GB}$$
Accounting for attention logits, RMSNorm buffers, and LoRA forward branches during the backward recomputation phase, peak dynamic activation memory is typically:
$$M_{\text{act, peak}} \approx 2.5\text{ GB to } 4.0\text{ GB}$$

*(Without gradient checkpointing, activations scale linearly with all layers and attention matrices, requiring $>35\text{ GB}$ of VRAM alone!)*

---

### E. Runtime CUDA Overhead & PyTorch Workspace ($M_{\text{overhead}}$)
- CUDA context, cublas/cudnn workspace buffers, memory allocator fragmentation: $\approx 1.5\text{ GB to } 2.5\text{ GB}$.

---

## 3. Summary & Comparison Table (7B Model, Batch Size 2, Seq Len 2048, Grad Checkpointing)

| Memory Component | Full Fine-Tuning (16-bit) | LoRA (16-bit Base) | QLoRA (4-bit NF4 Base) |
| :--- | :--- | :--- | :--- |
| **Model Weights** | $14.0\text{ GB}$ | $14.04\text{ GB}$ | $3.89\text{ GB}$ |
| **Gradients** | $14.0\text{ GB}$ | $0.04\text{ GB}$ | $0.04\text{ GB}$ |
| **Optimizer States (AdamW)**| $84.0\text{ GB}$ | $0.24\text{ GB}$ | $0.24\text{ GB}$ |
| **Static State Total** | **$112.0\text{ GB}$** | **$14.32\text{ GB}$** | **$4.17\text{ GB}$** |
| **Peak Activations** (w/ Grad Checkpointing)| $\approx 3.5\text{ GB}$ | $\approx 3.5\text{ GB}$ | $\approx 3.5\text{ GB}$ |
| **CUDA & PyTorch Overhead** | $\approx 2.0\text{ GB}$ | $\approx 2.0\text{ GB}$ | $\approx 1.8\text{ GB}$ |
| **Estimated Peak VRAM** | **$\approx 117.5\text{ GB}$** | **$\approx 19.8\text{ GB}$** | **$\approx 9.5\text{ GB}$** |
| **Feasible Hardware** | $2\times\text{A100 (80GB)}$ or $4\times\text{A100 (40GB)}$ | Single A10 (24GB) / RTX 3090/4090 (24GB) | Consumer GPU / T4 (16GB) / Colab Free Tier |

---

## 4. Key Engineering Insights
1. **The Optimizer dominates Full Fine-Tuning:** Over 70% of peak memory in standard full fine-tuning is consumed by Adam optimizer momentum terms.
2. **LoRA eliminates 99.8% of optimizer and gradient memory:** Because only adapter weights receive gradients, optimizer memory drops from $\sim 84\text{ GB}$ to $<0.3\text{ GB}$.
3. **QLoRA shrinks the base weight memory by 72%:** Compressing 16-bit weights to 4-bit NF4 allows a 7B model to run comfortably on a single free 16 GB GPU (like an Nvidia T4 on Colab or Kaggle) with headroom for batch size 4 or longer context windows.
