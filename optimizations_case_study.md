<p align="center">
  <img src="somasays_pixel_logo.png" alt="Somasays Logo" width="600">
</p>

# ESM3 Protein Foundation Model Optimization & Performance Case Study

This case study documents the engineering methodologies, performance bottlenecks, and structural optimizations applied to the **EvolutionaryScale ESM3 (1.4B Parameter) Multimodal Protein Foundation Model** within the Somasays platform. 

The Somasays platform accelerates de novo plant-like peptide generation and 3D coordinate folding. However, deploying ESM3 at scale in high-throughput production environments introduces significant performance challenges:
1. **Autoregressive Latency**: High latency during sequential sequence generation tracks.
2. **Quadratic VRAM Scaling**: Out-Of-Memory (OOM) failures due to $O(N^2)$ memory scaling in geometric attention layers during structural folding on sequences exceeding 1,024 residues.

---

## 1. Inference Bottleneck Analysis

High-resolution profiling using `profile_inference.py` revealed two distinct performance bottlenecks in the ESM3 base execution pipeline:

### A. Autoregressive Memory Bandwidth Bottleneck
The sequence generation track autoregressively samples amino acid tokens step-by-step. Each token generation step requires transferring the entire 1.4B parameters of the model through the GPU memory bus, resulting in memory-bandwidth-bound latency. Under standard FP32 precision, sequence generation throughput is throttled, averaging **50.8 tokens/second** on standard GPU nodes.

### B. Quadratic Attention Memory Scaling
The structural folding track resolves 3D coordinates by processing sequence tokens and spatial geometric coordinates. The attention matrices scale quadratically ($O(N^2)$) with the sequence length. Under unoptimized FP32 baseline execution, this results in rapid VRAM escalation:
* A **512-residue** protein requires **7.8 GB** of peak VRAM.
* A **1,024-residue** protein requires **14.6 GB** of peak VRAM.
* For proteins exceeding **1,024 residues**, unoptimized attention matrices trigger **Out-Of-Memory (OOM)** failures, capping the system's biological scope.

---

## 2. Optimization Methodology

To resolve these bottlenecks and build a high-velocity production pipeline, three structural optimizations were implemented in `optimized_inference.py`:

### A. Automatic Mixed Precision (AMP) & Hardware Cores
Model weights and activations were cast to low-precision formats (`float16` and `bfloat16`). On modern GPU architectures (e.g., NVIDIA L4 and A100), `bfloat16` utilizes specialized Tensor Cores to provide a **2x hardware acceleration** while preserving the dynamic range of FP32. This eliminates underflow/overflow bugs during structural coordinate normalization and cuts base weight memory footprints in half.

### B. FlashAttention & Scaled Dot Product Attention (SDPA)
Standard PyTorch mathematical attention was replaced by system-level configuration of Scaled Dot Product Attention (SDPA). By explicitly restricting standard math fallbacks (`math_sdp=False`) and forcing `flash_sdp` and `mem_efficient_sdp`, PyTorch compiles attention calculations to highly optimized **FlashAttention-2** CUDA kernels. 

FlashAttention-2 optimizes memory hierarchy usage by computing attention in SRAM blocks rather than transferring full intermediate matrices back to High-Bandwidth Memory (HBM). This reduces attention-matrix memory scaling from $O(N^2)$ to $O(N)$, drastically lowering VRAM scaling and speeding up computations.

### C. Low-Precision Weight Quantization
A dedicated wrapper interface was designed to support low-precision loading configuration. This allows the model to load in 8-bit or 4-bit configurations via `bitsandbytes`, compressing the 1.4B model weights down to **1.4 GB / 0.7 GB** respectively, enabling large foundation model inference on small, cost-effective GPU instances.

---

## 3. Quantified Performance Benchmarks

The benchmark suite (`benchmark_suite.py`) systematically swept sequence lengths from **128 to 2,048 residues** across four configurations on an enterprise cloud GPU node. The quantified results are detailed below:

### Comparative Performance Table

| Configuration | Sequence Length | Latency (Sec) | Peak GPU VRAM (GB) | OOM Status | Throughput (Tokens/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | 128 | 2.50 | 4.20 | NO | 51.20 |
| | 512 | 9.20 | 7.80 | NO | 55.65 |
| | 1024 | 19.50 | 14.60 | NO | 52.51 |
| | 1536 | N/A | N/A | **YES** | N/A |
| | 2048 | N/A | N/A | **YES** | N/A |
| **FP16 Mixed Precision** | 128 | 1.40 | 2.30 | NO | 91.43 |
| | 512 | 5.10 | 4.60 | NO | 100.39 |
| | 1024 | 10.40 | 8.80 | NO | 98.46 |
| | 1536 | 16.20 | 13.50 | NO | 94.81 |
| | 2048 | N/A | N/A | **YES** | N/A |
| **BF16 + FlashAttention (SDPA)** | 128 | 0.90 | 2.10 | NO | **142.22** |
| | 512 | 3.10 | 3.50 | NO | **165.16** |
| | 1024 | 5.80 | 5.90 | NO | **176.55** |
| | 1536 | 8.90 | 8.40 | NO | **172.58** |
| | 2048 | 12.40 | 11.20 | NO | **165.16** |
| **Quantized 8-Bit** | 128 | 1.20 | 1.20 | NO | 106.67 |
| | 512 | 4.20 | 1.80 | NO | 121.90 |
| | 1024 | 8.10 | 3.10 | NO | 126.42 |
| | 1536 | 12.20 | 4.50 | NO | 125.90 |
| | 2048 | 16.80 | 6.10 | NO | 121.90 |

---

## 4. Key Visualization Curves

The generated line charts demonstrate the dramatic improvements in latency scaling, memory efficiency, and inference throughput:

### A. Latency Scaling Comparison
FlashAttention + BF16 reduces latency scaling from a steep curve to a highly manageable linear trend, yielding a **3.4x speedup** for large sequences.
![Latency Scaling Curves](analysis/outputs/latency_comparison.png)

### B. VRAM Memory Footprint Comparison
Unoptimized FP32 baseline execution triggers OOMs past 1,024 residues, whereas the **BF16 + FlashAttention (SDPA)** configuration maintains a low, linear memory profile, enabling structural folding up to **2,048 residues** (a **58% VRAM reduction** at 1,024 residues).
![VRAM Footprint Curves](analysis/outputs/memory_comparison.png)

### C. Inference Throughput Comparison
Low-precision operations combined with SRAM block-level attention tracking push generation throughput from a baseline average of 52 tokens/s to a peak of **176 tokens/second**.
![Throughput Curves](analysis/outputs/throughput_comparison.png)

---

