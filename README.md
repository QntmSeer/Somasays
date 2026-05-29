<p align="center">
  <img src="somasays_pixel_logo.png" alt="Somasays Logo" width="600">
</p>

# Somasays: High-Throughput ESM3 Fine-Tuning & Inference Optimization Engine

Somasays is an end-to-end, production-grade computational biology framework for de novo plant-like peptide design, 3D structural folding, and high-velocity inference optimization using the **EvolutionaryScale ESM3 (1.4B Parameter) Multimodal Protein Foundation Model**. 

This repository provides a complete pipeline from raw sequence preprocessing to distributed multi-GPU training, real-time performance profiling, structural FlashAttention optimizations, and downstream biophysical evaluation.

---

## Core Platform Architecture

```mermaid
graph TD
    subgraph Data Pipeline
        A[UniProt Metadata Extraction] -->|fetch_uniprot_data.py| B[Sequence Preprocessing]
        B -->|preprocess_sequences.py| C[AlphaFold 3D Cache]
    end
    
    subgraph Distributed Training
        C -->|esm3_lora_finetune.py| D[Distributed Data Parallel DDP]
        D -->|NCCL, Static Graph, BF16 AMP| E[Fine-Tuned Adapter Weights]
    end
    
    subgraph Optimization & Generation
        E -->|generate_candidates.py| F[Optimized ESM3 Generator]
        F -->|ProteinMPNN sequence rescue| G[Candidate Sequences]
        G -->|evolutionary_optimizer.py| H[In Silico Directed Evolution]
        H -->|Simulated Annealing + pI / GRAVY / MHC Filters| I[Evolved Cysteine-Free Candidates]
    end
    
    subgraph Downstream Validation & Serving
        I -->|evaluate_cysteine_free_complexes.py| K[Joint WLSS Evaluation Pipeline]
        K -->|Calculates WLSS score| L[Wet-Lab Success Leaderboard]
        I -->|api_service/server.py| J[FastAPI Model Serving API]
    end
```

---

## Key Technical Features

1. **High-Throughput Distributed Fine-Tuning**:
   * Multi-GPU distributed training using PyTorch **Distributed Data Parallel (DDP)** with NCCL communication backends.
   * Leverages transformer **gradient checkpointing** and compiles a **static computation graph** topology (`_set_static_graph()`) to eliminate graph re-entrancy overhead.
   * Integrates pre-cached data-loading entirely in host memory (100GB Host RAM pre-caching) to eliminate disk I/O bottlenecks.

2. **Production-Grade Inference Optimizations**:
   * System-level Scaled Dot Product Attention (SDPA) backend configuration forcing **FlashAttention-2** execution, reducing matrix complexity from quadratic $O(N^2)$ to linear $O(N)$.
   * Automatic Mixed Precision (AMP) utilizing hardware bfloat16 Tensor Cores for a **2x latency speedup** and half-precision memory allocation.
   * Custom low-precision casting and quantization interfaces.

3. **High-Resolution Performance Profiling**:
   * Low-overhead GPU profiler tracking token-by-token latency, VRAM footprint allocation, and residue generation throughput curves.
   * Automated benchmark suites sweeping protein sequence contexts up to **2,048 residues**.

4. **In Silico Directed Evolution Engine**:
   * Simulated annealing sequence search loop to optimize monomer folding stability ($\Delta G$) using a 3-model **SaProtΔG** ensemble.
   * Implements strict biophysical constraint checks in the active search loop:
     * **pI Filter:** Automatically rejects sequences with an isoelectric point inside the physiological pH precipitation zone ($6.8 \le pI \le 8.0$).
     * **Solubility Filter:** GRAVY hydropathy index cannot exceed a baseline limit (0.45).
     * **MHC-II Immunogenicity Filter:** Ensures no new HLA-DRB1 binding core epitopes are introduced.
     * **Thiol-free Constraint:** Enforces 100% cysteine-free designs to prevent uncontrolled disulfide aggregation in wet-lab assays.

5. **Joint Wet-Lab Success Scoring (WLSS)**:
   * Consolidates multi-parameter metrics into a single **Wet-Lab Success Score (WLSS)**: 50% target binding kinetics, 30% folding energy, and 20% manufacturability.
   * Scans for post-translational modification (PTM) risk hotspots (glycosylation, deamidation, acid cleavage, methionine oxidation), hydrophobic patches, and estimates cyclization feasibility.

---

## Repository Directory Layout

```text
Somasays/
├── api_service/                 # Atlassian-inspired Model Serving Edge Service
│   ├── server.py                    # FastAPI web server, routing & polling schema
│   └── tasks.db                     # Decoupled task queue state tracking
├── data_pipeline/               # Data ingestion & caching
│   ├── fetch_uniprot_data.py        # Extracts raw metadata and sequence strings
│   ├── preprocess_sequences.py      # Standardizes sequences for tokenization
│   └── fetch_alphafold_structures.py # Pulls PDB coordinates from AlphaFold DB
├── model_training/              # Heavy GPU training scripts
│   ├── training_config.yaml         # Core hyperparameters configuration
│   ├── esm3_lora_finetune.py        # PEFT Masked Language Modeling
│   └── esm3_multimodal_trainer.py   # Distributed Multimodal DDP training
├── generation_engine/           # Synthesis and performance optimization
│   ├── generate_candidates.py       # MLM peptide sampling
│   ├── esm3_multimodal_generator.py # Dual-track sequence & coordinate generator
│   ├── optimized_inference.py       # FlashAttention SDPA / AMP optimization wrapper
│   └── profile_inference.py         # Latency and peak VRAM profiling engine
├── evaluation_and_rescue/       # Downstream verification
│   ├── evolutionary_optimizer.py    # Simulated annealing directed evolution
│   ├── evaluate_cysteine_free_complexes.py # Joint WLSS validation pipeline
│   ├── proteinmpnn_rescue.py        # Backbone sequence co-design
│   ├── mpnn_stability_rescue.py     # Stability optimization scripts
│   ├── candidate_rescuer.py         # Rescues wildtypes using ProteinMPNN
│   ├── cysteine_free_rescuer.py     # Mutates cysteines and evaluates stability
│   ├── manufacturability_profiler.py # Biophysical risk assessment profiler
│   ├── binding_interface_analyzer.py # Parses complexes for contact maps, ipTM & pLDDT
│   ├── codon_optimizer_carbon.py    # Codon optimization (E. coli & Human expression)
│   ├── structural_qc.py             # MHC-II epitope prediction using absolute stability methods
│   └── umap_embedding_analysis.py   # Synthesized space embedding projection
├── analysis/                    # Benchmarking suite & visualizers
│   ├── benchmark_suite.py           # Auto-sweeps lengths, batch sizes & configs
│   ├── plot_convergence_curves.py   # Visualizes training loss curves
│   └── outputs/                     # Latency, throughput, and memory charts
├── README.md                    # Platform overview & execution guide
└── optimizations_case_study.md  # Professional ESM3 performance report
```

---

## Execution & Quick Start Guide

### 1. Environment Activation & Dependencies
Ensure your environment contains CUDA 12+ and PyTorch 2.0+ with matching drivers:
```bash
# Activate virtual environment
source venv_somasays/bin/activate

# Install core packages
pip install torch torchvision torchaudio esm biopython matplotlib pandas --quiet
```

### 2. High-Resolution Model Profiling
Profile the latency and VRAM footprint of sequence autoregression and coordinate folding under baseline configurations:
```bash
python generation_engine/profile_inference.py --prompt "MKA___________________VLA" --steps 8
```

### 3. Running the Optimization Benchmark Sweep
Sweep sequence lengths up to 2,048 residues to generate comparative line charts mapping latency, throughput, and VRAM efficiency:
```bash
python analysis/benchmark_suite.py --outdir analysis/outputs
```

### 4. Deploying the Model Serving API Service (Edge Architecture)
Spin up the systems-optimized model serving API. This utilizes a decoupled, asynchronous queue architecture to isolate heavy GPU inference tasks:
```bash
# Install server dependencies
pip install fastapi uvicorn pydantic --quiet

# Launch the FastAPI web engine
uvicorn api_service.server:app --host 0.0.0.0 --port 8000 --reload
```

#### Interaction Flow:
* **Submit a folding/generation task**:
  ```bash
  curl -X POST "http://localhost:8000/v1/tasks" \
       -H "Content-Type: application/json" \
       -d "{\"prompt_sequence\": \"MKA___________________VLA\", \"num_steps\": 8, \"temperature\": 0.7}"
  ```
  *Response*: `{"task_id": "a90f117c-...", "status": "PENDING", ...}`

* **Poll for completion and coordinate outputs**:
  ```bash
  curl "http://localhost:8000/v1/tasks/<task_id>"
  ```

### 5. Downstream Validation & Analysis

#### A. Run In Silico Directed Evolution
Execute the simulated annealing directed evolution search starting from a baseline sequence to maximize folding stability under biophysical constraints (no cysteines, safe pI, solubility, non-immunogenic):
```bash
python evaluation_and_rescue/evolutionary_optimizer.py --steps 50 --output outputs/evolution_history.csv
```

#### B. Run Joint Validation & WLSS Scoring
Compile structural and biophysical characteristics for all generated candidates into a ranked leaderboard scoring Wet-Lab Success Score (WLSS):
```bash
python evaluation_and_rescue/evaluate_cysteine_free_complexes.py \
    --in_dir outputs/combined_runs \
    --out_dir outputs
```
This generates `outputs/joint_evaluation_report.md` (leaderboard) and `outputs/joint_evaluation_report.csv`.

#### C. Run Biophysical Manufacturability Profiler
Evaluate designed candidates for glycosylation traps, deamidation, acid cleavage susceptibility, and GRAVY hydropathy:
```bash
python evaluation_and_rescue/manufacturability_profiler.py \
    --in_dir outputs/mpnn_best_sequences \
    --out_dir outputs
```

#### D. Run Binding Interface & Contact Analyzer
Parse PDB structural complexes and confidence JSONs to compute contact maps, hydrogen bonds, salt bridges, pLDDT scores, and ipTM rankings:
```bash
python evaluation_and_rescue/binding_interface_analyzer.py \
    --in_dir outputs/af3_results \
    --out_dir outputs \
    --binder_chain A
```

---

## Quantified Performance Gains

Our detailed benchmarking indicates that combining **bfloat16 AMP** with **FlashAttention (SDPA)** eliminates baseline computational limits:
* **3.4x Speedup**: Latency during long structural folding loops scales linearly instead of quadratically.
* **58% VRAM Reduction**: Peak memory footprint at 1,024 residues drops from 14.6 GB to **5.9 GB**.
* **Expanded Context Limits**: Prevents unoptimized Out-Of-Memory (OOM) crashes, extending the maximum folding length from **1,024 to 2,048 residues**.

For a publication-grade breakdown of our findings, GPU hardware profiles, and optimization methodologies, read our full [ESM3 Optimization and Performance Case Study](optimizations_case_study.md).

### Scientific Validation & Design Space Charts

| Design Space Landscape | Directed Evolution Trajectory | Biophysical Heatmap Leaderboard |
| :---: | :---: | :---: |
| ![Design Space](analysis/outputs/design_space_landscape_minimal.png) | ![Evolution Trajectory](analysis/outputs/directed_evolution_trajectory_minimal.png) | ![Biophysical Heatmap](analysis/outputs/biophysical_profile_comparison_minimal.png) |

---


## What is Left to be Done (Roadmap)

To elevate Somasays into a fully automated, web-scale biological factory, the following roadmap features are planned for future development:

1. **Dynamic Tensor Parallelism (TP)**:
   * Integrate DeepSpeed or Megatron-LM to shard the 1.4B parameters and attention matrices across multiple GPU nodes. This will enable structural folding of large multi-domain complexes exceeding 4,000 residues.

2. **Hopper Native FP8 & FlashAttention-3**:
   * Migrate SDPA backends to native FlashAttention-3 kernels on Hopper GPU architectures (H100/H200). This will utilize low-precision FP8 Tensor Cores to speed up sequential autoregressive decoding.

3. **4-Bit NF4 Quantization (Double Quantization)**:
   * Implement 4-bit NormalFloat (NF4) dynamic loading interfaces using `bitsandbytes`. This will compress the active model footprint below 1 GB VRAM, allowing full 3D structural inference on low-cost consumer GPUs.

4. **Asynchronous AlphaFold 3 API Loop**:
   * Build a background daemon to automatically submit generated protein coordinates to the AlphaFold 3 server, parse confidence metrics (pLDDT, iPAE), and store results in a PostgreSQL database for real-time downstream validation.

