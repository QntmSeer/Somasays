#!/bin/bash
# Somasays Phase 3: Brev Multimodal Orchestrator (A100 Optimized FSDP)

set -e

echo "=========================================="
echo "Somasays Phase 3: Brev.dev Multimodal Execution"
echo "=========================================="

# 1. Environment & Dependencies
echo "[1/4] Setting up Environment..."
if [ ! -d "venv_somasays" ]; then
    python3 -m venv venv_somasays
fi
source venv_somasays/bin/activate

echo "Installing heavy GPU dependencies..."
pip install --upgrade pip --quiet
pip install torch torchvision torchaudio accelerate datasets requests tqdm esm tensorboard --quiet

# 1.5 Hugging Face Authentication
echo "[1.5/4] Authenticating with Hugging Face for ESM3 access..."
python3 hf_login.py

# 2. Extract 1D Sequences
echo "[2/4] Verifying 1D sequence dataset..."
cd data_pipeline
if [ ! -f "../data/processed/somasays_dataset.jsonl" ]; then
    python3 fetch_uniprot_data.py
    python3 preprocess_sequences.py
else
    echo "1D Dataset found."
fi

# 3. Fetch 3D Structures
echo "[3/4] Fetching 3D PDB structures from AlphaFold DB..."
python3 fetch_alphafold_structures.py
cd ..

# 4. Custom Multimodal Training
echo "[4/4] Executing Distributed Multimodal ESM3 Training..."
cd model_training
# Streamlining PyTorch memory allocation for A100s
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Launching with FSDP/DDP (torchrun)
# Use --nproc_per_node=gpu to automatically detect all available GPUs (in this case, both L40S cards!)
# Hardcode localhost binding to bypass Brev's custom DNS IPv6 failures
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"
export NCCL_SOCKET_IFNAME="lo"
export GLOO_SOCKET_IFNAME="lo"

torchrun --nproc_per_node=gpu --rdzv_endpoint=localhost:29500 esm3_multimodal_trainer.py
cd ..

echo "=========================================="
echo "Phase 3 Training Complete. Models saved."
echo "=========================================="
