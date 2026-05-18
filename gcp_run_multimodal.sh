#!/bin/bash
# Somasays Phase 3: GCP Multimodal Orchestrator (A100 Optimized)

set -e

echo "=========================================="
echo "Somasays Phase 3: GCP Multimodal Execution"
echo "=========================================="

# 1. Environment & Dependencies
echo "[1/4] Setting up Environment..."
if [ ! -d "venv_somasays" ]; then
    python3 -m venv venv_somasays
fi
source venv_somasays/bin/activate

echo "Installing heavy GPU dependencies..."
pip install --upgrade pip --quiet
pip install torch torchvision torchaudio accelerate datasets requests tqdm esm --quiet

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
echo "[4/4] Executing Multimodal ESM3 Training..."
cd model_training
# Streamlining PyTorch memory allocation for GCP A100s
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 esm3_multimodal_trainer.py
cd ..

echo "=========================================="
echo "Phase 3 Training Complete. Models saved."
echo "=========================================="
