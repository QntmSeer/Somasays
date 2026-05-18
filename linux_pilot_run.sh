#!/bin/bash
set -e

# ==============================================================================
# Somasays Pipeline - Linux Pilot Run Orchestrator
# This script sets up the local environment and executes the pipeline sequentially
# ==============================================================================

echo "========================================"
echo "Somasays Linux Pilot Orchestrator"
echo "========================================"

# 1. Environment Verification
echo "[1/5] Verifying Environment..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 could not be found."
    exit 1
fi

# Create a virtual environment specifically for the pilot if it doesn't exist
if [ ! -d "venv_somasays" ]; then
    echo "Creating virtual environment 'venv_somasays'..."
    python3 -m venv venv_somasays
fi

source venv_somasays/bin/activate

# Install essential pilot dependencies (CPU versions to avoid heavy GPU driver issues locally)
echo "Installing/Updating required ML dependencies (CPU versions for local pilot)..."
pip install --upgrade pip --quiet
pip install torch transformers peft datasets pyyaml requests --index-url https://download.pytorch.org/whl/cpu --quiet

echo "Environment ready."

# 2. Data Pipeline
echo "[2/5] Executing Data Pipeline..."
cd data_pipeline
python3 fetch_uniprot_data.py
python3 preprocess_sequences.py
cd ..

# 3. Model Training (Skipped/Dummy for Pilot)
# We don't want to run a 1.4B parameter training loop on a local CPU.
echo "[3/5] Skipping full ESM3 LoRA Training on local machine (Reserved for GCP A100)."
# Assuming the user has a weights directory or we will use the base model as fallback.

# 4. Generation Engine
echo "[4/5] Executing Candidate Generation Engine..."
cd generation_engine
python3 generate_candidates.py
cd ..

# 5. Evaluation Engine
echo "[5/5] Generating AlphaFold 3 evaluation jobs..."
cd evaluation_and_rescue
python3 af3_complex_predictor.py
python3 mpnn_stability_rescue.py
cd ..

echo "========================================"
echo "Somasays Pilot Run Complete!"
echo "Check the 'evaluation_and_rescue' directory for AF3 JSON jobs and MPNN FASTA outputs."
echo "========================================"
