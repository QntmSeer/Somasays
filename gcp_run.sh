#!/bin/bash
set -e

echo "========================================"
echo "Somasays FULL GCP GPU Orchestrator"
echo "========================================"

# 1. Environment Verification
echo "[1/5] Setting up Environment..."
sudo apt-get update -qq
sudo apt-get install -y python3-venv -qq

if [ ! -d "venv_somasays" ]; then
    python3 -m venv venv_somasays
fi
source venv_somasays/bin/activate

# Install GPU-enabled PyTorch and dependencies
echo "Installing GPU dependencies..."
pip install --upgrade pip --quiet
pip install torch torchvision torchaudio transformers peft datasets accelerate pyyaml requests tiktoken sentencepiece tokenizers protobuf esm --quiet

# 2. Data Pipeline
echo "[2/5] Executing Data Pipeline..."
cd data_pipeline
python3 fetch_uniprot_data.py
python3 preprocess_sequences.py
cd ..

# 3. Model Training (GPU Accelerated)
echo "[3/5] Executing ESM3 LoRA Training on L4 GPU..."
cd model_training
python3 esm3_lora_finetune.py
cd ..

# 4. Generation Engine
echo "[4/5] Executing Candidate Generation Engine..."
cd generation_engine
python3 generate_candidates.py
cd ..

# 5. Evaluation Engine
echo "[5/5] Generating AlphaFold 3 evaluation jobs and MPNN Rescue..."
cd evaluation_and_rescue
python3 af3_complex_predictor.py
python3 mpnn_stability_rescue.py
cd ..

echo "========================================"
echo "Somasays GCP Run Complete!"
echo "Outputs are ready in evaluation_and_rescue/af3_jobs/"
echo "========================================"
