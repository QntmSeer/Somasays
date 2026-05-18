#!/bin/bash
#SBATCH --job-name=somasays_multimodal
#SBATCH --output=somasays_hpc_%j.log
#SBATCH --error=somasays_hpc_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1          # Requesting 1x A100 for heavy multimodal tensors
#SBATCH --cpus-per-task=16     # Required for rapid threaded PDB downloading
#SBATCH --mem=128G             # High memory for caching 3D structures
#SBATCH --time=48:00:00

echo "=========================================="
echo "Somasays Phase 3: HPC Multimodal Execution"
echo "=========================================="

# 1. Setup Environment
echo "Setting up environment..."
module load python/3.10
module load cuda/11.8

if [ ! -d "venv_somasays" ]; then
    python3 -m venv venv_somasays
fi
source venv_somasays/bin/activate
pip install torch torchvision torchaudio transformers accelerate datasets requests tqdm esm --quiet

# 2. Fetch 3D Structures from AlphaFold DB
echo "Fetching massive 3D structure dataset..."
cd data_pipeline
python3 fetch_alphafold_structures.py
cd ..

# 3. Multimodal ESM3 Training
echo "Initiating Custom Multimodal ESM3 Training Loop..."
cd model_training
python3 esm3_multimodal_trainer.py
cd ..

echo "HPC Pipeline Complete. Models saved to weights/somasays_multimodal_v3/"
