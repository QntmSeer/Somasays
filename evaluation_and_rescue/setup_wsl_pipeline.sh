#!/usr/bin/env bash
# setup_wsl_pipeline.sh
# Run this script inside WSL (Ubuntu) from the Somasays root folder:
#   bash evaluation_and_rescue/setup_wsl_pipeline.sh

set -euo pipefail

WORKSPACE_DIR="/mnt/c/Users/Gebruiker/Documents/Computational Bio/Somasays"
cd "$WORKSPACE_DIR"

echo "======================================================="
echo "[*] Installing dependencies in WSL user-space..."
echo "======================================================="

echo "[*] Installing PyTorch with CUDA 12.4 support..."
python3 -m pip install --user --break-system-packages torch --index-url https://download.pytorch.org/whl/cu124

echo "[*] Installing biopython, numpy, scipy, pandas, transformers, peft, huggingface_hub..."
python3 -m pip install --user --break-system-packages biopython numpy scipy pandas transformers peft huggingface_hub

echo "[*] Installing AlphaJudge..."
python3 -m pip install --user --break-system-packages alphajudge

echo "======================================================="
echo "[*] Setting up Absolute Stability Predictor..."
echo "======================================================="
PREDICTOR_DIR="evaluation_and_rescue/absolute-stability-predictor"

if [ ! -d "$PREDICTOR_DIR" ]; then
    echo "Cloning absolute-stability-predictor repository..."
    git clone https://github.com/yehlincho/absolute-stability-predictor.git "$PREDICTOR_DIR"
else
    echo "absolute-stability-predictor repository already cloned."
fi

# Install predictor package in editable mode
echo "Installing absolute-stability-predictor..."
python3 -m pip install --user --break-system-packages -e "$PREDICTOR_DIR"

# Download SaProtΔG weights from Hugging Face
echo "Downloading SaProtΔG weights from Hugging Face (Yehlin/absolute-stability)..."
python3 - <<PYEOF
import os
from huggingface_hub import hf_hub_download

repo_id = "Yehlin/absolute-stability"
dest_dir = "$WORKSPACE_DIR/$PREDICTOR_DIR"
os.makedirs(os.path.join(dest_dir, "saprotdg_weights"), exist_ok=True)

files = [
    "saprotdg_weights/SaProtdG_weights_augmented_1_lora.ckpt",
    "saprotdg_weights/SaProtdG_weights_augmented_2_lora.ckpt",
    "saprotdg_weights/SaProtdG_weights_augmented_3_lora.ckpt",
]

for f in files:
    dest_path = os.path.join(dest_dir, f)
    if os.path.exists(dest_path):
        print(f"  [skip] {f} already exists.")
        continue
    print(f"  Downloading {f}...")
    hf_hub_download(
        repo_id=repo_id,
        filename=f,
        local_dir=dest_dir,
    )
PYEOF

echo "======================================================="
echo "[*] Verifying CUDA and GPU availability in WSL..."
echo "======================================================="
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "======================================================="
echo "[SUCCESS] WSL Pipeline Setup Completed!"
echo "To run the evaluation script, use:"
echo "  wsl python3 evaluation_and_rescue/evaluate_cysteine_free_complexes.py [downloads_dir]"
echo "======================================================="
