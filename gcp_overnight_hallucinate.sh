#!/bin/bash
# gcp_overnight_hallucinate.sh
# Somasays: Continuous Pulse Generation (Budget Protected)

BUCKET_NAME="somasays-storage"
OUTPUT_DIR="outputs/overnight_batch_1"
SYNC_INTERVAL=300 # 5 minutes
MAX_RUNTIME="6.5h"

echo "===================================================="
echo "   SOMASAYS UNSTOPPABLE ORACLE ENGINE   "
echo "===================================================="

# 1. Environment & Weights
source ~/soma_env/bin/activate || {
    python3 -m venv ~/soma_env
    source ~/soma_env/bin/activate
}
pip install esm google-cloud-storage httpx pydantic tqdm --quiet
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/gcp_key.json"

mkdir -p weights $OUTPUT_DIR

echo "[2/4] Retrieving fine-tuned weights..."
if command -v gsutil &> /dev/null; then
    gsutil cp gs://$BUCKET_NAME/weights/v3/esm3_multimodal_weights.pth ./weights/
else
    python3 pull_from_gcp.py
fi

# 2. Start BACKGROUND CLOUD SYNC
echo "[PULSE] Starting continuous cloud sync every 5m..."
(
    while true; do
        sleep $SYNC_INTERVAL
        python3 blast_to_gcp.py --src $OUTPUT_DIR --dest candidates/overnight_batch_1
        echo "[SYNC] Cloud heartbeat successful at $(date)"
    done
) &
SYNC_PID=$!

# 3. SET SELF-DESTRUCT TIMER (Safety Fuse)
echo "[FUSE] Safety shutdown scheduled for $MAX_RUNTIME..."
sudo shutdown -h +390 & # 390 minutes = 6.5 hours

# 4. RUN GENERATION CAMPAIGN
echo "[RUN] Launching Hallucination Campaign..."
python3 generation_engine/esm3_multimodal_generator.py \
    --weights weights/esm3_multimodal_weights.pth \
    --num 200 \
    --out $OUTPUT_DIR

# 5. FINAL CLEANUP
echo "[FINISH] Campaign complete. Final sync..."
kill $SYNC_PID
python3 blast_to_gcp.py --src $OUTPUT_DIR --dest candidates/overnight_batch_1
echo "===================================================="
echo "   MISSION COMPLETE. CLOUD IS SECURED.   "
echo "===================================================="
# sudo shutdown -h now
