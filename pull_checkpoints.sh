#!/bin/bash
# pull_checkpoints.sh
# Incremental backup script to continuously sync active training checkpoints down to local Windows disk.

REMOTE_HOST="somacooks-v3"
LOCAL_BACKUP_DIR="/mnt/c/Users/Gebruiker/Documents/Computational Bio/Somasays_Backup"

echo "=============================================="
echo "🔄 SOMASAYS INCREMENTAL CHECKPOINT SYNC"
echo "=============================================="

# 1. Create local backup directories
mkdir -p "$LOCAL_BACKUP_DIR/weights"
mkdir -p "$LOCAL_BACKUP_DIR/logs"
echo "[+] Sync destination: $LOCAL_BACKUP_DIR"

# 2. Sync the latest training logs
echo "[+] Syncing live loss logs and TensorBoard data..."
scp "$REMOTE_HOST:~/Somasays/weights/somasays_multimodal_v3/loss_log.csv" "$LOCAL_BACKUP_DIR/logs/" 2>/dev/null || true

# 3. Run incremental rsync for weights (only transfers newly finished epoch files!)
echo "[+] Running rsync for weights (Incremental & Compressed)..."
rsync -avzP --include="*.pth" --exclude="*" "$REMOTE_HOST:~/Somasays/weights/somasays_multimodal_v3/" "$LOCAL_BACKUP_DIR/weights/"

if [ $? -eq 0 ]; then
    echo "=============================================="
    echo "✅ SYNC COMPLETE! LOCAL BACKUP SECURED!"
    echo "📂 Destination: $LOCAL_BACKUP_DIR/weights"
    echo "=============================================="
else
    echo "=============================================="
    echo "⚠️  Sync partially failed. (Training is likely writing the file now)."
    echo "💡 Just re-run this script in a few minutes!"
    echo "=============================================="
fi
