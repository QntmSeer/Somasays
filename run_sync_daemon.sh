#!/bin/bash
# run_sync_daemon.sh
# A background daemon that loops forever and syncs down training checkpoints to local disk.

REMOTE_HOST="somacooks-v3"
LOCAL_BACKUP_DIR="/mnt/c/Users/Gebruiker/Documents/Computational Bio/Somasays_Backup"

mkdir -p "$LOCAL_BACKUP_DIR/weights"
mkdir -p "$LOCAL_BACKUP_DIR/logs"

echo "======================================================"
echo "🛡️  SOMASAYS AUTOMATIC SYNC DAEMON RUNNING..."
echo "📂 Local Backup: $LOCAL_BACKUP_DIR/weights"
echo "⏱️  Sync Frequency: Every 30 Seconds"
echo "======================================================"

while true; do
    # Quiet incremental sync of weights
    rsync -az --include="*.pth" --exclude="*" "$REMOTE_HOST:~/Somasays/weights/somasays_multimodal_v3/" "$LOCAL_BACKUP_DIR/weights/" 2>/dev/null
    
    # Overwrite the latest CSV metrics log file locally
    scp -q "$REMOTE_HOST:~/Somasays/weights/somasays_multimodal_v3/loss_log.csv" "$LOCAL_BACKUP_DIR/logs/" 2>/dev/null
    
    sleep 30
done
