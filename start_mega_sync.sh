#!/bin/bash
# start_mega_sync.sh
# Background sync for the 30k Mega-Batch

echo "🚀 Launching Somasays Mega-Sync Heartbeat..."

nohup bash -c "
while true; do
    echo '[SYNC] Starting parallel push to GCP at \$(date)...'
    python3 blast_to_gcp.py --src outputs/mega_batch_gpu0/ --dest mega_batch_gpu0
    python3 blast_to_gcp.py --src outputs/mega_batch_gpu1/ --dest mega_batch_gpu1
    echo '[SYNC] Cycle complete. Sleeping for 5m...'
    sleep 300
done
" > sync_mega.log 2>&1 &

echo "✅ Sync daemon is now running in the background."
echo "📜 Watch it with: tail -f sync_mega.log"
