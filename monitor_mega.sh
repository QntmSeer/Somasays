#!/bin/bash
# monitor_mega.sh
# Somasays "Mega-Factory" Dashboard

echo "===================================================="
echo "   SOMASAYS DUAL-GPU INDUSTRIAL MONITOR   "
echo "===================================================="

while true; do
    clear
    echo "===================================================="
    echo "   SOMASAYS MEGA-FACTORY STATUS | $(date)"
    echo "===================================================="
    echo ""

    # GPU 0 Status
    COUNT0=$(ls -1 outputs/mega_batch_gpu0/*.pdb 2>/dev/null | wc -l)
    echo "🔥 [GPU 0] Progress: $COUNT0 / 15000"
    echo "----------------------------------------------------"
    tail -n 2 generation_gpu0.log
    echo ""

    # GPU 1 Status
    COUNT1=$(ls -1 outputs/mega_batch_gpu1/*.pdb 2>/dev/null | wc -l)
    echo "🔥 [GPU 1] Progress: $COUNT1 / 15000"
    echo "----------------------------------------------------"
    tail -n 2 generation_gpu1.log
    echo ""

    # Total & Sync
    TOTAL=$((COUNT0 + COUNT1))
    echo "📊 TOTAL Hallucinations: $TOTAL / 30000"
    echo "📂 Local Folders: outputs/mega_batch_gpu0, outputs/mega_batch_gpu1"
    
    if [ -f sync_mega.log ]; then
        echo "☁️  Sync Status: $(tail -n 1 sync_mega.log | cut -c1-60)..."
    fi

    echo ""
    echo "Press [Ctrl+C] to exit dashboard."
    sleep 10
done
