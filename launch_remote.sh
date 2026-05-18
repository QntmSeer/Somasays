#!/bin/bash
# Remote Launcher for Somasays Resurrection Run
cd /home/shadeform/Somasays
chmod +x brev_run_multimodal.sh
screen -d -m -S somatraining bash -c "./brev_run_multimodal.sh 2>&1 | tee multimodal_run.log"
echo "=========================================================="
echo "🚀 Resurrection Run launched in background Screen session!"
echo "🚀 Session Name: somatraining"
echo "🚀 Live Log File: /home/shadeform/Somasays/multimodal_run.log"
echo "=========================================================="
