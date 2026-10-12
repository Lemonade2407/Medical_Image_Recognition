#!/bin/bash
# Full GPU pipeline: P0-2 (U-Net/Attention U-Net x3 seeds) then remaining methods.
cd /mnt/workspace/busi_p0
echo "START $(date)"
bash src/run_p02.sh 2>&1 | tee results/p02.log
bash src/run_others.sh 2>&1 | tee results/others.log
echo "ALL DONE $(date)"
