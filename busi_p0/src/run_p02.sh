#!/bin/bash
# P0-2 driver: 3 seeds x {U-Net, Attention U-Net} on GPU, then summarize + Wilcoxon.
set -e
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python
rm -f /mnt/workspace/busi_p0/results/smoke_*.json
for m in unet attunet; do
  for s in 0 1 2; do
    echo "=== training $m seed=$s ==="
    $PY -u train_supervised.py --model "$m" --size 128 --gray --epochs 24 --seed "$s"
  done
done
echo "=== summarizing ==="
$PY -u summarize_p02.py
echo "P0-2 DONE"
