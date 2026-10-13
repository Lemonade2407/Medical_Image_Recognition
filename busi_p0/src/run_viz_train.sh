#!/bin/bash
# Retrain the 4 supervised models (seed 0) to persist checkpoints for visualization.
# Uses a viz_ tag so the canonical results/*.json are NOT overwritten.
set -e
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python
$PY -u train_supervised.py --model unet     --size 128 --gray --epochs 24 --seed 0 --workers 4 --tag viz_unet
$PY -u train_supervised.py --model attunet  --size 128 --gray --epochs 24 --seed 0 --workers 4 --tag viz_attunet
$PY -u train_supervised.py --model unetpp   --size 256 --rgb  --epochs 24 --seed 0 --workers 4 --tag viz_unetpp
$PY -u train_supervised.py --model transunet --size 256 --rgb --epochs 24 --seed 0 --workers 4 --tag viz_transunet
echo "VIZ CKPT DONE"
