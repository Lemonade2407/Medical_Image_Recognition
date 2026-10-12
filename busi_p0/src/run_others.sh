#!/bin/bash
# Unified-protocol re-runs for the remaining methods, on GPU.
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python

echo "=== SAM ViT-B point-prompt inference ==="
$PY -u prompt_eval.py --method sam || echo "SAM FAILED"

echo "=== MedSAM ViT-B point-prompt inference ==="
$PY -u prompt_eval.py --method medsam || echo "MEDSAM FAILED"

echo "=== UNet++ 256 RGB seed 0 ==="
$PY -u train_supervised.py --model unetpp --size 256 --rgb --epochs 24 --seed 0 || echo "UNETPP FAILED"

echo "=== TransUNet 256 RGB seed 0 ==="
$PY -u train_supervised.py --model transunet --size 256 --rgb --epochs 24 --seed 0 || echo "TRANSUNET FAILED"

echo "OTHERS DONE"
