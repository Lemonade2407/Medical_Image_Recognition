#!/bin/bash
# Unified-protocol re-runs for the remaining methods, on GPU.
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python

echo "=== SAM ViT-B point-prompt inference ==="
$PY -u prompt_eval.py --method sam || echo "SAM FAILED"

echo "=== MedSAM ViT-B point-prompt inference ==="
$PY -u prompt_eval.py --method medsam || echo "MEDSAM FAILED"

echo "OTHERS DONE"
