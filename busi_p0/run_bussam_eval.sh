#!/bin/bash
# Wait for BUSSAM training to finish, then evaluate + regenerate the comparison doc.
cd /mnt/workspace/busi_p0
PY=/mnt/workspace/busi_p0/.venv/bin/python
for i in $(seq 1 240); do
  if ls /mnt/workspace/code/BUSSAM/outputs/BUSSAM_*/checkpoints/BUSSAM_*.pth >/dev/null 2>&1; then
    echo "training done at $(date)"; break
  fi
  sleep 30
done
echo "=== evaluating BUSSAM ==="
$PY -u src/eval_bussam.py 2>&1 | tee results/bussam_eval.log
echo "=== regenerating comparison doc ==="
$PY src/make_comparison.py
echo "BUSSAM PIPELINE DONE $(date)"
