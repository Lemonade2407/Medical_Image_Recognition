#!/bin/bash
# Wait for Stage-2 (UNet++/TransUNet A/B) to finish, then refresh the summary and
# regenerate the comparison doc so the new numbers are written in.
for i in $(seq 1 220); do
  grep -q "AB2 DONE" /mnt/workspace/busi_p0/results/ab2.log 2>/dev/null && break
  sleep 60
done
cd /mnt/workspace/busi_p0
PY=/mnt/workspace/busi_p0/.venv/bin/python
{
  echo "=== summarize $(date) ==="
  $PY -u src/summarize_ab.py
  echo "=== regenerate 七种模型对比.md ==="
  $PY src/make_comparison.py
} > results/finalize.log 2>&1
echo "DOCS UPDATED $(date)"
