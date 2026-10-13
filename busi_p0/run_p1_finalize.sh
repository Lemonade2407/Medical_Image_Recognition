#!/bin/bash
# Wait for P1 to finish, then summarize it and write the results into the comparison report.
for i in $(seq 1 300); do
  grep -q "P1 DONE" /mnt/workspace/busi_p0/results/p1.log 2>/dev/null && break
  sleep 60
done
cd /mnt/workspace/busi_p0
PY=/mnt/workspace/busi_p0/.venv/bin/python
{
  echo "=== P1 summarize $(date) ==="
  $PY -u src/summarize_p1.py
  echo "=== regenerate docs/七种模型对比.md ==="
  $PY src/make_comparison.py
} > results/p1_finalize.log 2>&1
echo "P1 DOCS UPDATED $(date)"
