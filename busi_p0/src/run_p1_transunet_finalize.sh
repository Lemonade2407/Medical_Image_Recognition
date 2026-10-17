#!/bin/bash
# Wait for the TransUNet P1 runs, then refresh the summary + both documents.
for i in $(seq 1 300); do
  grep -qh "P1 TRANSUNET DONE" /mnt/workspace/busi_p0/results/p1_transunet.log /mnt/workspace/busi_p0/results/p1_transunet2.log 2>/dev/null && break
  sleep 60
done
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python
{
  echo "=== summarize $(date) ==="
  $PY -u summarize_p1.py
  echo "=== regenerate docs ==="
  $PY make_comparison.py
  $PY make_p1_doc.py
} > /mnt/workspace/busi_p0/results/p1_transunet_finalize.log 2>&1
echo "DOCS UPDATED $(date)"
