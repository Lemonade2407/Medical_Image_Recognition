#!/bin/bash
# Wait for P1, then run P2 (prompt-quality -> Dice curve) with the two refiners in parallel.
for i in $(seq 1 300); do
  grep -q "P1 DONE" /mnt/workspace/busi_p0/results/p1.log 2>/dev/null && break
  sleep 60
done
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python
RES=/mnt/workspace/busi_p0/results
echo "P1 finished; starting P2 (2-way parallel) $(date)"
$PY -u eval_prompt_quality.py --method medsam > "$RES/p2_medsam.log" 2>&1 &
$PY -u eval_prompt_quality.py --method sam    > "$RES/p2_sam.log" 2>&1 &
wait
$PY -u merge_prompt_quality.py
echo "P2 DONE $(date)"
