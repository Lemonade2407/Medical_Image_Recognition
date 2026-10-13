#!/bin/bash
# Parallel A-phase evaluation. The eval is CPU-bound (GPU idle), so run several
# concurrently. Results are identical to the sequential run -- only wall time changes.
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python
CK=/mnt/workspace/busi_p0/results/ckpt
RES=/mnt/workspace/busi_p0/results
P="${P:-6}"

cmds=()
for m in unet attunet; do
  for l in ce_dice tversky boundary; do
    for s in 0 1 2; do
      suf=""; [ "$l" != "ce_dice" ] && suf="_$l"
      out="$RES/postproc_${m}_s128g${suf}_seed${s}.json"
      if [ -f "$out" ]; then echo "SKIP $(basename "$out")"; continue; fi
      cmds+=("$PY -u eval_postproc.py --ckpt $CK/ab_${m}${suf}_seed${s}.pt --model $m --size 128 --gray --seed $s --tta > $RES/eval_${m}${suf}_s${s}.log 2>&1")
    done
  done
done

echo "to run: ${#cmds[@]}   parallelism=$P"
i=0
for c in "${cmds[@]}"; do
  bash -c "$c" &
  i=$((i + 1))
  if [ $((i % P)) -eq 0 ]; then wait; fi
done
wait
echo "=== summarizing ==="
$PY -u summarize_ab.py
echo "A DONE"
