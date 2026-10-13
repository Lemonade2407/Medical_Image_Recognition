#!/bin/bash
# Re-run TransUNet A-evals with FLIP-ONLY TTA (multi-scale breaks its fixed pos-embeddings).
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python
CK=/mnt/workspace/busi_p0/results/ckpt
RES=/mnt/workspace/busi_p0/results

cmds=()
for l in ce_dice tversky; do
  for s in 0 1 2; do
    suf=""; [ "$l" != "ce_dice" ] && suf="_$l"
    out="$RES/postproc_transunet_s256rgb${suf}_seed${s}.json"
    [ -f "$out" ] && { echo "SKIP $(basename "$out")"; continue; }
    cmds+=("$PY -u eval_postproc.py --ckpt $CK/ab_transunet${suf}_seed${s}.pt --model transunet --size 256 --rgb --seed $s --tta --scales 1.0 > $RES/eval_transunet${suf}_s${s}.log 2>&1")
  done
done
echo "to run: ${#cmds[@]}   parallelism=3"
i=0
for c in "${cmds[@]}"; do
  bash -c "$c" &
  i=$((i + 1))
  if [ $((i % 3)) -eq 0 ]; then wait; fi
done
wait
echo "=== summarize + regenerate doc ==="
$PY -u summarize_ab.py
$PY make_comparison.py
echo "TRANSUNET A DONE"
