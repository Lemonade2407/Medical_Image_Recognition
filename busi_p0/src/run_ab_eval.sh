#!/bin/bash
# A-only re-run (B checkpoints already exist). NOTE: --seed must be passed or the
# three seeds overwrite the same output file.
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python
CK=/mnt/workspace/busi_p0/results/ckpt

echo "===== A: inference-side variants (A1 postproc / A2 TTA / A3 threshold) ====="
for m in unet attunet; do
  for l in ce_dice tversky boundary; do
    for s in 0 1 2; do
      suf=""; [ "$l" != "ce_dice" ] && suf="_$l"
      echo "--- A $m $l seed$s ---"
      $PY -u eval_postproc.py --ckpt "$CK/ab_${m}${suf}_seed${s}.pt" --model $m \
          --size 128 --gray --seed $s --tta || echo "EVAL FAIL $m $l $s"
    done
  done
done

echo "=== summarizing ==="
$PY -u summarize_ab.py
echo "A DONE"
