#!/bin/bash
# Stage-2b: remaining A/B for UNet++ / TransUNet, boundary loss SKIPPED (proven harmful).
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python
CK=/mnt/workspace/busi_p0/results/ckpt
RES=/mnt/workspace/busi_p0/results

echo "===== B (rest): TransUNet ce_dice / tversky x 3 seeds ====="
for l in ce_dice tversky; do
  for s in 0 1 2; do
    echo "--- B transunet $l seed$s ---"
    $PY -u train_supervised.py --model transunet --size 256 --rgb --epochs 24 --seed $s \
        --workers 4 --loss $l --save-ckpt --tag ab_transunet || echo "TRAIN FAIL $l $s"
  done
done

echo "===== A: UNet++ / TransUNet (ce_dice, tversky) ====="
cmds=()
for m in unetpp transunet; do
  for l in ce_dice tversky; do
    for s in 0 1 2; do
      suf=""; [ "$l" != "ce_dice" ] && suf="_$l"
      out="$RES/postproc_${m}_s256rgb${suf}_seed${s}.json"
      [ -f "$out" ] && { echo "SKIP $(basename "$out")"; continue; }
      cmds+=("$PY -u eval_postproc.py --ckpt $CK/ab_${m}${suf}_seed${s}.pt --model $m --size 256 --rgb --seed $s --tta > $RES/eval_${m}${suf}_s${s}.log 2>&1")
    done
  done
done
echo "to run: ${#cmds[@]}   parallelism=4"
i=0
for c in "${cmds[@]}"; do
  bash -c "$c" &
  i=$((i + 1))
  if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait
echo "AB2 DONE"
