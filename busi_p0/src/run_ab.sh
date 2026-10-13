#!/bin/bash
# A/B optimisation sweep.
#   B = loss functions (retrain): ce_dice (baseline) / tversky / boundary x 3 seeds
#   A = inference-side add-ons: A1 postproc, A2 TTA, A3 threshold search
# Uses the `ab_` tag so canonical results/*.json are NOT overwritten.
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python
CK=/mnt/workspace/busi_p0/results/ckpt

echo "===== B: retrain U-Net / AttU-Net (losses x 3 seeds) ====="
for m in unet attunet; do
  for l in ce_dice tversky boundary; do
    for s in 0 1 2; do
      echo "--- B $m $l seed$s ---"
      $PY -u train_supervised.py --model $m --size 128 --gray --epochs 24 --seed $s \
          --workers 4 --loss $l --save-ckpt --tag ab_$m || echo "TRAIN FAIL $m $l $s"
    done
  done
done

echo "===== A: inference-side variants ====="
for m in unet attunet; do
  for l in ce_dice tversky boundary; do
    for s in 0 1 2; do
      suf=""; [ "$l" != "ce_dice" ] && suf="_$l"
      echo "--- A $m $l seed$s ---"
      $PY -u eval_postproc.py --ckpt "$CK/ab_${m}${suf}_seed${s}.pt" \
          --model $m --size 128 --gray --seed $s --tta || echo "EVAL FAIL $m $l $s"
    done
  done
done
echo "AB DONE"
