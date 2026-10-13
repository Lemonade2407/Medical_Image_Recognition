#!/bin/bash
# P1: longer training (150 ep + early stop, best-val rollback) to test whether the
# "Boundary harmful" / "Tversky hurts UNet++" findings are just underfitting.
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python

echo "===== P1-1: U-Net / AttU-Net x 3 losses x 3 seeds (150 ep, patience 15) ====="
for m in unet attunet; do
  for l in ce_dice tversky boundary; do
    for s in 0 1 2; do
      echo "--- P1 $m $l seed$s ---"
      $PY -u train_supervised.py --model $m --size 128 --gray --epochs 150 --seed $s \
          --workers 4 --loss $l --early-stop 15 --min-epochs 20 --tag p1_$m || echo "FAIL $m $l $s"
    done
  done
done

echo "===== P1-2: UNet++ x {ce_dice,tversky} x 3 seeds (needed for the Tversky claim) ====="
for l in ce_dice tversky; do
  for s in 0 1 2; do
    echo "--- P1 unetpp $l seed$s ---"
    $PY -u train_supervised.py --model unetpp --size 256 --rgb --epochs 150 --seed $s \
        --workers 4 --loss $l --early-stop 15 --min-epochs 20 --tag p1_unetpp || echo "FAIL unetpp $l $s"
  done
done

echo "=== summarize ==="
$PY -u summarize_p1.py
echo "P1 DONE"
