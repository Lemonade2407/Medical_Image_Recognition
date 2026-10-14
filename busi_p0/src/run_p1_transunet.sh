#!/bin/bash
# P1 for TransUNet: 150 ep + early stop (sequential; TransUNet saturates the GPU).
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python
RES=/mnt/workspace/busi_p0/results

for l in ce_dice tversky; do
  for s in 0 1 2; do
    suf=""; [ "$l" != "ce_dice" ] && suf="_$l"
    out="$RES/p1_transunet${suf}_seed${s}.json"
    [ -f "$out" ] && { echo "SKIP $(basename "$out")"; continue; }
    echo "--- P1 transunet $l seed$s ---"
    $PY -u train_supervised.py --model transunet --size 256 --rgb --epochs 150 --seed $s \
        --workers 4 --loss $l --early-stop 15 --min-epochs 20 --tag p1_transunet || echo "FAIL transunet $l $s"
  done
done
echo "P1 TRANSUNET DONE"
