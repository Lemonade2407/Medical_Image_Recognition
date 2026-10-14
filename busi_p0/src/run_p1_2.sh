#!/bin/bash
# P1-2 (SEQUENTIAL): UNet++ x {ce_dice, tversky} x 3 seeds, 256 RGB, 150 ep + early stop.
# Sequential because 3-way parallel was a net loss (~35 s/ep vs ~10 s/ep solo).
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python
RES=/mnt/workspace/busi_p0/results

for l in ce_dice tversky; do
  for s in 0 1 2; do
    suf=""; [ "$l" != "ce_dice" ] && suf="_$l"
    out="$RES/p1_unetpp${suf}_seed${s}.json"
    [ -f "$out" ] && { echo "SKIP $(basename "$out")"; continue; }
    echo "--- P1 unetpp $l seed$s ---"
    $PY -u train_supervised.py --model unetpp --size 256 --rgb --epochs 150 --seed $s \
        --workers 4 --loss $l --early-stop 15 --min-epochs 20 --tag p1_unetpp || echo "FAIL unetpp $l $s"
  done
done
echo "P1 DONE"
