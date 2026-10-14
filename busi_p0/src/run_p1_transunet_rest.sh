#!/bin/bash
# TransUNet P1 remainder (tversky). Launched via setsid so it survives session reaping.
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python
RES=/mnt/workspace/busi_p0/results

for s in 0 1 2; do
  out="$RES/p1_transunet_tversky_seed${s}.json"
  [ -f "$out" ] && { echo "SKIP $(basename "$out")"; continue; }
  echo "--- P1 transunet tversky seed$s ---"
  $PY -u train_supervised.py --model transunet --size 256 --rgb --epochs 150 --seed $s \
      --workers 4 --loss tversky --early-stop 15 --min-epochs 20 --tag p1_transunet \
      || echo "FAIL transunet tversky $s"
done
echo "P1 TRANSUNET DONE"
