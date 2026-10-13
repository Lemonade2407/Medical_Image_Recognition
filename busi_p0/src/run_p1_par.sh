#!/bin/bash
# P1 (parallel): longer training to rule out underfitting. GPU is ~40% utilised with
# one job, so run several trainings concurrently. Skips already-finished runs.
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python
RES=/mnt/workspace/busi_p0/results
W="${W:-3}"     # DataLoader workers per job
P="${P:-4}"     # concurrent jobs (128-gray phase)
Q="${Q:-3}"     # concurrent jobs (256-RGB UNet++ phase)

run_pool() {  # $1 = parallelism ; reads commands from stdin
  local max=$1 n=0
  while IFS= read -r c; do
    [ -z "$c" ] && continue
    bash -c "$c" &
    n=$((n + 1))
    if [ $((n % max)) -eq 0 ]; then wait; fi
  done
  wait
}

echo "===== P1-1: U-Net / AttU-Net x 3 losses x 3 seeds (150 ep, patience 15) ====="
{
  for m in unet attunet; do
    for l in ce_dice tversky boundary; do
      for s in 0 1 2; do
        suf=""; [ "$l" != "ce_dice" ] && suf="_$l"
        out="$RES/p1_${m}${suf}_seed${s}.json"
        [ -f "$out" ] && { echo "SKIP $(basename "$out")" >&2; continue; }
        echo "$PY -u train_supervised.py --model $m --size 128 --gray --epochs 150 --seed $s --workers $W --loss $l --early-stop 15 --min-epochs 20 --tag p1_$m > $RES/p1log_${m}${suf}_s${s}.log 2>&1"
      done
    done
  done
} | run_pool "$P"

echo "===== P1-2: UNet++ x {ce_dice,tversky} x 3 seeds (256 RGB) ====="
{
  for l in ce_dice tversky; do
    for s in 0 1 2; do
      suf=""; [ "$l" != "ce_dice" ] && suf="_$l"
      out="$RES/p1_unetpp${suf}_seed${s}.json"
      [ -f "$out" ] && { echo "SKIP $(basename "$out")" >&2; continue; }
      echo "$PY -u train_supervised.py --model unetpp --size 256 --rgb --epochs 150 --seed $s --workers $W --loss $l --early-stop 15 --min-epochs 20 --tag p1_unetpp > $RES/p1log_unetpp${suf}_s${s}.log 2>&1"
    done
  done
} | run_pool "$Q"

echo "=== summarize ==="
$PY -u summarize_p1.py
echo "P1 DONE"
