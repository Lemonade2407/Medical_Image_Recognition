#!/bin/bash
# Apply the A group (postproc / TTA / threshold) on top of the P1 (150 ep) weights.
# Parallel (CPU-bound evals); skips finished ones so it can be re-run after TransUNet lands.
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python
RES=/mnt/workspace/busi_p0/results
P="${P:-4}"

cmds=()
add() {  # model loss size mode scales
  local m=$1 l=$2 sz=$3 mode=$4 scales=$5 s suf ck out
  for s in 0 1 2; do
    suf=""; [ "$l" != "ce_dice" ] && suf="_$l"
    ck="$RES/p1_${m}${suf}_seed${s}.pth"
    out="$RES/postproc_p1_${m}_s${sz}${mode}${suf}_seed${s}.json"
    [ -f "$ck" ] || continue
    [ -f "$out" ] && { echo "SKIP $(basename "$out")"; continue; }
    local extra=""; [ -n "$scales" ] && extra="--scales $scales"
    local flag="--gray"; [ "$mode" = "rgb" ] && flag="--rgb"
    cmds+=("$PY -u eval_postproc.py --ckpt $ck --model $m --size $sz $flag --seed $s --tta $extra --out-tag postproc_p1_${m}_s${sz}${mode}${suf} > $RES/p1alog_${m}${suf}_s${s}.log 2>&1")
  done
}

for m in unet attunet; do for l in ce_dice tversky boundary; do add "$m" "$l" 128 g ""; done; done
for l in ce_dice tversky; do add unetpp "$l" 256 rgb ""; done
for l in ce_dice tversky; do add transunet "$l" 256 rgb "1.0"; done

echo "to run: ${#cmds[@]}   parallelism=$P"
i=0
for c in "${cmds[@]}"; do
  bash -c "$c" &
  i=$((i + 1))
  if [ $((i % P)) -eq 0 ]; then wait; fi
done
wait
echo "=== summarize ==="
$PY -u summarize_p1a.py
echo "P1A DONE"
