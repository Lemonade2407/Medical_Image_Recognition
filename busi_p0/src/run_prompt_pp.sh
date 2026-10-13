#!/bin/bash
# Stage-3: A-group (postproc/TTA/threshold) for the prompt methods, run in parallel.
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python
RES=/mnt/workspace/busi_p0/results

pids=()
for m in sam medsam bussam; do
  echo "--- prompt A $m (TTA) ---"
  $PY -u eval_prompt_postproc.py --method $m --tta > "$RES/prompt_pp_$m.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "PROMPT PP DONE"
