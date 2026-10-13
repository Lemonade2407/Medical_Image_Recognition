#!/bin/bash
# Wait for all 4 supervised viz checkpoints, then render the qualitative figures.
cd /mnt/workspace/busi_p0
PY=/mnt/workspace/busi_p0/.venv/bin/python
for i in $(seq 1 80); do
  if ls results/viz_unet_seed0.pth results/viz_attunet_seed0.pth results/viz_unetpp_seed0.pth results/viz_transunet_seed0.pth >/dev/null 2>&1; then
    echo "all ckpts ready $(date)"; break
  fi
  sleep 30
done
$PY -u src/make_qualitative.py 2>&1 | tee results/viz.log
echo "VIZ DONE $(date)"
