#!/bin/bash
# generate.sh NAME WORKERS EPISODES_PER_WORKER SEED_BASE RECOVERY_FRAC
# Record IK-demonstrator episodes into $WORK/data/NAME/w*, one headless worker per process.
# Workers render on the GPUs in $GPUS (MuJoCo picks its render GPU from MUJOCO_EGL_DEVICE_ID).
source "$(dirname "$0")/env.sh"
read -ra gpus <<< "${GPUS:-0 1 3}"
mkdir -p data/$1 logs/$1
for k in $(seq 0 $(($2 - 1))); do
  g=${gpus[$((k % ${#gpus[@]}))]}
  MUJOCO_EGL_DEVICE_ID=$g CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 nohup $PY $S/gen_bucket_sort.py --out data/$1/w$k \
    --episodes $3 --seed0 $(($4 + k * 100000)) --recovery-frac $5 > logs/$1/w$k.log 2>&1 < /dev/null &
done
echo "started $2 workers into $WORK/data/$1"
