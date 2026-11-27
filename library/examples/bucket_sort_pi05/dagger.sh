#!/bin/bash
# dagger.sh NAME CKPT WORKERS EPISODES_PER_WORKER SEED_BASE GPU
# The policy at CKPT drives for 2-25 s, then the IK demonstrator takes over; only the takeover is recorded.
source "$(dirname "$0")/env.sh"
name=$1 ckpt=$2 n=$3 eps=$4 base=$5 gpu=${6:-0}
mkdir -p data/$name logs/$name
for k in $(seq 0 $((n - 1))); do
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu OMP_NUM_THREADS=4 nohup $PY $S/gen_dagger.py \
    --out data/$name/w$k --ckpt $ckpt --episodes $eps --seed0 $((base + k * 100000)) > logs/$name/w$k.log 2>&1 < /dev/null &
done
echo "started $n DAgger workers into $WORK/data/$name"
