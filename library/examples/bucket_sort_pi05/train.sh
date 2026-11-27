#!/bin/bash
# train.sh fp32|bf16 DATA OUT [GPUS]
# The two physicalai recipes compared in the findings deck: pi05_base, effective batch 16, 12k steps,
# a weights-only checkpoint every 6k steps. fp32 needs ~71 GB per GPU at batch 8, so it runs on 2 GPUs.
# Each save writes up to three full-size files (stepstep=N, best, last): 9.4 GB each in bf16, 16.6 GB in fp32.
source "$(dirname "$0")/env.sh"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mode=$1 data=$2 out=$3
common=(--policy pi05 --data "$data" --out "$out" --max-steps 12000 --val-every 6000 --ckpt-every 6000
        --num-workers 8 --val-split 0.02 --limit-val 20)
case $mode in
  fp32) CUDA_VISIBLE_DEVICES=${4:-0,1} nohup $PY $S/train.py "${common[@]}" --batch-size 8 --devices 2 \
          --kw '{"dtype":"float32"}' > logs/$(basename "$out").log 2>&1 < /dev/null & ;;
  bf16) CUDA_VISIBLE_DEVICES=${4:-0} nohup $PY $S/train.py "${common[@]}" --batch-size 16 \
          > logs/$(basename "$out").log 2>&1 < /dev/null & ;;
  *) echo "usage: train.sh fp32|bf16 DATA OUT [GPUS]"; exit 1 ;;
esac
echo "training $mode -> $WORK/$out, log $WORK/logs/$(basename "$out").log"
