#!/bin/bash
# train_lerobot.sh DATA OUT [GPUS]: LeRobot's pi05 trainer, fp32, same recipe as train.sh fp32.
# lerobot_train_ddp.py patches the SigLIP key names that transformers 5 renamed; without it LeRobot 0.6
# silently leaves pi05_base's vision encoder at random init (check the log for "All keys loaded").
source "$(dirname "$0")/env.sh"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} CUDA_VISIBLE_DEVICES=${3:-0,1}
nohup $PY -m accelerate.commands.launch --num_processes 2 --main_process_port 29511 $S/lerobot_train_ddp.py \
  --dataset.repo_id=local/$(basename "$1") --dataset.root="$1" --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=float32 --policy.gradient_checkpointing=true --policy.compile_model=false --policy.push_to_hub=false \
  --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=12000 --batch_size=8 --steps=12000 --save_freq=6000 \
  --log_freq=100 --num_workers=8 --wandb.enable=false --output_dir="$2" > logs/$(basename "$2").log 2>&1 < /dev/null &
echo "training lerobot fp32 -> $WORK/$2"
