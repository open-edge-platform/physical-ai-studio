#!/bin/bash
# demo.sh stop | start POLICY CKPT [GPU]: interactive language demo on port 8090 (3D viewer on 9090).
# POLICY is pi05 for a physicalai checkpoint, or lerobot for a lerobot-train pretrained_model dir.
source "$(dirname "$0")/env.sh"
pkill -f "$S/demo_server.py"; sleep 3; pkill -9 -f "$S/demo_server.py"
[ "$1" = "start" ] || exit 0
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${4:-0} MUJOCO_EGL_DEVICE_ID=${4:-0} nohup $PY $S/demo_server.py --policy "$2" --ckpt "$3" \
  --port 8090 --viser-port 9090 > logs/demo.log 2>&1 < /dev/null &
echo "demo starting: http://127.0.0.1:8090 (log $WORK/logs/demo.log); forward the port with ssh -L 8090:127.0.0.1:8090"
