# Temporary LeRobot Export POC Validation Scripts

> **Temporary directory:** Added to this draft PR so the POC can be reproduced and benchmarked
> on additional hardware (e.g. Intel Panther Lake / Core Ultra iGPU + NPU). Remove `tmp/lerobot_export_poc/`
> before finalizing the PR.

## Setup on a Fresh Machine (e.g. ASUS NUC Pro / Panther Lake)

```bash
git fetch origin
git checkout feat/lerobot-export-poc

# Sync the plugin environment (also pulls physicalai + physicalai-train)
uv sync --directory plugins/policies/lerobot --extra train --extra tests

# Verify OpenVINO devices visible on this machine (CPU, GPU, NPU)
uv run --directory plugins/policies/lerobot --no-sync python -c \
  "import openvino as ov; c = ov.Core(); print([(d, c.get_property(d, 'FULL_DEVICE_NAME')) for d in c.available_devices])"
```

## 1. Run Unit & Boundary Tests

```bash
uv run --directory plugins/policies/lerobot --no-sync pytest -q
uv run --directory plugins/policies/lerobot --no-sync pytest ../../../library/tests/unit/engine -q
```

## 2. Run Local Policy-Family Sweep (9+ Policies × `openvino` / `executorch_openvino`)

Exports each `openvino` bundle **once** and runs inference on every requested OpenVINO device
(default `--devices auto` tests all detected OpenVINO devices, e.g. `CPU`, `GPU`, `NPU`):

```bash
# Default: all 9 local policy families across all detected OpenVINO devices
uv run --directory plugins/policies/lerobot --no-sync python ../../../tmp/lerobot_export_poc/validate_local.py

# Target specific devices / policies (fast smoke test on Panther Lake CPU + iGPU + NPU)
uv run --directory plugins/policies/lerobot --no-sync python ../../../tmp/lerobot_export_poc/validate_local.py \
  --policies act diffusion vqbet tdmpc gaussian_actor \
  --devices CPU GPU NPU

# Reproduce the OpenVINO CPU GroupNorm SIGSEGV on diffusion / vqbet
uv run --directory plugins/policies/lerobot --no-sync python ../../../tmp/lerobot_export_poc/validate_local.py \
  --policies diffusion vqbet --group-norm --devices CPU
```

## 3. Run Pretrained Hugging Face Hub VLA Sweep

```bash
# Default cases: smolvla_base (ov, ov int8, executorch_ov), xvla-libero (ov, executorch_ov), pi05_libero (ov)
uv run --directory plugins/policies/lerobot --no-sync python ../../../tmp/lerobot_export_poc/validate_hub.py \
  --devices CPU GPU

# Fast SmolVLA + Pi0.5 INT4/INT8 benchmark on Panther Lake iGPU/CPU, keeping exported bundles on disk
uv run --directory plugins/policies/lerobot --no-sync python ../../../tmp/lerobot_export_poc/validate_hub.py \
  --cases smolvla_ov smolvla_ov_fp16 smolvla_ov_int8 smolvla_et pi05_ov pi05_ov_int4 \
  --devices CPU GPU \
  --keep-exports /tmp/lerobot_exports
```
