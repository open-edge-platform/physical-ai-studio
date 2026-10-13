# Physical AI LeRobot Policy Plugin

This standalone plugin connects LeRobot v0.6.0 to the Physical AI framework Engine and Runtime inference adapter registries.

`pip install 'physicalai-train[lerobot]'` installs this plugin and LeRobot's
training requirements. For CPU PyTorch, use
`pip install 'physicalai-train[cpu,lerobot]'`; choose the matching hardware extra
for your environment. The plugin's `train` extra installs
LeRobot's `[training]` requirements, including Accelerate and W&B, plus Diffusers
for native Diffusion-policy training. Install `physicalai` and
`physicalai-lerobot-policy-plugin` directly when only the native policy loader is
needed. The plugin does not depend on `physicalai-train`, so the deployment
installation has no dependency cycle.

The training framework delegates to LeRobot's `lerobot-train` entry point and native `TrainPipelineConfig`. The Runtime adapter loads the native `pretrained_model/` files from a Runtime policy package and forwards episode resets to LeRobot's policy and processor pipelines.

## Development Checks

From `library/`, run the pinned public-dataset training smoke tests with:

```bash
PHYSICALAI_RUN_LEROBOT_INTEGRATION=1 \
HF_LEROBOT_HOME=/path/to/lerobot-cache \
uv run --extra cpu --extra tests --extra lerobot pytest tests/integration/test_engine_lerobot_hf_training.py -s
```

They default to one CPU update per case. Set `PHYSICALAI_LEROBOT_TEST_STEPS=100` for a longer run and `PHYSICALAI_LEROBOT_DEVICE=cuda`, `xpu`, or `cpu` to select the device. An unavailable requested accelerator skips with a reason. Each Engine run records finite per-step metrics and prints/saves its trajectory; the native-versus-Engine ACT regression compares resolved configs, losses, and final safetensors.

Pretrained ACT and Diffusion action parity is separately gated because it may download large Hub checkpoints:

```bash
PHYSICALAI_RUN_LEROBOT_PRETRAINED_INFERENCE=1 \
PHYSICALAI_LEROBOT_DEVICE=cpu \
HF_LEROBOT_HOME=/path/to/lerobot-cache \
uv run --extra cpu --extra tests --extra lerobot pytest tests/integration/test_engine_lerobot_pretrained_inference_parity.py -s
```

Those tests pin model and dataset commit SHAs, verify matching input shapes, package local native checkpoints with LeRobot processor pipelines, and compare action sequences before and after reset. Hub downloads are limited to safetensors and JSON/text/Markdown files.
