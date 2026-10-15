# LeRobot integration tests

Use the hardware-aware launcher from `library/` after installing `uv`:

```bash
uv run python scripts/run_lerobot_integration.py
```

The launcher prepares the matching `uv` extra and runs the focused native/Engine
PuShT training-parity test. In automatic mode it prefers a usable Intel XPU, then
a usable NVIDIA CUDA device. When GPU hardware is detected before a matching
PyTorch runtime is installed, it prepares that backend and verifies it before
running. CPU is selected only when no GPU hardware is detected and neither GPU
backend is usable. It sets `PHYSICALAI_LEROBOT_DEVICE` for pytest. If detected
GPU hardware remains unusable, the launcher stops with setup guidance instead of
silently running on CPU.

Override device selection with `--device xpu`, `--device cuda`, or `--device cpu`.
To pass a different pytest node or additional pytest options, put them after `--`:

```bash
uv run python scripts/run_lerobot_integration.py -- tests/integration/test_engine_lerobot_hf_training.py -q -s
```

Set `PHYSICALAI_LEROBOT_TEST_STEPS` to change the small parity run's update
count. The tests download the pinned public PuShT dataset on first use. The
launcher also supports `PHYSICALAI_LEROBOT_DEVICE` as an environment override;
`--device` takes precedence.
