# [EXPERIMENTAL] VLA Evaluation Harness

In this section a model server to allow users to benchmark PhysicalAI policies on popular benchmarks.

## Installation

### vla-evaluation-harness

Please follow installation instructions for AllenAI's [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness).

It requires a docker installation.

### PhysicalAI

This model server either works with [PhysicalAI](https://github.com/openvinotoolkit/physicalai) inference framework or [PhysicalAI-Studio](https://github.com/open-edge-platform/physical-ai-studio).

Please have either installed.

## Examples

Run these from within `library/benchmarks/vla-evaluation-harness`.

### Smoke test

```bash
vla-eval test -c configs/pi05_libero_policy.yaml
```

### Run Pi05 Libero Policy

```bash
vla-eval serve --config configs/pi05_libero_policy.yaml
```

### API Version

Instead of using configs, we can also subclass the harness and implement a custom `Pi05LiberoServer`.

```bash
python model_servers/pi05_libero.py
# or with args

python model_servers/pi05_libero.py --port 8000 --pretrained_name_or_path lerobot/pi05_libero_finetuned_v044 --device cuda
```
