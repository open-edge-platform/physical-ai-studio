<p align="center">
  <img src="../docs/assets/banner_library.png" alt="Geti Action Library" width="100%">
</p>

# Geti Action Library

Python SDK for training, evaluating, and deploying Vision-Language-Action policies.

## Features

- **Train** imitation learning policies from demonstration data
- **Benchmark** policies on standardized environments (LIBERO, PushT)
- **Export** models to OpenVINO, ONNX, or Torch formats
- **Deploy** with a unified inference API

## Installation

### Prerequisites

FFMPEG is required (dependency of LeRobot):

```bash
# Ubuntu
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg
```

### Install

```bash
cd library
uv venv
source .venv/bin/activate
uv sync --all-extras
```

## Quick Start

### Train a Policy

```bash
# Using a config file
getiaction fit --config configs/getiaction/act.yaml

# Or specify components directly
getiaction fit \
    --model.class_path getiaction.policies.ACT \
    --data.class_path getiaction.data.LeRobotDataModule \
    --data.repo_id lerobot/pusht
```

### Benchmark

```bash
getiaction benchmark \
    --benchmark getiaction.benchmark.LiberoBenchmark \
    --policy getiaction.policies.ACT \
    --ckpt_path ./checkpoints/model.ckpt
```

### Export and Deploy

```python
from getiaction.policies import ACT
from getiaction.inference import InferenceModel

# Export
policy = ACT.load_from_checkpoint("checkpoints/model.ckpt")
policy.export("./exports", backend="openvino")

# Deploy
model = InferenceModel.load("./exports")
action = model.select_action(observation)
```

## Documentation

- **[User Guides](docs/guides/)** - CLI, benchmarking, export/inference
- **[Design Docs](docs/design/)** - Architecture and implementation details

## See Also

- [Main Repository](../README.md) - Project overview
- [Application](../application/) - GUI for data collection and training
