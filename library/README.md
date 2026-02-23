<p align="center">
  <img src="../docs/assets/physical_ai.png" alt="PhysicalAI Library" width="100%">
</p>

<div align="center">

**A library for training and deploying Vision-Language-Action policies for robotic imitation learning**

---

[Key Features](#key-features) •
[Installation](#installation) •
[Training](#training) •
[Benchmark](#benchmark) •
[Export](#export) •
[Inference](#inference) •
[Docs](docs/README.md)

</div>

---

# Introduction

PhysicalAI Library is a Python SDK for training, evaluating, and deploying Vision-Language-Action (VLA) policies. It provides implementations of imitation learning algorithms built on PyTorch Lightning, with a focus on robotic manipulation tasks. The library supports the full ML lifecycle: from training on demonstration data to deploying optimized models for real-time inference.

## Key Features

- Simple and modular API and CLI for training, inference, and benchmarking.
- Built on [Lightning](https://www.lightning.ai/) for reduced boilerplate and distributed training support.
- Export models to [OpenVINO](https://docs.openvino.ai/), ONNX, or Torch formats for accelerated inference.
- Benchmark policies on standardized environments like LIBERO and PushT.
- Unified inference API across all export backends.

## Supported Policies

| Policy       | Description                                                 | Paper                                                                          |
| ------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **ACT**      | Action Chunking with Transformers                           | [Zhao et al. 2023](https://arxiv.org/abs/2304.13705)                           |
| **SmolVLA**  | Lightweight vision-language-action model                    | [Cadene et al. 2024](https://huggingface.co/lerobot/smolvla_base)              |
| **Pi0**      | Physical Intelligence foundation model                      | [Black et al. 2024](https://www.physicalintelligence.company/download/pi0.pdf) |
| **GR00T N1** | Vision-language grounded policy                             | [Bjork et al. 2025](https://arxiv.org/abs/2503.14734)                          |
| **Pi0.5**    | Vision-Language-Action Model with Open-World Generalization | [Black et al. 2025](https://arxiv.org/pdf/2504.16054)                          |

# Installation

```bash
pip install physicalai-train
```

<details>
<summary><strong>Prerequisites</strong></summary>

PhysicalAI Library requires Python 3.12+.

FFMPEG is required as a dependency of LeRobot:

```bash
# Ubuntu
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg
```

</details>

<details>
<summary><strong>Install from Source (for development)</strong></summary>

```bash
git clone https://github.com/open-edge-platform/physical-ai-studio.git
cd physical-ai-studio/library

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv sync --all-extras
```

</details>

# Training

PhysicalAI supports both API and CLI-based training. Checkpoints are saved to `experiments/lightning_logs/` by default.

## API

```python test="skip" reason="requires dataset download"
from physicalai.data import LeRobotDataModule
from physicalai.policies import ACT
from physicalai.train import Trainer

# Initialize components
datamodule = LeRobotDataModule(repo_id="lerobot/aloha_sim_transfer_cube_human")
model = ACT()
trainer = Trainer(max_epochs=100)

# Train
trainer.fit(model=model, datamodule=datamodule)
```

## CLI

```bash
# Train with config file
physicalai fit --config configs/physicalai/act.yaml

# Train with CLI arguments
physicalai fit \
    --model physicalai.policies.ACT \
    --data physicalai.data.LeRobotDataModule \
    --data.repo_id lerobot/aloha_sim_transfer_cube_human

# Override config values
physicalai fit \
    --config configs/physicalai/act.yaml \
    --trainer.max_epochs 200 \
    --data.train_batch_size 64
```

# Benchmark

Evaluate trained policies on standardized simulation environments.

## API

```python test="skip" reason="requires checkpoint and libero"
from physicalai.benchmark import LiberoBenchmark
from physicalai.policies import ACT

# Load trained policy (path from training output)
policy = ACT.load_from_checkpoint("experiments/lightning_logs/version_0/checkpoints/last.ckpt")
policy.eval()

# Run benchmark
benchmark = LiberoBenchmark(task_suite="libero_10", num_episodes=20)
results = benchmark.evaluate(policy)

# View results
print(results.summary())
results.to_json("results.json")
```

## CLI

```bash
# Basic benchmark
physicalai benchmark \
    --benchmark physicalai.benchmark.LiberoBenchmark \
    --benchmark.task_suite libero_10 \
    --policy physicalai.policies.ACT \
    --ckpt_path ./checkpoints/model.ckpt

# With video recording
physicalai benchmark \
    --benchmark physicalai.benchmark.LiberoBenchmark \
    --benchmark.task_suite libero_10 \
    --benchmark.video_dir ./videos \
    --benchmark.record_mode failures \
    --policy physicalai.policies.ACT \
    --ckpt_path ./checkpoints/model.ckpt
```

# Export

Export trained policies to optimized formats for deployment.

## API

```python test="skip" reason="requires checkpoint"
from physicalai.policies import ACT

# Load and export
policy = ACT.load_from_checkpoint("checkpoints/model.ckpt")
policy.export("./exports", backend="openvino")
```

## CLI

```bash
# CLI coming soon - use Python API for now
# See API section above for export usage
```

### Supported Backends

| Backend      | Best For                     | Install                |
| ------------ | ---------------------------- | ---------------------- |
| OpenVINO     | Intel hardware (CPU/GPU/NPU) | `pip install openvino` |
| ONNX         | NVIDIA GPUs, cross-platform  | `pip install onnx`     |
| Torch Export | Edge/mobile devices          | Built-in               |

# Inference

Deploy exported models with a unified inference API.

## API

```python test="skip" reason="requires exported model and environment"
from physicalai.inference import InferenceModel

# Load exported model (auto-detects backend)
policy = InferenceModel.load("./exports")

# Run inference loop
obs, info = env.reset()
policy.reset()
done = False

while not done:
    action = policy.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

The inference API is consistent across all export backends, making it easy to switch between OpenVINO, ONNX, and Torch depending on your deployment target.

# Documentation

- [Getting Started](docs/getting-started/README.md) - Installation, quickstart, first benchmark, first deployment
- [How-To Guides](docs/how-to/README.md) - Goal-oriented guides for specific tasks
- [Explanation](docs/explanation/README.md) - Architecture and design documentation

# See Also

- [Main Repository](../README.md) - Project overview
- [Application](../application/) - GUI for data collection and training
