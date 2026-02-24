<p align="center">
  <img src="docs/assets/physical_ai_studio.png" alt="Physical AI Studio" width="100%">
</p>

<div align="center">

**Train and deploy Vision-Language-Action (VLA) models for robotic imitation learning**

[Key Features](#key-features) •
[Quick Start](#quick-start) •
[Documentation](#documentation) •
[Contributing](#contributing)

<!-- TODO: Add badges here -->
<!-- [![python](https://img.shields.io/badge/python-3.10%2B-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) -->

</div>

---

## What is Physical AI Studio?

Physical AI Studio is an end-to-end framework for teaching robots to perform tasks through imitation learning from human demonstrations.

## Key Features

- **End-to-End Pipeline** - From demonstration recording to robot deployment
- **State-of-the-Art Policies** - Native policy implementations such as [ACT](https://arxiv.org/abs/2304.13705), [Pi0](https://www.physicalintelligence.company/download/pi0.pdf), [SmolVLA](https://huggingface.co/lerobot/smolvla_base), [GR00T](https://arxiv.org/abs/2503.14734) and [Pi0.5](https://arxiv.org/pdf/2504.16054), plus full [LeRobot](https://github.com/huggingface/lerobot) policy zoo
- **Flexible Interface** - Use Python API, CLI, or GUI
- **Production Export** - Deploy to [OpenVINO](https://docs.openvino.ai/), [ONNX](https://onnx.ai/), or [Torch](https://docs.pytorch.org/executorch/stable/index.html) for any hardware
- **Standardized Benchmarks** - Evaluate on benchmarks such as [LIBERO](https://libero-project.github.io/) and [PushT](https://diffusion-policy.cs.columbia.edu/)
- **Built on Lightning** - [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for distributed training, mixed precision, and more

## Quick Start

### Application (GUI)

For users who prefer a visual interface for end-to-end workflow:

<!-- markdownlint-disable MD033 -->
<p align="center">
  <img src="docs/assets/application.gif" alt="Application demo" width="100%">
</p>
<!-- markdownlint-enable MD033 -->

[Application Documentation →](./application/README.md)

#### Installation & Running

```bash
# Clone the repository
git clone https://github.com/open-edge-platform/physical-ai-studio.git
cd physical-ai-studio

# Install and run backend
cd application/backend && uv sync
source .venv/bin/activate
uvicorn src.main:app --reload

# In a new terminal: install and run UI
cd application/ui 
nvm use
npm install
# Fetch the api from the backend and build the types and start the frontend.
npm run build:api:download && npm run build:api && npm run start
```

Open http://localhost:3000 in your browser.

##### Docker

Run the full application (backend + UI) in a single container:

```bash
cd docker
cp .env.example .env
docker compose up
```

Application runs at http://localhost:7860. See the [Docker README](./docker/readme.md) for
hardware configuration (Intel XPU, NVIDIA CUDA) and device setup.

### Library (Python/CLI)

For programmatic control over training, benchmarking, and deployment with both API and CLI

```bash
pip install physicalai-train
```

<details open>
<summary>Training</summary>

```python test="skip" reason="requires dataset download"
from physicalai.data import LeRobotDataModule
from physicalai.policies import ACT
from physicalai.train import Trainer

datamodule = LeRobotDataModule(repo_id="lerobot/aloha_sim_transfer_cube_human")
model = ACT()
trainer = Trainer(max_epochs=100)
trainer.fit(model=model, datamodule=datamodule)
```

</details>

<details>
<summary>Benchmark</summary>

```python test="skip" reason="requires checkpoint and libero"
from physicalai.benchmark import LiberoBenchmark
from physicalai.policies import ACT

policy = ACT.load_from_checkpoint("experiments/lightning_logs/version_0/checkpoints/last.ckpt")
benchmark = LiberoBenchmark(task_suite="libero_10", num_episodes=20)
results = benchmark.evaluate(policy)
print(f"Success rate: {results.aggregate_success_rate:.1f}%")
```

</details>

<details>
<summary>Export</summary>

```python test="skip" reason="requires checkpoint"
from physicalai.export import get_available_backends
from physicalai.policies import ACT

# See available backends
print(get_available_backends())  # ['onnx', 'openvino', 'torch', 'torch_export_ir']

# Export to OpenVINO
policy = ACT.load_from_checkpoint("experiments/lightning_logs/version_0/checkpoints/last.ckpt")
policy.export("./policy", backend="openvino")
```

</details>

<details>
<summary>Inference</summary>

```python test="skip" reason="requires exported model and environment"
from physicalai.inference import InferenceModel

policy = InferenceModel.load("./policy")
obs, info = env.reset()
done = False

while not done:
    action = policy.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

</details>

<details>
<summary>CLI Usage</summary>

```bash
# Train
physicalai fit --config configs/physicalai/act.yaml

# Evaluate
physicalai benchmark --config configs/benchmark/libero.yaml --ckpt_path model.ckpt

# Export (Python API only - CLI coming soon)
# Use: policy.export("./policy", backend="openvino")
```

</details>

[Library Documentation →](./library/README.md)

## Documentation

| Resource                                    | Description                         |
| ------------------------------------------- | ----------------------------------- |
| [Library Docs](./library/README.md)         | API reference, guides, and examples |
| [Application Docs](./application/README.md) | GUI setup and usage                 |
| [Contributing](./CONTRIBUTING.md)           | Contributing and development setup  |

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.
