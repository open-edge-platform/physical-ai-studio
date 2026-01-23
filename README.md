<p align="center">
  <img src="docs/assets/banner.png" alt="Geti Action" width="100%">
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

## What is Geti Action?

Geti Action helps you teach robots to perform tasks by learning from human demonstrations. Instead of manually programming every robot movement, you:

1. **Record** demonstrations of a task (e.g., picking up objects)
2. **Train** a policy on those observations
3. **Deploy** the trained policy to your robot

This approach is called **imitation learning** - the robot learns by imitating what you showed it.

## Key Features

- **End-to-End Pipeline** - From demonstration recording to robot deployment
- **State-of-the-Art Policies** - Native ACT, Pi0, SmolVLA, GR00T implementations plus full LeRobot policy zoo
- **Flexible Interface** - Use Python API, CLI, or GUI
- **Production Export** - Deploy to OpenVINO, ONNX, or Torch for any hardware
- **Standardized Benchmarks** - Evaluate on benchmarks such as LIBERO and PushT
- **Built on Lightning** - Distributed training, mixed precision, and more

## Quick Start

### Application (GUI)

For users who prefer a visual interface for end-to-end workflow:

- Visual demonstration recording
- Real-time training monitoring
- One-click model deployment

<!-- markdownlint-disable MD033 -->
<p align="center">
  <img src="docs/assets/application.gif" alt="Application demo" width="100%">
</p>
<!-- markdownlint-enable MD033 -->

[Application Documentation →](./application/README.md)

#### Installation & Running

```bash
# Clone the repository
git clone https://github.com/open-edge-platform/geti-action.git
cd geti-action

# Install and run backend
cd application/backend && uv sync
source .venv/bin/activate
uvicorn src.main:app --reload

# In a new terminal: install and run UI
cd application/ui && npm install
npm run start
```

Open http://localhost:3000 in your browser.

### Library (Python/CLI)

For programmatic control over training, benchmarking, and deployment:

- Full Python API for scripting and automation
- CLI for quick experiments
- Integrate into existing ML pipelines

```bash
pip install getiaction
```

<details open>
<summary>Training</summary>

```python test="skip" reason="requires dataset download"
from getiaction.data import LeRobotDataModule
from getiaction.policies import ACT
from getiaction.train import Trainer

datamodule = LeRobotDataModule(repo_id="lerobot/aloha_sim_transfer_cube_human")
model = ACT()
trainer = Trainer(max_epochs=100)
trainer.fit(model=model, datamodule=datamodule)
```

</details>

<details>
<summary>Benchmark</summary>

```python test="skip" reason="requires checkpoint and libero"
from getiaction.benchmark import LiberoBenchmark
from getiaction.policies import ACT

policy = ACT.load_from_checkpoint("experiments/lightning_logs/version_0/checkpoints/last.ckpt")
benchmark = LiberoBenchmark(task_suite="libero_10", num_episodes=20)
results = benchmark.evaluate(policy)
print(f"Success rate: {results.aggregate_success_rate:.1f}%")
```

</details>

<details>
<summary>Export</summary>

```python test="skip" reason="requires checkpoint"
from getiaction.export import get_available_backends
from getiaction.policies import ACT

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
from getiaction.inference import InferenceModel

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
getiaction fit --config configs/getiaction/act.yaml

# Evaluate
getiaction benchmark --config configs/benchmark/libero.yaml --ckpt_path model.ckpt

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
