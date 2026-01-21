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

## Key Features

- **End-to-End Pipeline** - From demonstration recording to robot deployment
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

#### Installation

```bash
# Backend
cd application/backend && uv sync
# UI
cd application/ui && npm install
```

### Library (Python/CLI)

```bash
pip install getiaction
```

<details open>
<summary>Training</summary>

```python
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

```python
from getiaction.benchmark import LiberoBenchmark

benchmark = LiberoBenchmark(task_suite="libero_10", num_episodes=20)
results = benchmark.evaluate(model)
print(f"Success rate: {results.success_rate:.1%}")
```

</details>

<details>
<summary>Export</summary>

```python
model.export("./policy", backend="openvino")
```

</details>

<details>
<summary>Inference</summary>

```python
from getiaction.inference import InferenceModel

policy = InferenceModel.load("./policy")
while not done:
    action = policy.select_action(observation)
    observation, reward, done, info = env.step(action)
```

</details>

<details>
<summary>CLI Usage</summary>

```bash
# Train
getiaction fit --config configs/getiaction/act.yaml

# Evaluate
getiaction benchmark --config configs/benchmark/libero.yaml --ckpt_path model.ckpt

# Export
getiaction export --ckpt_path model.ckpt --export_path ./policy --backend openvino
```

</details>

[Library Documentation →](./library/README.md)

## Documentation

| Resource                                     | Description                         |
| -------------------------------------------- | ----------------------------------- |
| [Library Docs](./library/docs/)              | API reference, guides, and examples |
| [Application Docs](./application/README.md)  | GUI setup and usage                 |
| [Developer Guide](./docs/developer_guide.md) | Contributing and development setup  |

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.
