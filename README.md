<p align="center">
  <img src="docs/assets/banner.png" alt="Geti Action" width="100%">
</p>

<div align="center">

**Train and deploy Vision-Language-Action (VLA) models for robotic imitation learning**

</div>

---

<!-- TODO: Add application demo GIF here -->
<!-- Recommended size: 800-1000px wide, 10-15 seconds -->
<!-- Shows: Data collection → Training → Deployment workflow -->
<!--
<p align="center">
  <img src="docs/assets/demo-app.gif" alt="Geti Action Application Demo" width="800">
</p>
-->

## What is Geti Action?

Geti Action enables you to train robotic policies through imitation learning. Record demonstrations of a task, train a policy, and deploy it to your robot. The library supports the full ML lifecycle: from data collection to real-time inference on edge devices.

<p align="center">
  <img src="docs/assets/architecture.png" alt="Geti Action Architecture" width="700">
</p>

> **Architecture**: Demonstrations → Training (Library/Application) → Optimized Model → Robot Deployment

<!-- TODO: Replace with actual architecture diagram -->
<!-- Recommended: Simple flowchart showing data → train → export → deploy -->

## Choose Your Workflow

Geti Action provides two interfaces for different use cases:

<table>
<tr>
<td width="50%" valign="top">

### Library (Python/CLI)

**For developers who want programmatic control**

<!-- TODO: Add terminal recording GIF -->
<!-- Shows: getiaction fit command with training progress -->
<!--
<img src="docs/assets/demo-cli.gif" alt="CLI Demo" width="100%">
-->

```bash
# Install
pip install getiaction

# Train
getiaction fit --config configs/getiaction/act.yaml

# Benchmark
getiaction benchmark \
    --benchmark LiberoBenchmark \
    --policy ACT \
    --ckpt_path model.ckpt
```

[Library Documentation →](./library/README.md)

</td>
<td width="50%" valign="top">

### Application (GUI)

**For users who prefer a visual interface**

<!-- TODO: Add application screenshot -->
<!-- Shows: Training dashboard or data collection interface -->
<!--
<img src="docs/assets/screenshot-app.png" alt="Application UI" width="100%">
-->

```bash
# Start application
cd application/backend && ./run.sh
cd application/ui && npm start
```

- Record demonstrations with visual feedback
- Monitor training progress in real-time
- Deploy models with one click

[Application Documentation →](./application/README.md)

</td>
</tr>
</table>

## Key Features

- **Simple API & CLI** - Train policies with minimal boilerplate using PyTorch Lightning
- **Multiple Export Formats** - Deploy to OpenVINO, ONNX, or Torch for different hardware targets
- **Standardized Benchmarks** - Evaluate on LIBERO and PushT simulation environments
- **Unified Inference** - Consistent API across all export backends
- **Production Ready** - Optimized models for real-time inference on edge devices

## Installation

### Library

```bash
pip install getiaction
```

<details>
<summary>See advanced installation options</summary>

```bash
# Install from source
git clone https://github.com/open-edge-platform/geti-action.git
cd geti-action/library
uv sync --all-extras
```

</details>

### Application

See [Application README](./application/README.md) for setup instructions.

## Quick Start

### Train a Policy

```python
from getiaction.data import LeRobotDataModule
from getiaction.policies import ACT
from getiaction.train import Trainer

# Load dataset and train
datamodule = LeRobotDataModule(repo_id="lerobot/aloha_sim_transfer_cube_human")
model = ACT()
trainer = Trainer(max_epochs=100)
trainer.fit(model=model, datamodule=datamodule)
```

### Benchmark Performance

```python
from getiaction.benchmark import LiberoBenchmark

benchmark = LiberoBenchmark(task_suite="libero_10", num_episodes=20)
results = benchmark.evaluate(model)
print(results.summary())
```

### Export and Deploy

```python
# Export to OpenVINO
model.export("./exports", backend="openvino")

# Deploy
from getiaction.inference import InferenceModel
policy = InferenceModel.load("./exports")
action = policy.select_action(observation)
```

## Documentation

- **[Library Documentation](./library/docs/)** - User guides, API reference, and design docs
- **[Developer Guide](./docs/developer_guide.md)** - Development setup and contribution workflow

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## Creating Demo Assets

If you're a contributor and want to help create the demo assets:

**Application Demo GIF** (`docs/assets/demo-app.gif`):
- Record 10-15 second workflow: data collection → training → deployment
- Tools: [LICEcap](https://www.cockos.com/licecap/), [Kap](https://getkap.co/)
- Size: 800-1000px wide

**CLI Demo GIF** (`docs/assets/demo-cli.gif`):
- Record training command with progress output
- Tools: [asciinema](https://asciinema.org/), [terminalizer](https://terminalizer.com/)
- Size: 600-800px wide

**Architecture Diagram** (`docs/assets/architecture.png`):
- Simple flowchart: Data → Train → Export → Deploy
- Tools: [Excalidraw](https://excalidraw.com/), draw.io, or mermaid
