<p align="center">
  <img src="docs/assets/physicalai.png" alt="Physical AI" width="100%">
</p>

<div align="center">

**Lightweight runtime for Physical AI inference and deployment**

[Key Features](#key-features) •
[Quick Start](#quick-start) •
[Documentation](#documentation) •
[Contributing](#contributing)

<!-- TODO: Add badges here -->
<!-- [![python](https://img.shields.io/badge/python-3.12%2B-green)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) -->

</div>

---

## What is Physical AI Runtime?

Physical AI Runtime is the lightweight deployment layer for Physical AI. It runs exported policies with a unified API and CLI, and provides camera/robot interfaces for real‑world execution.

## Key Features

- **Unified Inference API** - Load and run exported policies with `InferenceModel`
- **Runtime CLI** - `physicalai run/serve/validate` for deployment workflows
- **Hardware Interfaces** - Camera and robot interfaces with clean boundaries
- **Benchmarking** - NumPy‑only benchmarking runner and protocols
- **Format‑agnostic** - Works with exported models from physicalai-train, LeRobot, and custom frameworks

## Quick Start

### Install

```bash
pip install physicalai
```

### Inference (Python)

```python
from physicalai import InferenceModel

policy = InferenceModel("./exports/act_policy")
action = policy.select_action(observation)
```

### CLI

```bash
physicalai run --model ./exports/act_policy --robot robot.yaml
```

## Documentation

| Resource                                  | Description                         |
| ----------------------------------------- | ----------------------------------- |
| [Design Docs](./docs/design/README.md)    | Architecture and design proposals   |
| [Contributing](./CONTRIBUTING.md)         | Development setup and guidelines    |

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.
