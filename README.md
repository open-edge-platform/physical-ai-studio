<p align="center">
  <img src="docs/assets/banner.png" alt="Geti Action" width="100%">
</p>

# Geti Action

Train and deploy Vision-Language-Action (VLA) models for robotic imitation learning.

## Overview

This monorepo provides two ways to work with VLA policies:

| Component                         | Description                                                  | Documentation                      |
| --------------------------------- | ------------------------------------------------------------ | ---------------------------------- |
| **[Library](./library/)**         | Python SDK for training, evaluating, and deploying policies  | [Library Docs](./library/docs/)    |
| **[Application](./application/)** | Studio app with GUI for data collection and model management | [Application Docs](./application/) |

**Library** is for developers who want programmatic control via Python or CLI. **Application** is for users who prefer a visual interface.

## Quick Start

### Library (Python/CLI)

```bash
cd library
uv sync --all-extras
source .venv/bin/activate

# Train a policy
getiaction fit --config configs/train.yaml

# Benchmark
getiaction benchmark --config configs/benchmark/libero.yaml
```

See the [Library README](./library/README.md) for full setup and usage.

### Application (GUI)

```bash
cd application
# Start backend and frontend (see Application README for details)
```

See the [Application README](./application/README.md) for setup instructions.

## Documentation

- **[Library Documentation](./library/docs/)** - Guides, API reference, and design docs
- **[Developer Guide](./docs/developer_guide.md)** - Development setup and contribution workflow

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on how to contribute.
