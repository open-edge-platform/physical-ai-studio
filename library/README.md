# Geti Action

PyTorch Lightning-based framework for training robotic action policies.

## Features

- ✅ **Lightning Integration** - Built on PyTorch Lightning for scalable training
- ✅ **LeRobot Policies** - Seamless integration with HuggingFace LeRobot
- ✅ **LightningCLI** - Configuration-driven training workflow
- ✅ **Verified Equivalence** - LeRobot wrappers produce identical outputs

## Installation

### Prerequisites

Before you begin, ensure you have FFMPEG installed.
This is a dependency for the LeRobot library.

To install FFMPEG on an Ubuntu system:

```bash
sudo apt-get install -y ffmpeg
```

Create a new python environment and install the development requirements:

```bash
uv venv .act
source .act/bin/activate
uv sync --all-extras --active
```

## Quick Start

### Using LeRobot Policies

```python
from getiaction.policies.lerobot import ACT
import lightning as L

# Create policy
policy = ACT(
    dim_model=512,
    chunk_size=100,
    n_action_steps=100,
)

# Train
trainer = L.Trainer(max_epochs=100)
trainer.fit(policy, datamodule)
```

### Using LightningCLI

```bash
getiaction fit --config configs/lerobot_act.yaml
```

## Documentation

- **User Guides**: [docs/guides/](docs/guides/)
  - [LeRobot Usage Guide](docs/guides/lerobot.md)
- **Design Docs**: [docs/design/](docs/design/)
  - [LeRobot Integration](docs/design/policy/lerobot.md)
- **Module READMEs**: See README.md files in each module
