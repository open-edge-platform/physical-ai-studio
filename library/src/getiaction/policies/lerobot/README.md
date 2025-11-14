# LeRobot Policies Module

PyTorch Lightning wrappers for
[LeRobot](https://github.com/huggingface/lerobot) robotics policies.

## Installation

```bash
uv pip install getiaction[lerobot]
```

## Quick Start

```python
from getiaction.policies.lerobot import ACT
from getiaction.train import Trainer

policy = ACT(dim_model=512, chunk_size=100)
trainer = Trainer(max_epochs=100)
trainer.fit(policy, datamodule)
```

## Available Policies

### Explicit Wrappers

- **ACT** - Action Chunking Transformer (full IDE support)
- **Diffusion** - Diffusion Policy (full IDE support)

### Universal Wrapper

- **LeRobotPolicy** - All 9 LeRobot policies via `policy_name`
  parameter
- **Convenience Aliases**: `VQBeT()`, `TDMPC()`, `SAC()`, `PI0()`,
  `PI05()`, `PI0Fast()`, `SmolVLA()`

## Features

- ✅ Verified output equivalence with native LeRobot
- ✅ Full PyTorch Lightning integration
- ✅ Thin wrapper pattern (no reimplementation)
- ✅ All LeRobot features accessible via `policy.lerobot_policy`

## Documentation

- **Design & Architecture**: [docs/design/policy/lerobot.md](../../../docs/design/policy/lerobot.md)
- **Usage Guide**: [docs/guides/lerobot.md](../../../docs/guides/lerobot.md)
- **LeRobot**: <https://github.com/huggingface/lerobot>
