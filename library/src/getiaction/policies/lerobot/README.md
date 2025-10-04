# LeRobot Policies Module

PyTorch Lightning wrappers for [LeRobot](https://github.com/huggingface/lerobot) robotics policies.

## Installation

```bash
pip install getiaction[lerobot]
```

## Quick Start

```python
from getiaction.policies.lerobot import ACT
import lightning as L

policy = ACT(dim_model=512, chunk_size=100)
trainer = L.Trainer(max_epochs=100)
trainer.fit(policy, datamodule)
```

## Available Policies

### Explicit Wrappers

- **ACT** - Action Chunking Transformer (full IDE support)

### Universal Wrapper

- **LeRobotPolicy** - All 9 LeRobot policies via `policy_name` parameter
- **Convenience Aliases**: `Diffusion()`, `VQBeT()`, `TDMPC()`

## Features

- ✅ Verified output equivalence with native LeRobot
- ✅ Full PyTorch Lightning integration
- ✅ Thin wrapper pattern (no reimplementation)
- ✅ All LeRobot features accessible via `policy.lerobot_policy`

## Documentation

- **Design & Architecture**: [docs/design/policy/lerobot.md](../../../docs/design/policy/lerobot.md)
- **Usage Guide**: [docs/guides/lerobot.md](../../../docs/guides/lerobot.md)
- **LeRobot**: https://github.com/huggingface/lerobot

## Testing

```bash
pytest tests/test_lerobot_act.py -v
pytest tests/test_lerobot_universal.py -v
```
