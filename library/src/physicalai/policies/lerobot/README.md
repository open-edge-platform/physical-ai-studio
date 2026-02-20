# LeRobot Policies Module

PyTorch Lightning wrappers for
[LeRobot](https://github.com/huggingface/lerobot) robotics policies.

## Installation

```bash
# Base installation (ACT, Diffusion, VQBeT, TDMPC, SAC)
pip install physicalai-train

# With Groot (NVIDIA GR00T-N1) support
pip install physicalai-train[groot]

# Everything
pip install physicalai-train[all]
```

> **Note**: Groot has heavy dependencies including transformers, flash-attn,
> and peft. Only install if needed.

## Quick Start

```python
from physicalai.policies.lerobot import ACT
from physicalai.train import Trainer

policy = ACT(dim_model=512, chunk_size=100)
trainer = Trainer(max_epochs=100)
trainer.fit(policy, datamodule)
```

## Available Policies

### Explicit Wrappers (Full IDE Support)

- **ACT** - Action Chunking Transformer
- **Diffusion** - Diffusion Policy
- **Groot** - NVIDIA GR00T-N1 Foundation Model

### Universal Wrapper

- **LeRobotPolicy** - All LeRobot policies via `policy_name` parameter
- **Convenience Aliases**: `VQBeT()`, `TDMPC()`, `SAC()`, `PI0()`,
  `PI05()`, `PI0Fast()`, `SmolVLA()`

### Supported Policies

| Policy      | Type      | Description                          |
| ----------- | --------- | ------------------------------------ |
| `act`       | Explicit  | Action Chunking Transformer          |
| `diffusion` | Explicit  | Diffusion Policy                     |
| `groot`     | Explicit  | NVIDIA GR00T-N1 VLA Foundation Model |
| `vqbet`     | Universal | VQ-BeT (VQ-VAE Behavior Transformer) |
| `tdmpc`     | Universal | TD-MPC (Temporal Difference MPC)     |
| `sac`       | Universal | Soft Actor-Critic                    |
| `pi0`       | Universal | Vision-Language Policy               |
| `pi05`      | Universal | PI0.5 (Improved PI0)                 |
| `pi0fast`   | Universal | Fast Inference PI0                   |
| `smolvla`   | Universal | Small Vision-Language-Action         |

## Features

- ✅ Verified output equivalence with native LeRobot
- ✅ Full PyTorch Lightning integration
- ✅ Thin wrapper pattern (no reimplementation)
- ✅ All LeRobot features accessible via `policy.lerobot_policy`
- ✅ Support for VLA (Vision-Language-Action) models

## Examples

### Using Groot (VLA Foundation Model)

```python
from physicalai.policies.lerobot import Groot
from physicalai.data.lerobot import LeRobotDataModule
from physicalai.train import Trainer

# Create Groot policy with fine-tuning settings
policy = Groot(
    chunk_size=50,
    n_action_steps=50,
    tune_projector=True,
    tune_diffusion_model=True,
    lora_rank=16,  # Enable LoRA fine-tuning
)

# Create datamodule
datamodule = LeRobotDataModule(
    repo_id="lerobot/pusht",
    train_batch_size=8,
)

# Train
trainer = Trainer(max_epochs=100)
trainer.fit(policy, datamodule)
```

### Using Universal Wrapper

```python
from physicalai.policies.lerobot import LeRobotPolicy

# Use any LeRobot policy by name
policy = LeRobotPolicy(
    policy_name="vqbet",
    learning_rate=1e-4,
)
```

## Documentation

- **Design & Architecture**: [docs/design/policy/lerobot.md](../../../docs/design/policy/lerobot.md)
- **Usage Guide**: [docs/guides/lerobot.md](../../../docs/guides/lerobot.md)
- **LeRobot**: <https://github.com/huggingface/lerobot>
