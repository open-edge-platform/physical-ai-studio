# LeRobot Policies Module

PyTorch Lightning wrappers for
[LeRobot](https://github.com/huggingface/lerobot) robotics policies.

## Installation

```bash
# Base installation (ACT, Diffusion, SmolVLA, PI0/PI05/PI0Fast, XVLA)
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

### Named Wrappers (Full IDE Support)

One-line factories for the curated set of LeRobot policies physicalai
explicitly supports: `ACT`, `Diffusion`, `Groot`, `PI0`, `PI05`, `PI0Fast`,
`SmolVLA`, `XVLA`. Wrapper-vs-native equivalence is verified end-to-end for
the validated subset (see _Supported Policies_ below).

### Universal Wrapper (Escape Hatch)

`LeRobotPolicy(policy_name=...)` accepts any LeRobot-registered policy name
(e.g. `"vqbet"`, `"tdmpc"`, `"sac"`). Behavior is best-effort and not covered
by the equivalence test suite; a one-time `UserWarning` is emitted for any
name outside the named set.

### Supported Policies

| Policy        | Equivalence             | Description                             |
| ------------- | ----------------------- | --------------------------------------- |
| `act`         | Validated               | Action Chunking Transformer             |
| `diffusion`   | Validated               | Diffusion Policy                        |
| `smolvla`     | Validated               | Small Vision-Language-Action            |
| `pi0`         | Validated               | Vision-Language Policy                  |
| `pi05`        | Validated               | PI0.5 (Improved PI0)                    |
| `pi0_fast`    | Validated               | Fast Inference PI0                      |
| `groot`       | Named, xfail (upstream) | NVIDIA GR00T-N1 VLA Foundation Model    |
| `xvla`        | Named, xfail (upstream) | XVLA Vision-Language-Action             |
| anything else | Best-effort (warns)     | Routed via `LeRobotPolicy` escape hatch |

## Features

- ✅ Verified output equivalence with native LeRobot for the validated subset
- ✅ Full PyTorch Lightning integration
- ✅ Thin wrapper pattern (no reimplementation)
- ✅ All LeRobot features accessible via `policy.lerobot_policy`
- ✅ Support for VLA (Vision-Language-Action) models
- ✅ Universal escape hatch for non-named LeRobot policies (best-effort)

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

# Use any LeRobot policy by name. The equivalence guarantee applies only to
# the validated subset (see Supported Policies above); all other names —
# including named-but-unvalidated wrappers and arbitrary policies like
# "vqbet" — are best-effort and emit a one-time UserWarning.
policy = LeRobotPolicy(
    policy_name="vqbet",
    optimizer_lr=1e-4,
)
```

## Documentation

- **Design & Architecture**: [docs/design/policy/lerobot.md](../../../docs/design/policy/lerobot.md)
- **Usage Guide**: [docs/guides/lerobot.md](../../../docs/guides/lerobot.md)
- **LeRobot**: <https://github.com/huggingface/lerobot>
