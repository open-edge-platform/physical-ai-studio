# Trainer

Lightning Trainer wrapper with getiaction-specific defaults and callbacks.

## Overview

The `Trainer` class extends Lightning's Trainer with:

- **Automatic callback injection** - `PolicyDatasetInteraction` for policy-data coordination
- **Better defaults** - Experiments saved to `experiments/` directory
- **Experiment naming** - Optional `experiment_name` parameter for organization

## Usage

```python
from getiaction.train import Trainer
from getiaction.policies import ACT
from getiaction.data import LeRobotDataModule

# Basic training
policy = ACT()
datamodule = LeRobotDataModule(repo_id="lerobot/pusht")
trainer = Trainer(max_epochs=100)
trainer.fit(policy, datamodule)

# With experiment name
trainer = Trainer(max_epochs=100, experiment_name="pusht_act_v1")
# Saves to: experiments/pusht_act_v1/version_0/

# Custom logger
from lightning.pytorch.loggers import WandbLogger
trainer = Trainer(max_epochs=100, logger=WandbLogger(project="robot"))
```

## Interface

```python
class Trainer(lightning.Trainer):
    def __init__(
        self,
        *,
        # getiaction-specific
        experiment_name: str | None = None,

        # Hardware
        accelerator: str = "auto",
        devices: int | list[int] = "auto",
        precision: str | None = None,

        # Training control
        max_epochs: int | None = None,
        gradient_clip_val: float | None = None,

        # Logging
        logger: Logger | bool | None = None,
        default_root_dir: str = "experiments",  # Changed from Lightning default

        # Validation
        num_sanity_val_steps: int = 0,  # Changed from Lightning default (2)
        check_val_every_n_epoch: int = 1,

        # All other Lightning Trainer parameters...
    ): ...
```

## Key Differences from Lightning

| Parameter | Lightning Default | getiaction Default | Reason |
|-----------|------------------|-------------------|--------|
| `default_root_dir` | Current directory | `"experiments"` | Cleaner project structure |
| `num_sanity_val_steps` | 2 | 0 | Embodied AI typically doesn't need sanity checks |

## Callbacks

### PolicyDatasetInteraction

Automatically injected callback that coordinates policy initialization with datamodule:

```python
class PolicyDatasetInteraction(Callback):
    def on_fit_start(self, trainer, pl_module):
        # Ensures policy can access dataset features for lazy initialization
```

This enables policies to be created without specifying input/output shapes, deferring initialization until the dataset is available.

## Output Structure

```
experiments/
├── lightning_logs/           # Default (no experiment_name)
│   └── version_0/
│       ├── checkpoints/
│       ├── hparams.yaml
│       └── events.out.tfevents.*
└── pusht_act_v1/            # With experiment_name="pusht_act_v1"
    └── version_0/
        └── ...
```

## Examples

### GPU Training with Mixed Precision

```python
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    max_epochs=200,
)
```

### Multi-GPU Training

```python
trainer = Trainer(
    accelerator="gpu",
    devices=4,
    strategy="ddp",
    max_epochs=100,
)
```

### Development/Debug Mode

```python
# Quick run with 1 batch
trainer = Trainer(fast_dev_run=True)

# Overfit on small subset
trainer = Trainer(overfit_batches=10)
```

## See Also

- [CLI Guide](../../guides/cli.md) - Training via command line
- [Policy Design](../policy/overview.md) - Policy implementation
- [Data Module](../data/overview.md) - Dataset management
