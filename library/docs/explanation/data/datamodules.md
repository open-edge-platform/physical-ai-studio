# DataModule

Lightning data module with support for gym environments.

## Interface

```python
class DataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        train_batch_size: int,
        val_gym: Gym | None = None,
        val_dataset: Dataset | None = None,
        test_gym: Gym | None = None,
        test_dataset: Dataset | None = None,
        max_episode_steps: int | None = None,
    ):
        """Initialize data module."""

    def train_dataloader(self) -> DataLoader:
        """Training data loader."""

    def val_dataloader(self) -> DataLoader:
        """Validation data loader (dataset + gym rollouts)."""

    def test_dataloader(self) -> DataLoader:
        """Test data loader (dataset + gym rollouts)."""
```

## Features

- Combines datasets and gym environments
- Wraps gyms with `StepLimit`
- Configurable rollout counts for validation/test

## `pin_memory` and `persistent_workers`

Both default to speeding up training: `pin_memory="auto"` pins host memory for
faster host-to-GPU transfers whenever an accelerator is available, and
`persistent_workers=True` keeps train DataLoader workers alive between epochs
instead of re-spawning them.

These come at the cost of extra host RAM: pinned pages can't be swapped out,
and persistent workers hold their process (and any prefetched batches) open
for the whole run. If you're training a smaller policy (e.g. ACT, SmolVLA) on
a machine with limited RAM and see OOM kills or heavy swapping, set
`pin_memory=False` and/or `persistent_workers=False` in your datamodule
config.
