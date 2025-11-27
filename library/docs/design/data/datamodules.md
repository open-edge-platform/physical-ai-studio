# DataModule

Lightning data module with support for gym environments.

## Interface

```python
class DataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        train_batch_size: int,
        val_gyms: Gym | None = None,
        val_dataset: Dataset | None = None,
        test_gyms: Gym | None = None,
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
