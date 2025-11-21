# Trainer

Lightning trainer wrapper with policy-datamodule interaction callback.

## Interface

```python
class Trainer:
    def __init__(
        self,
        num_sanity_val_steps: int = 0,
        callbacks: list | None = None,
        **kwargs
    ):
        """Wrap Lightning trainer."""

    def fit(self, model: TrainerModule, datamodule: DataModule, **kwargs):
        """Train the model."""
```

## Components

- `L.Trainer` - PyTorch Lightning trainer backend
- `PolicyDatasetInteraction` - Callback for policy-data coordination
- `TrainerModule` - Lightning module interface
- `DataModule` - Lightning data module interface
