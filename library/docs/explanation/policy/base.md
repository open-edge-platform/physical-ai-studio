# TrainerModule

Base class for Lightning training modules.

## Interface

```python test="skip" reason="interface definition, not executable"
class PolicyModule(LightningModule):
    def __init__(self):
        """Initialize module."""

    def forward(self, batch: dict[str, Tensor], *args, **kwargs) -> Tensor:
        """Training forward pass."""

    @abstractmethod
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Action selection (inference)."""

    model: nn.Module  # PyTorch model
```
