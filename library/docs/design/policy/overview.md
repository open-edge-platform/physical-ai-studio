# Policy Design

Policies are action models consisting of a Lightning module wrapper and a
PyTorch model.

## Structure

Each policy follows this structure:

```bash
policy_name/
├── config.py   # Configuration
├── model.py    # PyTorch model
└── policy.py   # Lightning module
```

## Interface

```python
class Model(nn.Module):
    def forward(self, obs: dict[str, Tensor]) -> Tensor:
        """Training forward pass."""

    def select_action(self, obs: dict[str, Tensor]) -> Tensor:
        """Single-step action selection."""

    def predict_action_chunk(self, obs: dict[str, Tensor]) -> Tensor:
        """Multi-step action prediction."""
```

The model should depend only on PyTorch for easy extraction from the framework.
