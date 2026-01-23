# Policy Design

Policies are Lightning modules that wrap PyTorch models for training and inference.

## Structure

Each policy follows a consistent structure:

```
policy_name/
├── config.py   # Dataclass configuration
├── model.py    # PyTorch nn.Module
└── policy.py   # Lightning module wrapper
```

## Architecture

```mermaid
graph TD
    A[Policy] --> B[Model]
    A --> C[Export Mixin]
    A --> D[FromCheckpoint Mixin]

    B --> E[forward - training]
    B --> F[predict_action_chunk - inference]
    B --> G[select_action - single step]

    A --> H[training_step]
    A --> I[validation_step]
    A --> J[configure_optimizers]
```

## Base Class

All policies inherit from `Policy`, which provides:

- **Action queue management** - For action chunking (predicting multiple future actions)
- **Gym evaluation** - Built-in rollout evaluation with torchmetrics
- **Device transfer** - Automatic batch-to-device handling for `Observation` objects

```python
class Policy(L.LightningModule, ABC):
    def __init__(self, n_action_steps: int = 1) -> None:
        self._action_queue: deque[Tensor] = deque(maxlen=n_action_steps)
        self.val_rollout = Rollout()
        self.test_rollout = Rollout()

    @abstractmethod
    def forward(self, batch: Observation) -> Any: ...

    @abstractmethod
    def predict_action_chunk(self, batch: Observation) -> Tensor: ...

    def select_action(self, batch: Observation) -> Tensor:
        """Returns single action, using queue for chunked predictions."""
```

## PolicyLike Protocol

For inference flexibility, a `PolicyLike` protocol defines the minimal interface:

```python
@runtime_checkable
class PolicyLike(Protocol):
    def select_action(self, observation: Observation) -> Tensor: ...
    def reset(self) -> None: ...
```

Both `Policy` and `InferenceModel` satisfy this protocol, enabling:

- Training with full Lightning infrastructure
- Production inference with exported models
- Unified benchmarking interface

## Implemented Policies

| Policy | Description | Source |
|--------|-------------|--------|
| **ACT** | Action Chunking Transformer | Native implementation |
| **Diffusion** | Diffusion Policy | LeRobot wrapper |
| **GROOT** | Vision-language policy | Native implementation |
| **Pi0** | Physical Intelligence model | Native implementation |
| **SmolVLA** | Small Vision-Language-Action | Native + LeRobot |

## Creating a New Policy

1. **Define config** (`config.py`):

```python
@dataclass
class MyConfig:
    hidden_dim: int = 256
    chunk_size: int = 100
```

2. **Implement model** (`model.py`):

```python
class MyModel(nn.Module):
    def __init__(self, config: MyConfig, input_features, output_features):
        self.config = config
        # ... build network

    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        """Training: returns (loss, loss_dict)"""

    def predict_action_chunk(self, batch: dict) -> Tensor:
        """Inference: returns action chunk [B, T, D]"""
```

3. **Create policy wrapper** (`policy.py`):

```python
class MyPolicy(FromCheckpoint, Export, Policy):
    model_type = MyModel
    model_config_type = MyConfig

    def __init__(self, model: MyModel | None = None, optimizer_fn=None):
        super().__init__(n_action_steps=model.config.chunk_size if model else 1)
        self.model = model

    def forward(self, batch: Observation):
        if self.training:
            return self.model(batch.to_dict())
        return self.predict_action_chunk(batch)

    def predict_action_chunk(self, batch: Observation) -> Tensor:
        return self.model.predict_action_chunk(batch.to(self.device).to_dict())

    def training_step(self, batch, batch_idx):
        loss, _ = self.model(batch.to_dict())
        self.log("train/loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)
```

## Mixins

Policies compose functionality through mixins:

| Mixin | Purpose |
|-------|---------|
| `Export` | Adds `export()`, `to_onnx()`, `to_openvino()` methods |
| `FromCheckpoint` | Adds `load_from_checkpoint()` with config reconstruction |

## See Also

- [Base Policy](base.md) - Detailed base class documentation
- [Export Design](../export/overview.md) - Model export system
- [Data Module](../data/overview.md) - Dataset integration
