<!-- markdownlint-disable MD013 -->

# Dummy

## Dummy policy

A dummy policy here just takes in what shape the action it should output.

The idea is to use in integration with our `Trainer`.

```mermaid
classDiagram
    class TrainerModule
    class DummyModel
    class DummyConfig

    class Dummy {
        - DummyConfig config
        - torch.Size action_shape
        - DummyModel model
        + __init__(config: DummyConfig)
        + _validate_action_shape(shape: torch.Size|Iterable) torch.Size
        + select_action(batch: dict[str, torch.Tensor]) torch.Tensor
        + training_step(batch: dict[str, torch.Tensor], batch_idx: int) dict
        + configure_optimizers() torch.optim.Optimizer
        + evaluation_step(batch: dict[str, torch.Tensor], stage: str) None
        + validation_step(batch: dict[str, torch.Tensor], batch_idx: int) None
        + test_step(batch: dict[str, torch.Tensor], batch_idx: int) None
    }

    TrainerModule <|-- Dummy
    Dummy --> DummyModel
    Dummy --> DummyConfig
```

## Dummy Model

Similarly a dummy model is to ensure we can expose the correct params,
for dataset interaction and also predict fake actions for use in a `Trainer`.

```mermaid
classDiagram
    class nn.Module {
    }

    class Dummy {
        - action_shape: torch.Size
        - n_action_steps: int
        - temporal_ensemble_coeff: float | None
        - n_obs_steps: int
        - horizon: int
        - temporal_buffer: None
        - _action_queue: deque
        - dummy_param: nn.Parameter
        + __init__(action_shape: torch.Size, n_action_steps: int=1, temporal_ensemble_coeff: float|None=None, n_obs_steps: int=1, horizon: int|None=None)
        + observation_delta_indices: list[int]
        + action_delta_indices: list[int]
        + reward_delta_indices: None
        + reset() void
        + select_action(batch: dict[str, torch.Tensor]) torch.Tensor
        + predict_action_chunk(batch: dict[str, torch.Tensor]) torch.Tensor
        + forward(batch: dict[str, torch.Tensor]) torch.Tensor | tuple[torch.Tensor, dict]
    }

    DummyModel --|> nn.Module : inherits
```

Example:

```python
from physicalai.data import LeRobotDataModule
from physicalai.policies import Dummy, DummyConfig

if __name__ == "__main__":
    l_dm = LeRobotDataModule(repo_id="lerobot/pusht")
    policy = Dummy(DummyConfig(action_shape=l_dm.train_dataset.action_features["action"]["shape"]))
```
