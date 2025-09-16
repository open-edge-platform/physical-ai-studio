<!-- markdownlint-disable MD013 -->

# Dummy

## Dummy policy

A dummy policy here just takes in what shape the action it should output.

The idea is to use in integration with our `Trainer`.

```mermaid
classDiagram
    class ActionTrainerModule {
    }

    class DummyModel {
    }

    class DummyPolicy {
        - action_shape: torch.Size
        - model: DummyModel
        + __init__(action_shape: torch.Size | Iterable)
        - _validate_action_shape(shape: torch.Size | Iterable) torch.Size
    }

    DummyPolicy --|> ActionTrainerModule : inherits
    DummyPolicy *-- DummyModel : contains
```

## Dummy Model

Similarly a dummy model is to ensure we can expose the correct params,
for dataset interaction and also predict fake actions for use in a `Trainer`.

```mermaid
classDiagram
    class nn.Module {
    }

    class DummyModel {
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
from action_trainer.data import LeRobotActionDataModule
from action_trainer.policies.dummy import DummyPolicy
action_shape = (2,)
policy = DummyPolicy(action_shape=action_shape)
```

or from a `ActionDataModule`:

```python
from action_trainer.data import LeRobotActionDataModule
from action_trainer.policies.dummy import DummyPolicy
lerobot_action_datamodule = LeRobotActionDataModule(repo_id="lerobot/pusht", train_batch_size=16)
action_shape = lerobot_action_datamodule.train_dataset.action_features["action"]["shape"]
policy = DummyPolicy(action_shape=action_shape)
```
