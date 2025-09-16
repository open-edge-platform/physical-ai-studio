# ActionDataModule

The `ActionDataModue` is a child class of the `LightningDataModule`.

We in the future will support the use of `gymnasium` environments.
For now we allow them as optional args,
with the added feature of wrapping the `gym` with a `TimeLimit`.

```mermaid
classDiagram
    class LightningDataModule
    class ActionDataset
    class BaseGym
    class TimeLimit
    class GymDataset
    class ConcatDataset

    class ActionDataModule {
        + ActionDataset train_dataset
        + int train_batch_size
        + BaseGym | list<BaseGym> | None eval_gyms
        + Optional<Dataset> eval_dataset
        + int num_rollouts_eval
        + BaseGym | list<BaseGym> | None test_gyms
        + Optional<Dataset> test_dataset
        + int num_rollouts_test
        + int | None max_episode_steps
        --
        + __init__(train_dataset: ActionDataset, train_batch_size: int, ...)
        + setup(stage: str) None
        + train_dataloader() DataLoader
        + val_dataloader() DataLoader
        + test_dataloader() DataLoader
        + predict_dataloader() NotImplementedError
    }

    ActionDataModule --|> LightningDataModule : inherits
    ActionDataModule ..> ActionDataset : uses
    ActionDataModule ..> BaseGym : uses
    ActionDataModule ..> TimeLimit : uses
    ActionDataModule ..> GymDataset : uses
    ActionDataModule ..> ConcatDataset : uses
```
