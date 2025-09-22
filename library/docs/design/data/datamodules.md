# DataModule

The `DataModue` is a child class of the `LightningDataModule`.

We in the future will support the use of `gymnasium` environments.
For now we allow them as optional args,
with the added feature of wrapping the `gym` with a `TimeLimit`.

```mermaid
classDiagram
    class LightningDataModule {
    }

    class DataModule {
        - Dataset train_dataset
        - int train_batch_size
        - BaseGym|list~BaseGym~|None eval_gyms
        - Dataset eval_dataset
        - int num_rollouts_eval
        - BaseGym|list~BaseGym~|None test_gyms
        - Dataset test_dataset
        - int num_rollouts_test
        - int|None max_episode_steps
        + setup(stage: str) void
        + train_dataloader() DataLoader
        + val_dataloader() DataLoader
        + test_dataloader() DataLoader
        + predict_dataloader() DataLoader
    }

    class Dataset
    class DataLoader
    class BaseGym
    class GymDataset
    class ConcatDataset
    class TimeLimit

    LightningDataModule <|-- DataModule
    DataModule --> Dataset
    DataModule --> DataLoader
    DataModule --> BaseGym
    DataModule --> GymDataset
    DataModule --> ConcatDataset
    DataModule --> TimeLimit
```
