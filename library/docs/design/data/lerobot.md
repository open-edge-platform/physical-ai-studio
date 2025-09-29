# LeRobot

## LeRobotDatasetWrapper

To support but not re-implement LeRobot data standard have an interface.

The LeRobot dataset format is described in the [lerobot documentation](https://github.com/huggingface/lerobot?tab=readme-ov-file#the-lerobotdataset-format)

```mermaid
classDiagram
    class Dataset
    class LeRobotDataset
    class Observation

    class LeRobotDatasetWrapper {
        - LeRobotDataset _lerobot_dataset
        + __len__() int
        + __getitem__(idx) Observation
        + from_lerobot(LeRobotDataset) LeRobotDatasetWrapper
        + features
        + action_features
        + fps
        + tolerance_s
        + delta_indices
    }

    Dataset <|-- LeRobotDatasetWrapper
    LeRobotDatasetWrapper --> LeRobotDataset
    LeRobotDatasetWrapper --> Observation
```

Example (these examples will download data onto your disk):

```python
from getiaction.data import LeRobotDatasetWrapper

pusht_dataset = LeRobotDatasetWrapper("lerobot/pusht")

s0100_dataset = LeRobotDatasetWrapper("lerobot/svla_so100_pickplace")
```

or from `LeRobot` itself:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from getiaction.data import LeRobotDatasetWrapper
pusht_lerobot_dataset = LeRobotDataset("lerobot/pusht")
pusht_action_dataset = LeRobotDatasetWrapper.from_lerobot(pusht_lerobot_dataset)
```

## LeRobotDataModule

This serves as a wrapper of the `DataModule`,
specifically for ease of use with `LeRobot`.

```mermaid
classDiagram
    class DataModule
    class LeRobotDatasetWrapper
    class LeRobotDataset

    class LeRobotDataModule {
        + __init__(train_batch_size, repo_id, dataset, ...)
    }

    DataModule <|-- LeRobotDataModule
    LeRobotDataModule --> LeRobotDatasetWrapper
    LeRobotDataModule --> LeRobotDataset
```

Example (this will download data to disk if not cached already):

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from getiaction.data import LeRobotActionDataModule

repo_id = "lerobot/pusht"

# from repo id
datamodule = LeRobotActionDataModule(repo_id=repo_id, train_batch_size=16)

# from lerobot dataset
dataset = LeRobotDataset(repo_id=repo_id)
datamodule = LeRobotActionDataModule.from_lerobot(lerobot_dataset=dataset, train_batch_size=16)

```
