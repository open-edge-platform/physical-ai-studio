# GymDataset

The Gym dataset is needed for environments from `gymnasium` to work in training.

We calculate the length of an evaluation dataset based on the number of rollouts.

```mermaid
classDiagram
    class Dataset
    class BaseGym

    class GymDataset{
        +BaseGym env
        +int num_rollouts
        +__init__(env: BaseGym, num_rollouts: int)
        +__len__() int
        +__getitem__(index: int) BaseGym
    }
    GymDataset --|> Dataset
    GymDataset --> BaseGym : uses
```
