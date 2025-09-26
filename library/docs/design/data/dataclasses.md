# Observation

The `Observation` dataclass is an internal representation.

In imitation learning and reinforcement learning an observation,
represents the input enviroment a policy must map to perform an action.
It also for imitation learning holds the corresponding ground truth action,
the policy must learn to predict.

```mermaid
classDiagram
    class Observation {
        + dict[str, torch.Tensor|np.ndarray] | torch.Tensor | np.ndarray | None action
        + dict[str, torch.Tensor|np.ndarray] | torch.Tensor | np.ndarray | None task
        + dict[str, torch.Tensor|np.ndarray] | torch.Tensor | np.ndarray | None state
        + dict[str, torch.Tensor|np.ndarray] | torch.Tensor | np.ndarray | None images
        + torch.Tensor | np.ndarray | None next_reward
        + bool | None next_success
        + torch.Tensor | np.ndarray | None episode_index
        + torch.Tensor | np.ndarray | None frame_index
        + torch.Tensor | np.ndarray | None index
        + torch.Tensor | np.ndarray | None task_index
        + torch.Tensor | np.ndarray | None timestamp
        + dict[str, Any] | None info
        + dict[str, Any] | None extra
    }
```

Example:

```python
from action_trainer.data import Observation
import torch
obs_1 = Observation(
    action=torch.randn((10,)),
    images={"top": torch.randn((3, 256, 256))}
)
```
