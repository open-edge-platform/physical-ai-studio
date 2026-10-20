# GoalConditionedGym

A wrapper that attaches the goal for the gym's current task to every observation.

On every `reset()` the gym's task description (`Observation.task`) is looked up in a
goal-conditioned dataset, and that task's canonical goal is attached to every observation
from `reset()` and `step()`, batched and laid out like the gym's own images. Everything
else is forwarded to the wrapped gym.

A single task description is broadcast across the observation batch. If a sequence is
provided, every item must describe the same task. Attached goal tensors are independent,
contiguous copies and cannot mutate the canonical goal.

```mermaid
classDiagram
    class GoalConditionedGym {
        + gym : Gym
        + dataset : GoalConditionedDataset
        + camera_map : dict[str, str]
        --
        + __init__(gym: Gym, dataset: GoalConditionedDataset, camera_map: dict|None)
        + get_goal(observation) dict|torch.Tensor|np.ndarray|None
        + reset(seed: int|None, **kwargs) Observation, dict|list[dict]
        + step(action) Observation, float|list[float], bool|list[bool], bool|list[bool], dict|list[dict]
        + render(...) Any
        + close() None
        + sample_action() torch.Tensor
        + to_observation(raw_obs) Observation
    }

    GoalConditionedGym --> Gym : wraps
    GoalConditionedGym --> GoalConditionedDataset : goal per task
```

Example:

```python test="skip" reason="requires LIBERO and downloads data"
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from physicalai.data.lerobot import LeRobotGoalConditionedDataset
from physicalai.gyms import GoalConditionedGym, LiberoGym

dataset = LeRobotGoalConditionedDataset(LeRobotDataset("lerobot/libero"))
gym = GoalConditionedGym(LiberoGym(task_suite="libero_goal", task_id=3), dataset)

obs, _ = gym.reset(seed=0)
obs.goal_images["image"].shape  # torch.Size([1, 3, 256, 256])
```

See [Goal Conditioning](../data/goal-conditioning.md) for how goals are chosen, task
matching and camera layout.
