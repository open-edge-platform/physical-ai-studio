# Goal Conditioning

A goal-conditioned policy receives an image of the state it should reach with every
observation.

Goal images are attached as `Observation.goal_images` in the same layout as
`Observation.images`. The field is `None` when goal conditioning is disabled.

## Goal lookup

`LeRobotGoalConditionedDataset` wraps a normal LeRobot dataset and selects one integer
goal row per episode and task. For every sample read, it uses the observation's task or episode index
to select the goal row and reads the wrapped dataset a second time:

```python
observation = wrapped_dataset[index]
goal_row = task_goal_rows[observation.task_index]
goal = wrapped_dataset[goal_row]
observation.goal_images = goal.images
```

Observation and goal therefore follow the same normal dataset path. Training image
transforms are invoked independently for both reads; validation uses its ordinary
unaugmented dataset.

Two sources are available:

- `goal_source="task"`: every sample for a task uses the selected frame from that
  task's representative episode (the first selected episode by default).
- `goal_source="episode"`: every sample uses the selected frame from its own episode.

```python test="skip" reason="requires physicalai install and downloads data"
from physicalai.data.lerobot import LeRobotDataModule

# One split-local representative goal per task.
task_data = LeRobotDataModule(
    repo_id="lerobot/libero",
    goal_source="task",
)

# One goal per episode, with optional frame offsets.
episode_data = LeRobotDataModule(
    repo_id="lerobot/libero",
    goal_source="episode",
    episode_goal_frames={0: -1, 1: -3},
)

sample = episode_data.train_dataset[0]
sample.goal_images["image"].shape
```

`episode_goal_frames` accepts a Python mapping or a JSON object whose keys are
canonical integer strings. Offsets are relative to the episode; negative values
count from the end and missing entries default to `-1`.

For task goals, `task_goal_episodes` optionally selects the representative episode:

```python
LeRobotDataModule(
    repo_id="lerobot/libero",
    goal_source="task",
    task_goal_episodes={"put the bowl on the plate": 12},
    episode_goal_frames={12: -3},
)
```

## Split isolation

Training and eval-loss validation wrappers select their goal rows independently.
A training goal can only reference a training row, and a validation goal can only
reference a validation row. A configured task representative outside a split falls
back to that task's first local episode.

This keeps task and episode goals split-local and prevents held-out demonstrations
from supplying training goals.

## Rollouts

`GoalConditionedGym` takes a goal-conditioned dataset. On each reset it asks the
training dataset for the current task goal, batches it to match the gym observation,
and reuses it throughout the rollout episode.

```python test="skip" reason="requires LIBERO and downloads data"
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from physicalai.data.lerobot import LeRobotGoalConditionedDataset
from physicalai.gyms import GoalConditionedGym, LiberoGym

raw_dataset = LeRobotDataset("lerobot/libero")
dataset = LeRobotGoalConditionedDataset(raw_dataset, goal_source="task")
gym = GoalConditionedGym(
    LiberoGym(task_suite="libero_goal", task_id=3),
    dataset,
)

obs, _ = gym.reset(seed=0)
obs.goal_images["image"].shape  # torch.Size([1, 3, 256, 256])
```

`LeRobotDataModule` automatically wraps raw validation and test gyms with its
training goal-conditioned dataset.

Goal conditioning requires `data_format="physicalai"` and
`delta_timestamps=None`.

## Caveats

- Every training sample performs two dataset reads: observation and goal.
- Observation and goal augmentations use independent random draws.
- The selected frame must show the desired target state.
- Gym task text must match a dataset task (capitalization, whitespace, and a trailing
  period are ignored).
- Policies should normalize `goal_images` with the same statistics as `images`.

## See Also

- [LeRobot Data Integration](lerobot.md)
- [Observation](observation.md)
- [GoalConditionedGym](../gyms/goal.md)
