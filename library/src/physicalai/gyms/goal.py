# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Goal conditioning for gyms.

`GoalConditionedGym` attaches the goal images for the gym's current task to every
`Observation` returned by ``reset`` / ``step``, so a goal-conditioned policy sees the same
``goal_images`` field during rollouts as it does during training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import Gym

if TYPE_CHECKING:
    import numpy as np

    from physicalai.data.goal import TaskGoalStore
    from physicalai.data.observation import Observation

    from .types import SingleOrBatch


def _device(obs: Observation) -> torch.device:
    for value in (obs.images, obs.state):
        if isinstance(value, dict):
            value = next(iter(value.values()), None)  # noqa: PLW2901
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _task_description(task: Any, batch_size: int) -> str:  # noqa: ANN401
    """Return the single task represented by a rollout batch.

    Raises:
        ValueError: If the task is missing, malformed, or differs within the batch.
    """
    if isinstance(task, str):
        return task
    if isinstance(task, (list, tuple)) and task and all(isinstance(item, str) for item in task):
        if len(task) not in {1, batch_size}:
            msg = f"Expected one task description or {batch_size} repeated descriptions, got {len(task)}."
            raise ValueError(msg)
        if len(set(task)) != 1:
            msg = "GoalConditionedGym requires every environment in a batch to use the same task."
            raise ValueError(msg)
        return task[0]
    msg = "GoalConditionedGym needs Observation.task to contain one task description."
    raise ValueError(msg)


def _batch_goal(
    value: dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None:
    """Broadcast an unbatched goal into independent contiguous batch tensors.

    Returns:
        Goal images with a leading batch dimension.
    """

    def batch(image: Any) -> torch.Tensor:  # noqa: ANN401
        tensor = torch.as_tensor(image, device=device)
        return tensor.unsqueeze(0).expand(batch_size, *tensor.shape).clone().contiguous()

    if isinstance(value, dict):
        return {key: batch(image) for key, image in value.items()}
    return None if value is None else batch(value)


def _clone_goal(
    value: dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None,
    device: torch.device,
) -> dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None:
    if isinstance(value, dict):
        return {key: torch.as_tensor(item, device=device).clone().contiguous() for key, item in value.items()}
    return None if value is None else torch.as_tensor(value, device=device).clone().contiguous()


def _match_cameras(
    goal: dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None,
    images: dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None,
    camera_map: dict[str, str],
) -> dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None:
    """Put goal images in the same layout as the gym's images (dict of cameras or single tensor).

    Returns:
        The goal images, keyed like ``images``.

    Raises:
        KeyError: If the goal cameras cannot be matched to the gym cameras.
    """
    if goal is None:
        return None
    if isinstance(goal, dict):
        goal = {camera_map.get(k, k): v for k, v in goal.items()}
    if not isinstance(images, dict):
        # The gym gives a single image, so the goal must have exactly one camera too.
        if isinstance(goal, dict):
            if len(goal) == 1:
                return next(iter(goal.values()))
            msg = f"Cannot match goal cameras {sorted(goal)} to a single-tensor gym image; expected exactly one camera."
            raise KeyError(msg)
        return goal
    if not isinstance(goal, dict):
        goal = {"": goal}
    if goal.keys() == images.keys():
        return goal
    if len(goal) == 1 and len(images) == 1:
        # One camera on each side: use the gym's name.
        return {next(iter(images)): next(iter(goal.values()))}
    msg = f"Cannot match goal cameras {sorted(goal)} to gym cameras {sorted(images)}; pass camera_map."
    raise KeyError(msg)


class GoalConditionedGym(Gym):
    """Gym whose observations carry the goal for its current task.

    On every ``reset`` the gym's task description (``Observation.task``) is looked up in the
    goal dataset, and that task's canonical goal is attached to every observation of the episode,
    batched and laid out like the gym's own images. Works for single-task gyms (PushT) and
    multi-task suites (such as LIBERO) alike. A vectorized batch must contain one task.
    Everything else is forwarded to the wrapped gym.

    Examples:
        >>> dataset = LeRobotGoalConditionedDataset(LeRobotDataset("lerobot/libero"))
        >>> gym = GoalConditionedGym(LiberoGym(task_suite="libero_goal", task_id=3), dataset)
        >>> obs, _ = gym.reset(seed=0)
        >>> obs.goal_images["image"].shape
        torch.Size([1, 3, 256, 256])
    """

    def __init__(
        self,
        gym: Gym,
        dataset: TaskGoalStore,
        camera_map: dict[str, str] | None = None,
    ) -> None:
        """Initialize the wrapper.

        Args:
            gym: Gym to wrap. Its observations must carry the task description in ``task``.
            dataset: Goal-conditioned dataset used to resolve task goals on reset.
            camera_map: Renames goal cameras to gym cameras, e.g. ``{"image": "top"}``. Only
                needed when names differ and there is more than one camera.
        """
        self.gym = gym
        self.dataset = dataset
        self.camera_map = camera_map or {}
        self._goal: dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None = None

        # Fail early if the gym already knows a task the dataset does not have.
        description = getattr(gym, "task_description", None)
        if isinstance(description, str) and description:
            dataset.get_task_goal(description)

    def get_goal(
        self,
        observation: Observation,
    ) -> dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None:
        """Return independent batched goal images for ``observation``'s task."""
        task = _task_description(observation.task, observation.batch_size)
        goal = _match_cameras(self.dataset.get_task_goal(task), observation.images, self.camera_map)
        return _batch_goal(goal, observation.batch_size, _device(observation))

    def reset(
        self,
        *,
        seed: int | None = None,
        episode_index: int = 0,
        **reset_kwargs: Any,  # noqa: ANN401
    ) -> tuple[Observation, dict[str, Any] | list[dict[str, Any]]]:
        """Reset the gym and pick the goal for the new episode.

        Returns:
            A tuple ``(observation, info)`` with the goal attached to the observation.
        """
        obs, info = self.gym.reset(seed=seed, episode_index=episode_index, **reset_kwargs)
        self._goal = self.get_goal(obs)
        return self._attach(obs), info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        Observation,
        SingleOrBatch[float],
        SingleOrBatch[bool],
        SingleOrBatch[bool],
        SingleOrBatch[dict[str, Any]],
    ]:
        """Step the gym.

        Returns:
            A tuple ``(observation, reward, terminated, truncated, info)`` with the goal attached.
        """
        obs, reward, terminated, truncated, info = self.gym.step(action)
        return self._attach(obs), reward, terminated, truncated, info

    def _attach(self, obs: Observation) -> Observation:
        if self._goal is None:
            return obs
        obs.goal_images = _clone_goal(self._goal, _device(obs))
        return obs

    def render(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Render the wrapped gym.

        Returns:
            The render output of the wrapped gym.
        """
        return self.gym.render(*args, **kwargs)

    def get_max_episode_steps(self) -> int | None:
        """Return the wrapped gym's step limit."""
        return self.gym.get_max_episode_steps()

    def close(self) -> None:
        """Close the wrapped gym."""
        self.gym.close()

    def sample_action(self) -> torch.Tensor:
        """Sample an action from the wrapped gym.

        Returns:
            A valid action.
        """
        return self.gym.sample_action()

    def to_observation(self, raw_obs: Any) -> Observation:  # noqa: ANN401
        """Convert a raw observation with the wrapped gym and attach the current goal.

        Returns:
            The observation with the goal attached.
        """
        return self._attach(self.gym.to_observation(raw_obs))

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Forward attribute access to the wrapped gym.

        Returns:
            The attribute of the wrapped gym.

        Raises:
            AttributeError: If the wrapped gym is not set yet.
        """
        # Avoid infinite recursion before __init__ has set `gym` (e.g. while unpickling).
        if name == "gym":
            raise AttributeError(name)
        return getattr(self.gym, name)


__all__ = ["GoalConditionedGym"]
