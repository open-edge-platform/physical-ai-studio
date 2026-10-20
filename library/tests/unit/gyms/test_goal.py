# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for GoalConditionedGym."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from physicalai.data import GoalConditionedDataset, Observation
from physicalai.gyms import GoalConditionedGym, Gym


class FakeGym(Gym):
    """Gym returning batched observations with a task description and configurable cameras."""

    def __init__(
        self,
        images: dict[str, torch.Tensor] | torch.Tensor,
        batch_size: int = 1,
        task: str | list[str] = "task",
    ) -> None:
        self.images = images
        self.batch_size = batch_size
        self.task_description = task
        self.extra_attribute = "forwarded"

    def _observation(self) -> Observation:
        task = self.task_description if isinstance(self.task_description, list) else [self.task_description]
        return Observation(images=self.images, state=torch.zeros(self.batch_size, 2), task=task)

    def reset(self, *, seed=None, episode_index=0, **reset_kwargs):
        return self._observation(), {"seed": seed}

    def step(self, action):
        return self._observation(), 1.0, False, False, {}

    def close(self) -> None:
        pass

    def sample_action(self) -> torch.Tensor:
        return torch.zeros(2)

    def to_observation(self, raw_obs) -> Observation:
        return self._observation()


class TaskGoals(GoalConditionedDataset):
    """Goal dataset returning fixed goal images per task, counting rollout lookups."""

    def __init__(self, goal_images, tasks=("task",)) -> None:  # noqa: D107
        self.goal_images = goal_images
        self.tasks = tasks
        self.lookups: list[str] = []

    def get_goal(
        self, observation: Observation,
    ) -> dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None:
        raise NotImplementedError

    def get_task_goal(self, task: str) -> dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None:
        if task not in self.tasks:
            raise KeyError(task)
        self.lookups.append(task)
        return self.goal_images


class PerTaskGoals:
    """Canonical image value derived from a task name's final character."""

    def get_task_goal(self, task: str) -> dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None:
        return {"top": _image(float(task[-1]))}


def _image(value: float = 1.0) -> torch.Tensor:
    return torch.full((3, 2, 2), value)


def _gym(images, goal_images, **kwargs) -> GoalConditionedGym:
    return GoalConditionedGym(FakeGym(images), TaskGoals(goal_images), **kwargs)


class TestGoalConditionedGym:
    def test_goal_attached_on_reset_and_step(self):
        gym = _gym({"top": torch.zeros(1, 3, 2, 2)}, {"top": _image()})

        obs, info = gym.reset(seed=3)
        step_obs, reward, *_ = gym.step(torch.zeros(2))

        assert info == {"seed": 3}
        assert reward == 1.0
        for o in (obs, step_obs):
            assert torch.equal(o.goal_images["top"], _image().unsqueeze(0))

    def test_goal_looked_up_once_per_episode_by_task(self):
        gym = _gym({"top": torch.zeros(1, 3, 2, 2)}, {"top": _image()})
        gym.dataset.lookups.clear()  # construction checks the task once

        gym.reset()
        gym.step(torch.zeros(2))
        gym.step(torch.zeros(2))

        assert gym.dataset.lookups == ["task"]

    def test_unknown_task_fails_at_construction(self):
        with pytest.raises(KeyError, match="other"):
            GoalConditionedGym(FakeGym(torch.zeros(1, 3, 2, 2), task="other"), TaskGoals(_image()))

    def test_goal_expanded_to_batch(self):
        gym = GoalConditionedGym(FakeGym({"top": torch.zeros(4, 3, 2, 2)}, batch_size=4), TaskGoals({"top": _image()}))

        obs, _ = gym.reset()

        assert obs.goal_images["top"].shape == (4, 3, 2, 2)

    def test_no_goal_before_reset(self):
        gym = _gym({"top": torch.zeros(1, 3, 2, 2)}, {"top": _image()})

        assert gym.to_observation(None).goal_images is None

    def test_forwards_to_wrapped_gym(self):
        gym = _gym({"top": torch.zeros(1, 3, 2, 2)}, {"top": _image()})

        assert gym.extra_attribute == "forwarded"
        assert gym.sample_action().shape == (2,)

    def test_rejects_heterogeneous_batched_tasks(self):
        gym = GoalConditionedGym(
            FakeGym(
                {"top": torch.zeros(3, 3, 2, 2)},
                batch_size=3,
                task=["task1", "task2", "task3"],
            ),
            PerTaskGoals(),
        )

        with pytest.raises(ValueError, match="same task"):
            gym.reset()

    def test_rejects_wrong_task_batch_length(self):
        gym = GoalConditionedGym(
            FakeGym({"top": torch.zeros(3, 3, 2, 2)}, batch_size=3, task=["task1", "task2"]),
            PerTaskGoals(),
        )

        with pytest.raises(ValueError, match="Expected one task description or 3"):
            gym.reset()

    def test_attached_goal_does_not_mutate_store_or_next_reset(self):
        gym = _gym({"top": torch.zeros(1, 3, 2, 2)}, {"top": _image()})
        first, _ = gym.reset()
        first.goal_images["top"].zero_()

        second, _ = gym.reset()

        assert torch.equal(second.goal_images["top"], _image().unsqueeze(0))


class TestCameraMatching:
    def test_single_goal_image_takes_gym_camera_name(self):
        # PushT: the dataset stores one bare image, the gym calls its camera "top".
        obs, _ = _gym({"top": torch.zeros(1, 3, 2, 2)}, _image()).reset()

        assert set(obs.goal_images) == {"top"}

    def test_single_camera_is_renamed(self):
        obs, _ = _gym({"top": torch.zeros(1, 3, 2, 2)}, {"image": _image()}).reset()

        assert set(obs.goal_images) == {"top"}

    def test_matching_cameras_are_kept(self):
        # LIBERO: dataset and gym both use "image" / "image2".
        images = {"image": torch.zeros(1, 3, 2, 2), "image2": torch.zeros(1, 3, 2, 2)}

        obs, _ = _gym(images, {"image": _image(1.0), "image2": _image(2.0)}).reset()

        assert obs.goal_images["image2"][0, 0, 0, 0] == 2.0

    def test_camera_map_renames_goal_cameras(self):
        images = {"front": torch.zeros(1, 3, 2, 2), "wrist": torch.zeros(1, 3, 2, 2)}
        gym = _gym(
            images, {"image": _image(1.0), "image2": _image(2.0)}, camera_map={"image": "front", "image2": "wrist"}
        )

        obs, _ = gym.reset()

        assert obs.goal_images["wrist"][0, 0, 0, 0] == 2.0

    def test_unmatched_cameras_raise(self):
        images = {"front": torch.zeros(1, 3, 2, 2), "wrist": torch.zeros(1, 3, 2, 2)}

        with pytest.raises(KeyError, match="camera_map"):
            _gym(images, {"image": _image(), "image2": _image()}).reset()

    def test_single_tensor_gym_images(self):
        obs, _ = _gym(torch.zeros(1, 3, 2, 2), {"image": _image()}).reset()

        assert obs.goal_images.shape == (1, 3, 2, 2)

    @pytest.mark.parametrize("goal_images", [{}, {"image": _image(), "image2": _image()}])
    def test_single_tensor_gym_rejects_ambiguous_goal_cameras(self, goal_images):
        with pytest.raises(KeyError, match="exactly one camera"):
            _gym(torch.zeros(1, 3, 2, 2), goal_images).reset()
