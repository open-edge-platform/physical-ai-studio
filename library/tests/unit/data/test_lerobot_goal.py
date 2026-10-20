# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for LeRobot task-goal conditioning, dataset wrapping and DataModule integration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from physicalai.data.lerobot import (
    LeRobotDataModule,
    LeRobotGoalConditionedDataset,
)
from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter
from physicalai.data import Observation
from physicalai.data.observation import GOAL_IMAGES
from physicalai.gyms import GoalConditionedGym, Gym

# 4 episodes of 3 frames. Episodes 0-1 are task "a" (0), episodes 2-3 are task "b" (1).
EPISODE_LENGTH = 3
EPISODE_TASKS = ["a", "a", "b", "b"]
# Task descriptions as stored in the dataset metadata (formatting is normalised on lookup).
TASK_DESCRIPTIONS = pd.DataFrame({"task_index": [0, 1]}, index=["Task A.", "task  b"])


def _stats(dim: int) -> dict:
    return {
        "mean": np.zeros(dim),
        "std": np.ones(dim),
        "min": np.zeros(dim),
        "max": np.ones(dim),
        "q01": np.zeros(dim),
        "q99": np.ones(dim),
    }


class FakeLeRobotDataset:
    """Mimics ``LeRobotDataset`` indexing: filtered datasets are indexed by relative row.

    Every frame's image and state are filled with its absolute index, so a goal image
    reveals which frame it was taken from.
    """

    def __init__(
        self,
        repo_id=None,
        root=None,
        episodes=None,
        image_dtype=torch.float32,
        image_value=None,
        **kwargs,
    ):
        self.repo_id = repo_id
        self.root = root
        self.episodes = episodes
        self.delta_indices = None
        self.delta_timestamps = kwargs.get("delta_timestamps")
        self.image_dtype = image_dtype
        self.image_value = image_value
        self.image_transforms = kwargs.get("image_transforms")
        self.image_reads = 0
        selected = sorted(episodes) if episodes is not None else range(len(EPISODE_TASKS))
        self._frames = [ep * EPISODE_LENGTH + i for ep in selected for i in range(EPISODE_LENGTH)]
        self.meta = SimpleNamespace(
            tasks=TASK_DESCRIPTIONS,
            episodes=[
                {
                    "episode_index": ep,
                    "dataset_from_index": ep * EPISODE_LENGTH,
                    "dataset_to_index": (ep + 1) * EPISODE_LENGTH,
                }
                for ep in range(len(EPISODE_TASKS))
            ],
            features={
                "observation.images.top": {"shape": (3, 2, 2), "dtype": "video", "names": ["c", "h", "w"]},
                "observation.state": {"shape": (2,), "dtype": "float32", "names": None},
                "action": {"shape": (2,), "dtype": "float32", "names": None},
            },
            stats={
                "observation.images.top": _stats(3),
                "observation.state": _stats(2),
                "action": _stats(2),
            },
        )
        self.features = self.meta.features
        self.fps = 10
        self.tolerance_s = 1e-4

    def __len__(self) -> int:
        return len(self._frames)

    def get_raw_item(self, idx: int) -> dict:
        frame = self._frames[idx]
        episode = frame // EPISODE_LENGTH
        return {
            "episode_index": torch.tensor(episode),
            "task_index": torch.tensor(0 if EPISODE_TASKS[episode] == "a" else 1),
        }

    def set_image_transforms(self, image_transforms):
        self.image_transforms = image_transforms

    def clear_image_transforms(self):
        self.image_transforms = None

    def __getitem__(self, idx: int) -> dict:
        self.image_reads += 1
        frame = self._frames[idx]
        episode = frame // EPISODE_LENGTH
        pixel = frame / 255 if self.image_value is None else self.image_value
        image = torch.full((3, 2, 2), pixel, dtype=self.image_dtype)
        if self.image_transforms is not None:
            image = self.image_transforms(image)
        return {
            "observation.images.top": image,
            "observation.state": torch.full((2,), float(frame)),
            "observation.state.joint_positions": torch.tensor([float(frame), float(frame + 1)]),
            "observation.state.gripper_position": torch.tensor([float(frame + 2)]),
            "action": torch.zeros(2),
            "action.joint_position": torch.tensor([float(frame), float(frame + 1)]),
            "action.gripper_position": torch.tensor([float(frame + 2)]),
            "episode_index": torch.tensor(episode),
            "frame_index": torch.tensor(frame % EPISODE_LENGTH),
            "index": torch.tensor(frame),
            "task_index": torch.tensor(0 if EPISODE_TASKS[episode] == "a" else 1),
            "timestamp": torch.tensor(0.0),
        }


def _frame(goal_images: dict[str, torch.Tensor]) -> torch.Tensor:
    """Absolute frame index a (possibly batched) goal image was taken from."""
    return (goal_images["top"][..., 0, 0, 0] * 255).round().long()


def _last_frame(episode: int) -> int:
    return (episode + 1) * EPISODE_LENGTH - 1


def _task_index(episode: int) -> int:
    return 0 if EPISODE_TASKS[episode] == "a" else 1


def _task_goal(dataset: LeRobotGoalConditionedDataset, episode: int):
    return dataset.get_goal(Observation(task_index=torch.tensor(_task_index(episode))))


class TestTaskGoals:
    def test_default_uses_first_episode_last_frame_per_task(self):
        goals = LeRobotGoalConditionedDataset(FakeLeRobotDataset())

        # Task "a" -> episode 0, task "b" -> episode 2 (first selected episode of each task).
        assert _frame(_task_goal(goals, 0)) == _last_frame(0)
        assert _frame(_task_goal(goals, 2)) == _last_frame(2)

    def test_same_goal_for_all_episodes_of_a_task(self):
        goals = LeRobotGoalConditionedDataset(FakeLeRobotDataset())

        assert _frame(_task_goal(goals, 0)) == _frame(_task_goal(goals, 1))
        assert _frame(_task_goal(goals, 2)) == _frame(_task_goal(goals, 3))

    def test_filtered_dataset_uses_its_own_first_episode(self):
        goals = LeRobotGoalConditionedDataset(FakeLeRobotDataset(episodes=[1, 3]))

        assert _frame(_task_goal(goals, 1)) == _last_frame(1)
        assert _frame(_task_goal(goals, 3)) == _last_frame(3)

    def test_goal_indices_are_built_without_decoding_images(self):
        dataset = FakeLeRobotDataset()
        goals = LeRobotGoalConditionedDataset(dataset)

        assert dataset.image_reads == 0
        _task_goal(goals, 0)
        _task_goal(goals, 2)
        assert dataset.image_reads == 2

    def test_goals_use_the_supplied_dataset_instance(self):
        dataset = FakeLeRobotDataset(episodes=[1, 3])

        goals = LeRobotGoalConditionedDataset(dataset)

        assert goals.dataset._lerobot_dataset is dataset
        assert dataset.episodes == [1, 3]

    def test_temporal_windows_are_rejected(self):
        with pytest.raises(ValueError, match="delta_timestamps=None"):
            LeRobotGoalConditionedDataset(FakeLeRobotDataset(delta_timestamps={"action": [0.0, 0.1]}))

    def test_uint8_images_are_not_rescaled_as_float(self):
        goals = LeRobotGoalConditionedDataset(FakeLeRobotDataset(image_dtype=torch.uint8, image_value=255))

        expected = torch.full((3, 2, 2), 255, dtype=torch.uint8)
        assert torch.equal(_task_goal(goals, 0)["top"], expected)

    def test_observation_and_goal_use_normal_transformed_reads(self):
        class AddCallCount:
            def __init__(self):
                self.calls = 0

            def __call__(self, image):
                self.calls += 1
                return image + self.calls

        transform = AddCallCount()
        raw_dataset = FakeLeRobotDataset(image_transforms=transform)
        goals = LeRobotGoalConditionedDataset(raw_dataset)
        observation = goals[0]

        assert transform.calls == 2
        assert raw_dataset.image_transforms is transform
        assert torch.equal(observation.images["top"], torch.full((3, 2, 2), 1.0))
        expected_goal = torch.full((3, 2, 2), _last_frame(0) / 255 + 2)
        assert torch.equal(observation.goal_images["top"], expected_goal)

    def test_unknown_task_raises(self):
        goals = LeRobotGoalConditionedDataset(FakeLeRobotDataset())

        with pytest.raises(KeyError, match="Task 'task seven' is not in the dataset"):
            goals.get_task_goal("task seven")

    def test_task_goal_is_read_again_on_each_lookup(self):
        goals = LeRobotGoalConditionedDataset(FakeLeRobotDataset())
        first = goals.get_task_goal("task a")
        first["top"].zero_()

        assert _frame(goals.get_task_goal("task a")) == _last_frame(0)

    def test_task_representative_episode_and_frame_can_be_configured(self):
        goals = LeRobotGoalConditionedDataset(
            FakeLeRobotDataset(),
            goal_source="task",
            task_goal_episodes={"task a": 1},
            episode_goal_frames={1: 0},
        )

        assert _frame(_task_goal(goals, 0)) == 3


class TestEpisodeGoals:
    def test_each_sample_uses_its_episode_goal(self):
        goals = LeRobotGoalConditionedDataset(FakeLeRobotDataset(), goal_source="episode")

        for episode in range(len(EPISODE_TASKS)):
            observation = goals[episode * EPISODE_LENGTH]
            assert _frame(observation.goal_images) == _last_frame(episode)

    def test_custom_episode_offsets(self):
        goals = LeRobotGoalConditionedDataset(
            FakeLeRobotDataset(),
            goal_source="episode",
            episode_goal_frames={0: 0, 1: 1, 2: 2, 3: 0},
        )

        assert [_frame(goals[episode * EPISODE_LENGTH].goal_images).item() for episode in range(4)] == [0, 4, 8, 9]

    def test_json_episode_offsets(self, tmp_path):
        path = tmp_path / "episode_goal_frames.json"
        path.write_text('{"0": 1, "1": -1, "2": 0, "3": 2}', encoding="utf-8")
        goals = LeRobotGoalConditionedDataset(
            FakeLeRobotDataset(),
            goal_source="episode",
            episode_goal_frames=path,
        )

        assert [_frame(goals[episode * EPISODE_LENGTH].goal_images).item() for episode in range(4)] == [1, 5, 6, 11]


@pytest.fixture
def fake_lerobot(monkeypatch):
    """Route every ``LeRobotDataset`` the code opens to the fake."""
    monkeypatch.setattr("physicalai.data.lerobot.dataset.LeRobotDataset", FakeLeRobotDataset)
    monkeypatch.setattr("physicalai.data.lerobot.datamodule.LeRobotDataset", FakeLeRobotDataset)


class TestTaskLookup:
    def test_matches_description_ignoring_formatting(self):
        goals = LeRobotGoalConditionedDataset(FakeLeRobotDataset())

        assert _frame(goals.get_task_goal("task a")) == _last_frame(0)
        assert _frame(goals.get_task_goal("  TASK B. ")) == _last_frame(2)

    def test_unknown_description_lists_dataset_tasks(self):
        goals = LeRobotGoalConditionedDataset(FakeLeRobotDataset())

        with pytest.raises(KeyError, match="task b"):
            goals.get_task_goal("task c")

    def test_ambiguous_normalized_descriptions_raise(self):
        dataset = FakeLeRobotDataset()
        dataset.meta.tasks = pd.DataFrame({"task_index": [0, 1]}, index=["Task A", " task a. "])

        with pytest.raises(ValueError, match="normalize to the same key"):
            LeRobotGoalConditionedDataset(dataset)


@pytest.mark.usefixtures("fake_lerobot")
class TestLeRobotGoalConditionedDataset:
    def test_filtered_wrapper_uses_only_its_own_task_goal(self):
        dataset = LeRobotGoalConditionedDataset(FakeLeRobotDataset(episodes=[3]))

        obs = dataset[0]

        assert obs.episode_index == 3
        assert _frame(obs.goal_images) == _last_frame(3)
        assert torch.equal(obs.goal_images["top"], torch.full((3, 2, 2), _last_frame(3) / 255))

    def test_wraps_adapter(self):
        adapter = _LeRobotDatasetAdapter.from_lerobot(FakeLeRobotDataset(episodes=[1]))

        dataset = LeRobotGoalConditionedDataset(adapter)

        assert dataset.dataset is adapter
        assert _frame(dataset[0].goal_images) == _last_frame(1)

    def test_behaves_like_the_wrapped_dataset(self):
        dataset = LeRobotGoalConditionedDataset(FakeLeRobotDataset(episodes=[0, 1]))
        dataset.delta_indices = {"action": [0, 1]}

        assert len(dataset) == 2 * EPISODE_LENGTH
        assert set(dataset.observation_features) == {"top", "state"}
        assert dataset.fps == 10
        assert dataset.dataset._lerobot_dataset.delta_indices == {"action": [0, 1]}

    def test_splits_resolve_task_goals_locally(self):
        train = LeRobotGoalConditionedDataset(FakeLeRobotDataset(episodes=[0, 2]))
        val = LeRobotGoalConditionedDataset(FakeLeRobotDataset(episodes=[1, 3]))

        assert _frame(train[0].goal_images) == _last_frame(0)
        assert _frame(val[0].goal_images) == _last_frame(1)


@pytest.mark.usefixtures("fake_lerobot")
class TestLeRobotDataModuleGoals:
    def test_no_goal_source_leaves_dataset_unwrapped(self):
        dm = LeRobotDataModule(repo_id="fake", train_batch_size=2)

        assert not isinstance(dm.train_dataset, LeRobotGoalConditionedDataset)
        assert dm.train_dataset[0].goal_images is None

    def test_task_goal_source_wraps_dataset(self):
        dm = LeRobotDataModule(repo_id="fake", episodes=[2, 3], train_batch_size=2, goal_source="task")

        assert isinstance(dm.train_dataset, LeRobotGoalConditionedDataset)
        assert _frame(dm.train_dataset[0].goal_images) == _last_frame(2)

    def test_episode_goal_source_wraps_dataset(self):
        dm = LeRobotDataModule(repo_id="fake", episodes=[1], goal_source="episode")

        assert _frame(dm.train_dataset[0].goal_images) == _last_frame(1)

    def test_task_goals_use_the_same_transform_pipeline(self):
        class AddOne:
            def __init__(self):
                self.calls = 0

            def __call__(self, image):
                self.calls += 1
                return image + 1

        transform = AddOne()
        dm = LeRobotDataModule(
            repo_id="fake",
            episodes=[0],
            train_batch_size=2,
            image_transforms=transform,
            goal_source="task",
        )

        observation = dm.train_dataset[0]

        assert transform.calls == 2
        assert torch.equal(observation.images["top"], torch.ones(3, 2, 2))
        assert torch.equal(
            observation.goal_images["top"],
            torch.full((3, 2, 2), _last_frame(0) / 255 + 1),
        )

    def test_prebuilt_transformed_dataset_transfers_transform_to_adapter(self):
        class AddOne:
            def __call__(self, image):
                return image + 1

        raw_dataset = FakeLeRobotDataset(episodes=[0], image_transforms=AddOne())
        dm = LeRobotDataModule(
            dataset=raw_dataset,
            goal_source="task",
            train_batch_size=2,
        )

        observation = dm.train_dataset[0]
        assert torch.equal(
            observation.goal_images["top"],
            torch.full((3, 2, 2), _last_frame(0) / 255 + 1),
        )
        assert torch.equal(observation.images["top"], torch.full((3, 2, 2), 1.0))
        assert raw_dataset.image_transforms is not None

    def test_accepts_wrapped_dataset(self):
        dataset = LeRobotGoalConditionedDataset(FakeLeRobotDataset())
        adapter = dataset.dataset
        adapter._state_columns = ["observation.state.joint_positions", "observation.state.gripper_position"]
        adapter._action_columns = ["action.joint_position", "action.gripper_position"]

        dm = LeRobotDataModule(dataset=dataset, train_batch_size=2)

        assert dm.train_dataset is dataset
        sample = dm.train_dataset[0]
        assert torch.equal(sample.state, torch.tensor([0.0, 1.0, 2.0]))
        assert torch.equal(sample.action, torch.tensor([0.0, 1.0, 2.0]))

    def test_wrapped_dataset_applies_explicit_columns(self):
        dataset = LeRobotGoalConditionedDataset(FakeLeRobotDataset())
        state_columns = ["observation.state.joint_positions", "observation.state.gripper_position"]
        action_columns = ["action.joint_position", "action.gripper_position"]

        dm = LeRobotDataModule(
            dataset=dataset,
            train_batch_size=2,
            state_columns=state_columns,
            action_columns=action_columns,
        )

        sample = dm.train_dataset[0]
        assert torch.equal(sample.state, torch.tensor([0.0, 1.0, 2.0]))
        assert torch.equal(sample.action, torch.tensor([0.0, 1.0, 2.0]))

    def test_val_split_uses_training_task_goals(self, monkeypatch):
        monkeypatch.setattr("physicalai.data.lerobot.datamodule._read_total_episodes", lambda *_: 4)
        dm = LeRobotDataModule(
            repo_id="fake",
            train_batch_size=2,
            val_split=0.25,
            val_split_seed=0,
            goal_source="task",
        )

        assert isinstance(dm.val_eval_dataset, LeRobotGoalConditionedDataset)
        assert _frame(dm.train_dataset[0].goal_images) == _last_frame(0)
        assert _frame(dm.val_eval_dataset[0].goal_images) == _last_frame(2)

    def test_val_split_rejects_task_missing_from_training(self, monkeypatch):
        monkeypatch.setattr("physicalai.data.lerobot.datamodule._read_total_episodes", lambda *_: 4)
        monkeypatch.setattr("physicalai.data.lerobot.datamodule.random.sample", lambda episodes, count: [0, 1])

        with pytest.raises(ValueError, match="no training goal"):
            LeRobotDataModule(
                repo_id="fake",
                train_batch_size=2,
                val_split=0.5,
                goal_source="task",
            )

    def test_raw_gyms_are_wrapped(self):
        dm = LeRobotDataModule(
            repo_id="fake",
            train_batch_size=2,
            goal_source="task",
            val_gym=FakeTaskGym("task a"),
            test_gym=FakeTaskGym("task b"),
            num_rollouts_val=1,
            num_rollouts_test=1,
        )

        assert isinstance(_unwrap_step_limit(dm.val_gym), GoalConditionedGym)
        assert isinstance(_unwrap_step_limit(dm.test_gym), GoalConditionedGym)

    def test_already_wrapped_gym_not_double_wrapped(self):
        dataset = LeRobotGoalConditionedDataset(FakeLeRobotDataset())
        wrapped = GoalConditionedGym(FakeTaskGym("task a"), dataset)
        dm = LeRobotDataModule(dataset=dataset, train_batch_size=2, val_gym=wrapped, num_rollouts_val=1)

        inner = _unwrap_step_limit(dm.val_gym)
        assert inner is wrapped
        assert not isinstance(inner.gym, GoalConditionedGym)

    def test_goal_source_with_wrapped_dataset_raises(self):
        dataset = LeRobotGoalConditionedDataset(FakeLeRobotDataset())

        with pytest.raises(ValueError, match="already carries goals"):
            LeRobotDataModule(dataset=dataset, goal_source="task")

    def test_lerobot_format_raises(self):
        with pytest.raises(ValueError, match="physicalai"):
            LeRobotDataModule(repo_id="fake", data_format="lerobot", goal_source="task")


def _unwrap_step_limit(gym):
    """Return the underlying gym past any step-limit wrapper applied by the DataModule."""
    from physicalai.gyms.step_limit import StepLimit

    return gym.gym if isinstance(gym, StepLimit) else gym


@pytest.mark.usefixtures("fake_lerobot")
class TestGoalBatching:
    @pytest.fixture
    def batch(self):
        dm = LeRobotDataModule(repo_id="fake", train_batch_size=4, num_workers=0, goal_source="task")
        dm.setup("fit")
        return next(iter(dm.train_dataloader()))

    def test_collate_stacks_goals(self, batch):
        assert batch.goal_images["top"].shape == (4, 3, 2, 2)

    def test_goals_match_their_task(self, batch):
        expected = [
            _last_frame(0) if EPISODE_TASKS[int(episode)] == "a" else _last_frame(2)
            for episode in batch.episode_index
        ]

        assert _frame(batch.goal_images).tolist() == expected

    def test_to_device(self, batch):
        batch = batch.to("cpu")

        assert batch.goal_images["top"].device.type == "cpu"

    def test_index_slices_goals(self, batch):
        sample = batch[1:2]

        assert sample.goal_images["top"].shape == (1, 3, 2, 2)
        assert torch.equal(sample.goal_images["top"], batch.goal_images["top"][1:2])

    def test_to_dict_uses_goal_field_names(self, batch):
        flat = batch.to_dict()

        assert f"{GOAL_IMAGES}.top" in flat


class FakeTaskGym(Gym):
    """Minimal gym that reports a task description, like PushT / LIBERO."""

    def __init__(self, task_description: str) -> None:
        self.task_description = task_description

    def _observation(self) -> Observation:
        return Observation(
            images={"top": torch.zeros(1, 3, 2, 2)},
            state=torch.zeros(1, 2),
            task=[self.task_description],
        )

    def reset(self, *, seed=None, episode_index=0, **reset_kwargs):
        return self._observation(), {}

    def step(self, action):
        return self._observation(), 0.0, False, False, {}

    def close(self) -> None:
        pass

    def sample_action(self) -> torch.Tensor:
        return torch.zeros(2)

    def to_observation(self, raw_obs) -> Observation:
        return self._observation()


@pytest.mark.usefixtures("fake_lerobot")
class TestGoalConditionedGymWithLeRobot:
    @pytest.mark.parametrize(("description", "task_episode"), [("task a", 0), ("Task B.", 2)])
    def test_goal_matches_gym_task(self, description, task_episode):
        dataset = LeRobotGoalConditionedDataset(FakeLeRobotDataset())
        gym = GoalConditionedGym(FakeTaskGym(description), dataset)

        obs, _ = gym.reset(seed=0)

        # Rollouts use the task's canonical goal (first selected episode).
        assert _frame(obs.goal_images).tolist() == [_last_frame(task_episode)]
        assert obs.goal_images["top"].shape == (1, 3, 2, 2)

    def test_accepts_dataset_as_task_goal_store(self):
        dataset = LeRobotGoalConditionedDataset(FakeLeRobotDataset())
        gym = GoalConditionedGym(FakeTaskGym("task a"), dataset)

        obs, _ = gym.reset(seed=0)

        assert _frame(obs.goal_images).tolist() == [_last_frame(0)]

    def test_unknown_task_fails_at_construction(self):
        dataset = LeRobotGoalConditionedDataset(FakeLeRobotDataset())
        with pytest.raises(KeyError, match="not in the dataset"):
            GoalConditionedGym(FakeTaskGym("task c"), dataset)

    def test_missing_task_description_raises(self):
        dataset = LeRobotGoalConditionedDataset(FakeLeRobotDataset())
        gym = GoalConditionedGym(FakeTaskGym("task a"), dataset)
        gym.gym.task_description = None

        with pytest.raises(ValueError, match="Observation.task"):
            gym.reset()
