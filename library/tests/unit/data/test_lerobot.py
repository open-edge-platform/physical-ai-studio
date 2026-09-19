# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test for lerobot dataset using a mock to avoid ffmpeg/network dependencies."""

import numpy as np
import pytest
import torch

from physicalai.data import Dataset, Observation
from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter, _concat_columns


class FakeLeRobotDataset:
    """A mock that mimics LeRobotDataset without needing ffmpeg or network access."""
    def __init__(self, repo_id=None, episodes=None, **kwargs):
        """Accepts arguments but does nothing with them."""
        self._length = 150  # A fixed length for our mock dataset

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        """Returns a fake data dictionary, similar to the real dataset."""
        if idx >= self._length:
            raise IndexError("Index out of range")
        torch.manual_seed(idx)
        return {
            "observation.images.wrist": torch.randn(3, 64, 64),
            "observation.state": torch.randn(8),
            "action": torch.randn(7),
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(idx),
            "index": torch.tensor(idx),
            "task.instructions": "pusht",
            "task_index": torch.tensor(0),
            "timestamp": torch.tensor(float(idx) / 10.0),
            "random": ["thing"]
        }

    @property
    def features(self) -> dict[str, dict]:
        """Mock features property."""
        return {
            "observation.state": {
                "shape": (8,), "dtype": "float32",
            },
            "observation.action": {
                "shape": (7,), "dtype": "int64"
            },
        }

    @property
    def meta(self):
        """Mock meta property."""
        class MockMeta:
            @property
            def features(self):
                return {
                            "observation.state": {"shape": (8,), "dtype": "float32"},
                            "observation.action": {"shape": (7,), "dtype": "int64"},
                        }
            @property
            def stats(self):
                return {
                    "observation.state": {
                        "mean": np.zeros(8),
                        "std": np.ones(8),
                        "min": np.full(8, -1.0),
                        "max": np.ones(8),
                    },
                    "observation.action": {
                        "mean": np.zeros(7),
                        "std": np.ones(7),
                        "min": np.full(7, -1.0),
                        "max": np.ones(7),
                    },
                }

        return MockMeta()


class FakeLeRobotDataset2:
    """A mock that mimics LeRobotDataset without needing ffmpeg or network access."""
    def __init__(self, repo_id=None, episodes=None, **kwargs):
        """Accepts arguments but does nothing with them."""
        self._length = 150  # A fixed length for our mock dataset

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        """Returns a fake data dictionary, similar to the real dataset."""
        if idx >= self._length:
            raise IndexError("Index out of range")
        torch.manual_seed(idx)
        return {
            "observation.images.wrist": torch.randn(3, 64, 64),
            "observation.state": torch.randn(8),
            "action.continuous": torch.randn(7),
            "action.discrete": torch.randint(low=0, high=10, size=(7,)),
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(idx),
            "index": torch.tensor(idx),
            "task.instructions": "pusht",
            "task_index": torch.tensor(0),
            "timestamp": torch.tensor(float(idx) / 10.0),
        }


class FakeLeRobotDataset_no_task_or_image:
    """A mock that mimics LeRobotDataset without needing ffmpeg or network access."""
    def __init__(self, repo_id=None, episodes=None, **kwargs):
        """Accepts arguments but does nothing with them."""
        self._length = 150  # A fixed length for our mock dataset

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        """Returns a fake data dictionary, similar to the real dataset."""
        if idx >= self._length:
            raise IndexError("Index out of range")
        torch.manual_seed(idx)
        return {
            "observation.state": torch.randn(8),
            "action": torch.randint(low=0, high=10, size=(7,)),
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(idx),
            "index": torch.tensor(idx),
            "task_index": torch.tensor(0),
            "timestamp": torch.tensor(float(idx) / 10.0),
        }


@pytest.mark.parametrize(
    "dataset_cls",
    [FakeLeRobotDataset, FakeLeRobotDataset2, FakeLeRobotDataset_no_task_or_image],
)
class TestLeRobotActionDataset:
    """Groups tests for the LeRobotActionDataset wrapper, using multiple mock datasets."""

    @pytest.fixture
    def raw_lerobot_dataset(self, dataset_cls):
        """Fixture to provide a mock dataset instance for the current parameter."""
        return dataset_cls()

    def test_initialization(self, monkeypatch, dataset_cls):
        """Tests that LeRobotActionDataset initializes correctly by patching."""
        monkeypatch.setattr(
            "physicalai.data.lerobot.dataset.LeRobotDataset", dataset_cls
        )

        dataset = _LeRobotDatasetAdapter(repo_id="any/repo", episodes=[0])

        assert isinstance(dataset, Dataset)
        assert isinstance(dataset._lerobot_dataset, dataset_cls)
        assert len(dataset) > 0

    def test_len_delegation(self, raw_lerobot_dataset):
        """Tests that __len__ correctly delegates to the mock dataset."""
        action_dataset = _LeRobotDatasetAdapter.from_lerobot(raw_lerobot_dataset)
        assert len(action_dataset) == len(raw_lerobot_dataset)
        assert len(action_dataset) == 150

    def test_getitem_returns_observation(self, raw_lerobot_dataset):
        """Tests that __getitem__ returns a correctly formatted Observation object."""
        action_dataset = _LeRobotDatasetAdapter.from_lerobot(raw_lerobot_dataset)
        observation = action_dataset[5]

        assert isinstance(observation, Observation), "Returned object must be Observation"

        # Images may or may not exist depending on dataset variant
        if "observation.images.wrist" in raw_lerobot_dataset[0]:
            assert isinstance(observation.images, dict)
            assert "wrist" in observation.images
        else:
            assert observation.images == {}

        # Episode index should always be present
        assert observation.episode_index == 0

    def test_from_lerobot_factory_method(self, raw_lerobot_dataset):
        """Tests the `from_lerobot` static method with a mock instance."""
        action_dataset = _LeRobotDatasetAdapter.from_lerobot(raw_lerobot_dataset)

        assert action_dataset._lerobot_dataset is raw_lerobot_dataset

        observation = action_dataset[0]
        raw_item = raw_lerobot_dataset[0]

        # Action may be continuous or discrete
        if "action" in raw_item:
            assert torch.equal(observation.action, raw_item["action"])
        elif "action.continuous" in raw_item:
            assert torch.equal(observation.action["continuous"], raw_item["action.continuous"])
        else:
            raise AssertionError("No recognizable action field in mock dataset")


class TestLeRobotActionDatasetFeatures:
    def test_observation_features_meta_retrieval(self):
        """Tests that features metadata can be retrieved from the adapter."""
        action_dataset = _LeRobotDatasetAdapter.from_lerobot(FakeLeRobotDataset())
        obs_features = action_dataset.observation_features

        assert isinstance(obs_features, dict), "Observation features should be a dictionary"

        for k in obs_features:
            assert not k.startswith("observation."), "Keys should not have 'observation.' prefix"


class FakeSplitColumnDataset:
    """Mock of a DROID-style dataset with split state/action sub-columns and names=None video."""

    STATE_COLUMNS = ["observation.state.joint_positions", "observation.state.gripper_position"]
    ACTION_COLUMNS = ["action.joint_position", "action.gripper_position"]

    def __init__(self, repo_id=None, episodes=None, **kwargs):
        """Accepts arguments but does nothing with them."""
        self._length = 10

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        """Return a fake sample with split columns and a singular 'observation.image.' camera prefix."""
        if idx >= self._length:
            raise IndexError("Index out of range")
        return {
            "observation.state.joint_positions": torch.arange(7, dtype=torch.float32),
            "observation.state.gripper_position": torch.tensor([7.0]),
            "observation.state.joint_velocities": torch.zeros(7),  # extra, must be dropped
            "observation.image.wrist_image_left": torch.randn(3, 64, 64),
            "action.joint_position": torch.arange(10, 17, dtype=torch.float32),
            "action.gripper_position": torch.tensor([17.0]),
            "action.joint_velocity": torch.zeros(7),  # extra, must be dropped
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(idx),
            "index": torch.tensor(idx),
            "task_index": torch.tensor(0),
            "timestamp": torch.tensor(float(idx) / 10.0),
        }

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def meta(self):
        class MockMeta:
            @property
            def features(self):
                return {
                    "observation.state.joint_positions": {"shape": (7,), "dtype": "float32"},
                    "observation.state.gripper_position": {"shape": (1,), "dtype": "float32"},
                    "observation.state.joint_velocities": {"shape": (7,), "dtype": "float32"},
                    "observation.image.wrist_image_left": {"shape": (64, 64, 3), "dtype": "video", "names": None},
                    "action.joint_position": {"shape": (7,), "dtype": "float32"},
                    "action.gripper_position": {"shape": (1,), "dtype": "float32"},
                    "action.joint_velocity": {"shape": (7,), "dtype": "float32"},
                }

            @property
            def stats(self):
                def _stats(dim, base):
                    return {
                        "mean": np.full(dim, base),
                        "std": np.ones(dim),
                        "min": np.full(dim, base - 1.0),
                        "max": np.full(dim, base + 1.0),
                    }

                return {
                    "observation.state.joint_positions": _stats(7, 1.0),
                    "observation.state.gripper_position": _stats(1, 2.0),
                    "observation.state.joint_velocities": _stats(7, 0.0),
                    "observation.image.wrist_image_left": _stats(3, 0.5),
                    "action.joint_position": _stats(7, 3.0),
                    "action.gripper_position": _stats(1, 4.0),
                    "action.joint_velocity": _stats(7, 0.0),
                }

        return MockMeta()


class TestSplitColumnCombining:
    """Tests for combining split state/action sub-columns into single combined columns."""

    def _adapter(self):
        return _LeRobotDatasetAdapter.from_lerobot(
            FakeSplitColumnDataset(),
            state_columns=FakeSplitColumnDataset.STATE_COLUMNS,
            action_columns=FakeSplitColumnDataset.ACTION_COLUMNS,
        )

    def test_getitem_combines_state_and_action(self):
        """State/action sub-columns are concatenated in order into single 8-D tensors."""
        obs = self._adapter()[0]

        assert isinstance(obs.state, torch.Tensor)
        assert obs.state.shape == (8,)
        assert torch.equal(obs.state, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))

        assert isinstance(obs.action, torch.Tensor)
        assert obs.action.shape == (8,)
        assert torch.equal(obs.action, torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]))

    def test_extra_subcolumns_are_dropped(self):
        """Unlisted sub-columns (velocities) are dropped, not leaked into extra."""
        obs = self._adapter()[0]
        assert "observation.state.joint_velocities" not in (obs.extra or {})
        assert "action.joint_velocity" not in (obs.extra or {})

    def test_singular_image_prefix_collected(self):
        """A singular 'observation.image.<cam>' key is collected as a camera image."""
        obs = self._adapter()[0]
        assert isinstance(obs.images, dict)
        assert "wrist_image_left" in obs.images

    def test_combined_features_shapes(self):
        """Combined observation.state and action features report the summed width."""
        adapter = self._adapter()
        assert adapter.observation_features["state"].shape == (8,)
        assert adapter.action_features["action"].shape == (8,)
        # Individual split sub-columns are no longer exposed as separate features.
        assert "state.joint_positions" not in adapter.observation_features
        assert "action.joint_position" not in adapter.action_features

    def test_combined_feature_stats_concatenated(self):
        """Per-column normalization stats are concatenated in column order."""
        state_feature = self._adapter().observation_features["state"]
        norm = state_feature.normalization_data
        # joint_positions min = 0.0 (x7), gripper_position min = 1.0 (x1)
        assert norm.min == [0.0] * 7 + [1.0]
        assert norm.max == [2.0] * 7 + [3.0]

    def test_video_names_none_channel_last_fallback(self):
        """names=None video features are inferred as channel-last and reported as (c, h, w)."""
        image_feature = self._adapter().observation_features["wrist_image_left"]
        assert image_feature.shape == (3, 64, 64)

    def test_concat_columns_windowed_trajectory(self):
        """Windowed columns with unequal rank (e.g. [T, 7] and [T]) are concatenated into [T, 8]."""
        item = {
            "observation.state.joint_positions": torch.zeros(32, 7),
            "observation.state.gripper_position": torch.ones(32),
        }
        res = _concat_columns(
            item,
            ["observation.state.joint_positions", "observation.state.gripper_position"],
            "observation.state",
            "observation.state.",
        )
        assert res["observation.state"].shape == (32, 8)
        assert torch.equal(res["observation.state"][:, -1], torch.ones(32))

