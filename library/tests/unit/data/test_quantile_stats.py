# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantile statistics computation.

Validates that PSA's ``augment_dataset_quantile_stats`` delegates to
LeRobot's implementation and produces identical results.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch


class _FakeLeRobotDataset:
    """Minimal mock of ``LeRobotDataset`` for quantile stats tests.

    Provides the attributes required by both ``has_quantile_stats`` and
    LeRobot's ``compute_quantile_stats_for_dataset``:
    ``meta.stats``, ``meta.episodes``, ``meta.video_keys``,
    ``features``, ``num_episodes``, ``__len__``, ``__getitem__``.
    """

    def __init__(self, num_episodes: int = 3, frames_per_episode: int = 20, seed: int = 42) -> None:
        rng = np.random.RandomState(seed)
        self._frames_per_episode = frames_per_episode
        self._num_episodes = num_episodes
        total = num_episodes * frames_per_episode

        # Pre-generate deterministic data
        self._state_data = torch.from_numpy(rng.randn(total, 4).astype(np.float32))
        self._action_data = torch.from_numpy(rng.randn(total, 2).astype(np.float32))

        episodes = {}
        for ep in range(num_episodes):
            episodes[ep] = {
                "dataset_from_index": ep * frames_per_episode,
                "dataset_to_index": (ep + 1) * frames_per_episode,
            }

        features_dict = {
            "observation.state": {"shape": (4,), "dtype": "float32"},
            "action": {"shape": (2,), "dtype": "float32"},
            "episode_index": {"shape": (), "dtype": "int64"},
            "frame_index": {"shape": (), "dtype": "int64"},
        }

        stats = {
            "observation.state": {
                "mean": torch.zeros(4),
                "std": torch.ones(4),
                "min": torch.full((4,), -3.0),
                "max": torch.full((4,), 3.0),
            },
            "action": {
                "mean": torch.zeros(2),
                "std": torch.ones(2),
                "min": torch.full((2,), -3.0),
                "max": torch.full((2,), 3.0),
            },
        }

        self.meta = SimpleNamespace(
            episodes=episodes,
            video_keys=[],
            features=features_dict,
            stats=stats,
        )

    @property
    def features(self) -> dict:
        return self.meta.features

    @property
    def num_episodes(self) -> int:
        return self._num_episodes

    def __len__(self) -> int:
        return self._num_episodes * self._frames_per_episode

    def __getitem__(self, idx: int) -> dict:
        return {
            "observation.state": self._state_data[idx],
            "action": self._action_data[idx],
            "episode_index": torch.tensor(idx // self._frames_per_episode),
            "frame_index": torch.tensor(idx % self._frames_per_episode),
        }


class TestHasQuantileStats:
    """Tests for ``has_quantile_stats``."""

    def test_returns_false_when_missing(self) -> None:
        ds = _FakeLeRobotDataset()
        from physicalai.data.lerobot.utils.quantile_stats import has_quantile_stats

        assert has_quantile_stats(ds) is False

    def test_returns_true_when_present(self) -> None:
        ds = _FakeLeRobotDataset()
        ds.meta.stats["action"]["q01"] = torch.zeros(2)
        ds.meta.stats["action"]["q99"] = torch.ones(2)

        from physicalai.data.lerobot.utils.quantile_stats import has_quantile_stats

        assert has_quantile_stats(ds) is True


class TestAugmentDatasetQuantileStats:
    """Tests for ``augment_dataset_quantile_stats``."""

    def test_injects_q01_q99_into_stats(self) -> None:
        """Verify that q01 and q99 are injected into meta.stats."""
        ds = _FakeLeRobotDataset()

        from physicalai.data.lerobot.utils.quantile_stats import augment_dataset_quantile_stats

        augment_dataset_quantile_stats(ds)

        for key in ("observation.state", "action"):
            assert "q01" in ds.meta.stats[key], f"q01 missing for {key}"
            assert "q99" in ds.meta.stats[key], f"q99 missing for {key}"
            assert isinstance(ds.meta.stats[key]["q01"], torch.Tensor)
            assert isinstance(ds.meta.stats[key]["q99"], torch.Tensor)

    def test_q01_less_than_q99(self) -> None:
        """Verify q01 < q99 for all features."""
        ds = _FakeLeRobotDataset()

        from physicalai.data.lerobot.utils.quantile_stats import augment_dataset_quantile_stats

        augment_dataset_quantile_stats(ds)

        for key in ("observation.state", "action"):
            q01 = ds.meta.stats[key]["q01"]
            q99 = ds.meta.stats[key]["q99"]
            assert (q01 < q99).all(), f"q01 not less than q99 for {key}"

    def test_preserves_existing_stats(self) -> None:
        """Verify that existing mean/std/min/max are not overwritten."""
        ds = _FakeLeRobotDataset()
        original_mean = ds.meta.stats["action"]["mean"].clone()
        original_std = ds.meta.stats["action"]["std"].clone()

        from physicalai.data.lerobot.utils.quantile_stats import augment_dataset_quantile_stats

        augment_dataset_quantile_stats(ds)

        torch.testing.assert_close(ds.meta.stats["action"]["mean"], original_mean)
        torch.testing.assert_close(ds.meta.stats["action"]["std"], original_std)

    def test_delegates_to_lerobot(self) -> None:
        """Verify that PSA delegates to LeRobot's compute function."""
        ds = _FakeLeRobotDataset()

        with patch(
            "lerobot.scripts.augment_dataset_quantile_stats.compute_quantile_stats_for_dataset"
        ) as mock_compute:
            # Return fake stats matching what LeRobot would produce
            mock_compute.return_value = {
                "observation.state": {
                    "q01": np.zeros(4, dtype=np.float32),
                    "q99": np.ones(4, dtype=np.float32),
                },
                "action": {
                    "q01": np.zeros(2, dtype=np.float32),
                    "q99": np.ones(2, dtype=np.float32),
                },
            }

            from physicalai.data.lerobot.utils.quantile_stats import augment_dataset_quantile_stats

            augment_dataset_quantile_stats(ds)

            mock_compute.assert_called_once_with(ds)

    def test_deterministic(self) -> None:
        """Verify that results are deterministic with the same seed."""
        ds1 = _FakeLeRobotDataset(seed=42)
        ds2 = _FakeLeRobotDataset(seed=42)

        from physicalai.data.lerobot.utils.quantile_stats import augment_dataset_quantile_stats

        augment_dataset_quantile_stats(ds1)
        augment_dataset_quantile_stats(ds2)

        for key in ("observation.state", "action"):
            torch.testing.assert_close(ds1.meta.stats[key]["q01"], ds2.meta.stats[key]["q01"])
            torch.testing.assert_close(ds1.meta.stats[key]["q99"], ds2.meta.stats[key]["q99"])

    def test_matches_lerobot_directly(self) -> None:
        """Verify PSA produces the same q01/q99 as calling LeRobot directly."""
        from lerobot.scripts.augment_dataset_quantile_stats import compute_quantile_stats_for_dataset

        from physicalai.data.lerobot.utils.quantile_stats import augment_dataset_quantile_stats

        ds_psa = _FakeLeRobotDataset(seed=123)
        ds_lerobot = _FakeLeRobotDataset(seed=123)

        # Compute via PSA
        augment_dataset_quantile_stats(ds_psa)

        # Compute via LeRobot directly
        lerobot_stats = compute_quantile_stats_for_dataset(ds_lerobot)

        for key in ("observation.state", "action"):
            expected_q01 = torch.from_numpy(lerobot_stats[key]["q01"]).float()
            expected_q99 = torch.from_numpy(lerobot_stats[key]["q99"]).float()

            torch.testing.assert_close(ds_psa.meta.stats[key]["q01"], expected_q01)
            torch.testing.assert_close(ds_psa.meta.stats[key]["q99"], expected_q99)
