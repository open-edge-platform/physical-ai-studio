# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for val_split functionality in LeRobotDataModule."""

from __future__ import annotations

import json
import random
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from physicalai.data import Observation
from physicalai.data.lerobot.datamodule import LeRobotDataModule, _read_total_episodes
from physicalai.train.utils import reformat_dataset_to_match_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeLeRobotDataset:
    """Minimal LeRobotDataset mock that records which episodes were requested."""

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        torch.manual_seed(idx)
        return {
            "observation.state": torch.randn(6),
            "action": torch.randn(6),
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(idx),
            "index": torch.tensor(idx),
            "task_index": torch.tensor(0),
            "timestamp": torch.tensor(float(idx) / 10.0),
        }

    @property
    def features(self) -> dict:
        return {
            "observation.state": {"shape": (6,), "dtype": "float32"},
            "action": {"shape": (6,), "dtype": "float32"},
        }

    @property
    def meta(self):
        class MockMeta:
            @property
            def features(self):
                return {
                    "observation.state": {"shape": (6,), "dtype": "float32", "names": ["dim"]},
                    "action": {"shape": (6,), "dtype": "float32", "names": ["dim"]},
                }

            @property
            def stats(self):
                return {
                    "observation.state": {
                        "mean": np.zeros(6),
                        "std": np.ones(6),
                        "min": np.full(6, -1.0),
                        "max": np.ones(6),
                        "q01": np.full(6, -1.0),
                        "q99": np.ones(6),
                    },
                    "action": {
                        "mean": np.zeros(6),
                        "std": np.ones(6),
                        "min": np.full(6, -1.0),
                        "max": np.ones(6),
                        "q01": np.full(6, -1.0),
                        "q99": np.ones(6),
                    },
                }

        return MockMeta()

    @property
    def fps(self) -> int:
        return 30

    @property
    def tolerance_s(self) -> float:
        return 1e-4

    def __init__(self, repo_id=None, root=None, episodes=None, **kwargs):
        self.repo_id = repo_id
        self.root = root
        self.requested_episodes = episodes
        self._length = 100
        self._delta_indices = None

    @property
    def delta_indices(self):
        return self._delta_indices

    @delta_indices.setter
    def delta_indices(self, value):
        self._delta_indices = value


def _create_local_dataset(tmp_path: Path, total_episodes: int = 20) -> Path:
    """Create a minimal local LeRobot dataset directory with meta/info.json."""
    dataset_dir = tmp_path / "test_dataset"
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True)
    info = {"total_episodes": total_episodes, "codebase_version": "v2.1"}
    (meta_dir / "info.json").write_text(json.dumps(info))
    return dataset_dir


# ---------------------------------------------------------------------------
# Tests for _read_total_episodes
# ---------------------------------------------------------------------------


class TestReadTotalEpisodes:
    """Tests for the _read_total_episodes helper."""

    def test_local_dataset(self, tmp_path: Path):
        """Reads total_episodes from a local dataset's info.json."""
        dataset_dir = _create_local_dataset(tmp_path, total_episodes=34)
        result = _read_total_episodes("ignored_repo_id", root=str(dataset_dir))
        assert result == 34

    def test_local_dataset_missing(self, tmp_path: Path):
        """Raises FileNotFoundError when local info.json is missing."""
        missing_dir = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="Cannot read dataset metadata"):
            _read_total_episodes("any_repo", root=str(missing_dir))

    def test_hf_dataset_downloads_info(self, tmp_path: Path, monkeypatch):
        """Downloads info.json from HuggingFace when not cached locally."""
        # Fake HF_LEROBOT_HOME to use tmp_path
        monkeypatch.setattr(
            "physicalai.data.lerobot.datamodule.HF_LEROBOT_HOME", tmp_path
        )

        repo_id = "lerobot/pusht"
        expected_base = tmp_path / repo_id

        # Mock hf_hub_download to create the file instead of actually downloading
        def fake_download(repo_id, repo_type, filename, local_dir):
            meta_dir = Path(local_dir) / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / "info.json").write_text(json.dumps({"total_episodes": 206}))

        monkeypatch.setattr(
            "physicalai.data.lerobot.datamodule.hf_hub_download", fake_download
        )

        result = _read_total_episodes(repo_id, root=None)
        assert result == 206


# ---------------------------------------------------------------------------
# Tests for LeRobotDataModule val_split
# ---------------------------------------------------------------------------


class TestValSplit:
    """Tests for val_split in LeRobotDataModule."""

    @pytest.fixture(autouse=True)
    def _patch_lerobot(self, monkeypatch):
        """Patch LeRobotDataset to avoid real downloads/ffmpeg."""
        monkeypatch.setattr(
            "physicalai.data.lerobot.dataset.LeRobotDataset", FakeLeRobotDataset
        )

    def test_val_split_disabled(self, tmp_path: Path):
        """val_split=0 creates no validation dataset."""
        dataset_dir = _create_local_dataset(tmp_path, total_episodes=20)
        dm = LeRobotDataModule(
            root=str(dataset_dir),
            train_batch_size=4,
            val_split=0.0,
        )
        assert dm.val_eval_dataset is None

    def test_val_split_local_dataset(self, tmp_path: Path):
        """val_split creates train/val datasets from local folder."""
        dataset_dir = _create_local_dataset(tmp_path, total_episodes=20)
        dm = LeRobotDataModule(
            root=str(dataset_dir),
            train_batch_size=4,
            val_split=0.2,
        )
        assert dm.val_eval_dataset is not None
        # Check that episodes are split (20 total, 20% = 4 val)
        train_eps = dm.train_dataset._lerobot_dataset.requested_episodes
        val_eps = dm.val_eval_dataset._lerobot_dataset.requested_episodes
        assert len(val_eps) == 4
        assert len(train_eps) == 16
        # No overlap
        assert set(train_eps).isdisjoint(set(val_eps))
        # Union covers all episodes
        assert sorted(train_eps + val_eps) == list(range(20))

    def test_val_split_hf_repo(self, tmp_path: Path, monkeypatch):
        """val_split works with HuggingFace repo_id (downloads info.json)."""
        monkeypatch.setattr(
            "physicalai.data.lerobot.datamodule.HF_LEROBOT_HOME", tmp_path
        )

        def fake_download(repo_id, repo_type, filename, local_dir):
            meta_dir = Path(local_dir) / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / "info.json").write_text(json.dumps({"total_episodes": 50}))

        monkeypatch.setattr(
            "physicalai.data.lerobot.datamodule.hf_hub_download", fake_download
        )

        dm = LeRobotDataModule(
            repo_id="lerobot/pusht",
            train_batch_size=4,
            val_split=0.1,
        )
        assert dm.val_eval_dataset is not None
        train_eps = dm.train_dataset._lerobot_dataset.requested_episodes
        val_eps = dm.val_eval_dataset._lerobot_dataset.requested_episodes
        # 50 total, 10% = 5 val
        assert len(val_eps) == 5
        assert len(train_eps) == 45
        assert set(train_eps).isdisjoint(set(val_eps))

    def test_val_split_random_selection(self, tmp_path: Path):
        """Val episodes are randomly selected, not just the last N."""
        dataset_dir = _create_local_dataset(tmp_path, total_episodes=20)
        dm = LeRobotDataModule(
            root=str(dataset_dir),
            train_batch_size=4,
            val_split=0.2,
            val_split_seed=42,
        )
        val_eps = dm.val_eval_dataset._lerobot_dataset.requested_episodes
        # With seed=42 and 20 episodes, the val episodes should NOT be [16,17,18,19]
        # (which would be a sequential "last N" selection)
        assert val_eps != list(range(16, 20))

    def test_val_split_deterministic(self, tmp_path: Path):
        """Same val_split_seed produces the same split every time."""
        dataset_dir = _create_local_dataset(tmp_path, total_episodes=20)

        dm1 = LeRobotDataModule(root=str(dataset_dir), train_batch_size=4, val_split=0.2, val_split_seed=42)
        dm2 = LeRobotDataModule(root=str(dataset_dir), train_batch_size=4, val_split=0.2, val_split_seed=42)

        eps1 = dm1.val_eval_dataset._lerobot_dataset.requested_episodes
        eps2 = dm2.val_eval_dataset._lerobot_dataset.requested_episodes
        assert eps1 == eps2

    def test_val_split_seed_changes_split(self, tmp_path: Path):
        """Different val_split_seed produces a different split."""
        dataset_dir = _create_local_dataset(tmp_path, total_episodes=20)

        dm1 = LeRobotDataModule(root=str(dataset_dir), train_batch_size=4, val_split=0.2, val_split_seed=42)
        dm2 = LeRobotDataModule(root=str(dataset_dir), train_batch_size=4, val_split=0.2, val_split_seed=99)

        eps1 = dm1.val_eval_dataset._lerobot_dataset.requested_episodes
        eps2 = dm2.val_eval_dataset._lerobot_dataset.requested_episodes
        assert eps1 != eps2

    def test_val_split_respects_global_seed(self, tmp_path: Path):
        """Default (no val_split_seed) uses global random, respecting seed_everything."""
        dataset_dir = _create_local_dataset(tmp_path, total_episodes=20)

        random.seed(123)
        dm1 = LeRobotDataModule(root=str(dataset_dir), train_batch_size=4, val_split=0.2)
        eps1 = dm1.val_eval_dataset._lerobot_dataset.requested_episodes

        random.seed(123)
        dm2 = LeRobotDataModule(root=str(dataset_dir), train_batch_size=4, val_split=0.2)
        eps2 = dm2.val_eval_dataset._lerobot_dataset.requested_episodes

        assert eps1 == eps2

    def test_val_split_minimum_one_episode(self, tmp_path: Path):
        """Even with tiny split, at least 1 val episode is created."""
        dataset_dir = _create_local_dataset(tmp_path, total_episodes=5)
        dm = LeRobotDataModule(
            root=str(dataset_dir),
            train_batch_size=4,
            val_split=0.01,  # 0.01 * 5 = 0.05 → max(1, 0) = 1
        )
        val_eps = dm.val_eval_dataset._lerobot_dataset.requested_episodes
        assert len(val_eps) == 1

    def test_val_split_with_explicit_episodes(self, tmp_path: Path):
        """val_split works when user provides a subset of episodes."""
        dataset_dir = _create_local_dataset(tmp_path, total_episodes=20)
        dm = LeRobotDataModule(
            root=str(dataset_dir),
            train_batch_size=4,
            episodes=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18],  # 10 episodes
            val_split=0.2,
        )
        train_eps = dm.train_dataset._lerobot_dataset.requested_episodes
        val_eps = dm.val_eval_dataset._lerobot_dataset.requested_episodes
        # 10 episodes provided, 20% = 2 val
        assert len(val_eps) == 2
        assert len(train_eps) == 8
        # All episodes come from the provided list
        all_eps = set(train_eps) | set(val_eps)
        assert all_eps.issubset({0, 2, 4, 6, 8, 10, 12, 14, 16, 18})


# ---------------------------------------------------------------------------
# Tests for validation errors
# ---------------------------------------------------------------------------


class TestValSplitValidation:
    """Tests for val_split validation errors."""

    def test_invalid_val_split_negative(self, tmp_path: Path):
        """Raises ValueError for negative val_split."""
        dataset_dir = _create_local_dataset(tmp_path)
        with pytest.raises(ValueError, match="val_split.*must be in"):
            LeRobotDataModule(root=str(dataset_dir), val_split=-0.1)

    def test_invalid_val_split_one(self, tmp_path: Path):
        """Raises ValueError for val_split >= 1."""
        dataset_dir = _create_local_dataset(tmp_path)
        with pytest.raises(ValueError, match="val_split.*must be in"):
            LeRobotDataModule(root=str(dataset_dir), val_split=1.0)

    def test_val_split_with_dataset_raises(self, monkeypatch):
        """Raises ValueError when using val_split with a pre-built dataset."""
        monkeypatch.setattr(
            "physicalai.data.lerobot.dataset.LeRobotDataset", FakeLeRobotDataset
        )
        monkeypatch.setattr(
            "physicalai.data.lerobot.datamodule.LeRobotDataset", FakeLeRobotDataset
        )
        fake_ds = FakeLeRobotDataset(repo_id="test")
        with pytest.raises(ValueError, match="Cannot use 'val_split' with a pre-initialized"):
            LeRobotDataModule(dataset=fake_ds, val_split=0.1)

    def test_val_split_with_val_gym_raises(self, tmp_path: Path, monkeypatch):
        """Raises ValueError when using both val_split and val_gym."""
        monkeypatch.setattr(
            "physicalai.data.lerobot.dataset.LeRobotDataset", FakeLeRobotDataset
        )
        dataset_dir = _create_local_dataset(tmp_path)
        mock_gym = MagicMock()
        with pytest.raises(ValueError, match="Cannot use both 'val_split' and 'val_gym'"):
            LeRobotDataModule(
                root=str(dataset_dir),
                val_split=0.1,
                val_gym=mock_gym,
            )


# ---------------------------------------------------------------------------
# Tests for reformat_dataset_to_match_policy with val_eval_dataset
# ---------------------------------------------------------------------------


class TestReformatIncludesValDataset:
    """Tests that reformat_dataset_to_match_policy also reformats val_eval_dataset."""

    @pytest.fixture(autouse=True)
    def _patch_lerobot(self, monkeypatch):
        """Patch LeRobotDataset to avoid real downloads/ffmpeg."""
        monkeypatch.setattr(
            "physicalai.data.lerobot.dataset.LeRobotDataset", FakeLeRobotDataset
        )

    def test_val_eval_dataset_gets_delta_indices(self, tmp_path: Path):
        """reformat_dataset_to_match_policy sets delta_indices on val_eval_dataset."""
        dataset_dir = _create_local_dataset(tmp_path, total_episodes=20)
        dm = LeRobotDataModule(
            root=str(dataset_dir),
            train_batch_size=4,
            val_split=0.2,
        )

        # Both datasets should start without delta_indices
        assert dm.train_dataset.delta_indices == {}
        assert dm.val_eval_dataset.delta_indices == {}

        # Create a mock policy with action_delta_indices
        mock_policy = MagicMock()
        mock_policy.lerobot_policy = None
        mock_policy.model.action_delta_indices = list(range(50))
        mock_policy.model.observation_delta_indices = None
        mock_policy.model.reward_delta_indices = None

        reformat_dataset_to_match_policy(mock_policy, dm)

        # Both datasets should now have delta_indices set
        assert "action" in dm.train_dataset.delta_indices
        assert "action" in dm.val_eval_dataset.delta_indices
        assert dm.train_dataset.delta_indices == dm.val_eval_dataset.delta_indices

    def test_no_val_eval_dataset_still_works(self, tmp_path: Path):
        """reformat_dataset_to_match_policy works when val_eval_dataset is None."""
        dataset_dir = _create_local_dataset(tmp_path, total_episodes=20)
        dm = LeRobotDataModule(
            root=str(dataset_dir),
            train_batch_size=4,
            val_split=0.0,
        )
        assert dm.val_eval_dataset is None

        mock_policy = MagicMock()
        mock_policy.lerobot_policy = None
        mock_policy.model.action_delta_indices = list(range(50))
        mock_policy.model.observation_delta_indices = None
        mock_policy.model.reward_delta_indices = None

        # Should not raise
        reformat_dataset_to_match_policy(mock_policy, dm)
        assert "action" in dm.train_dataset.delta_indices
