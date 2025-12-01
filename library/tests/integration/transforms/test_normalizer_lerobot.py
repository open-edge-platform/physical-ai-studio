# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Normalizer with LeRobot datasets."""

from __future__ import annotations

import pytest
import torch

from getiaction.transforms import NormalizationMode, Normalizer

pytest.importorskip("lerobot", reason="lerobot not installed")


class TestNormalizerLeRobotIntegration:
    """Integration tests with LeRobot datasets."""

    @pytest.fixture
    def lerobot_datamodule(self):
        """Create a LeRobot datamodule for testing."""
        from getiaction.data.lerobot import LeRobotDataModule

        return LeRobotDataModule(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            train_batch_size=4,
            data_format="getiaction",
            episodes=[0],  # Just first episode for speed
        )

    def test_normalizer_from_lerobot_features(self, lerobot_datamodule) -> None:
        """Test creating normalizer from LeRobot dataset features."""
        # Setup the datamodule to get access to the dataset
        lerobot_datamodule.setup("fit")
        dataset = lerobot_datamodule.train_dataset

        # Get features from dataset
        observation_features = dataset.observation_features
        action_features = dataset.action_features

        # Merge all features
        all_features = {**observation_features, **action_features}

        # Create normalizer
        normalizer = Normalizer(features=all_features, norm_mode=NormalizationMode.MEAN_STD)

        # Verify it's single-embodiment
        assert not normalizer.is_multi_embodiment
        assert normalizer.embodiments == ["default"]

        # Verify feature names are registered
        feature_names = normalizer.get_feature_names()
        assert len(feature_names) > 0

        # Check that buffers were created for action
        assert hasattr(normalizer, "default_action_mean")
        assert hasattr(normalizer, "default_action_std")

    def test_normalizer_normalize_batch(self, lerobot_datamodule) -> None:
        """Test normalizing an actual batch from LeRobot."""
        lerobot_datamodule.setup("fit")
        dataset = lerobot_datamodule.train_dataset

        # Get a sample
        sample = dataset[0]

        # Create normalizer from action features only (simpler test)
        action_features = dataset.action_features
        normalizer = Normalizer(features=action_features, norm_mode=NormalizationMode.MEAN_STD)

        # Prepare batch (just action)
        batch = {"action": sample.action}

        # Normalize
        normalized = normalizer.normalize(batch)

        # Verify output
        assert isinstance(normalized, dict)
        assert "action" in normalized
        assert normalized["action"].shape == batch["action"].shape

        # Denormalize and verify roundtrip
        recovered = normalizer.denormalize(normalized)
        torch.testing.assert_close(recovered["action"], batch["action"], rtol=1e-4, atol=1e-6)

    def test_normalizer_state_dict_save_load(self, lerobot_datamodule) -> None:
        """Test saving and loading normalizer state dict."""
        lerobot_datamodule.setup("fit")
        dataset = lerobot_datamodule.train_dataset

        action_features = dataset.action_features
        normalizer1 = Normalizer(features=action_features)

        # Save state dict
        state_dict = normalizer1.state_dict()

        # Create new normalizer and load
        normalizer2 = Normalizer(features=action_features)
        normalizer2.load_state_dict(state_dict)

        # Verify they produce same results
        batch = {"action": torch.randn(4, action_features["action"].shape[0])}

        result1 = normalizer1.normalize(batch)
        result2 = normalizer2.normalize(batch)

        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        torch.testing.assert_close(result1["action"], result2["action"])
