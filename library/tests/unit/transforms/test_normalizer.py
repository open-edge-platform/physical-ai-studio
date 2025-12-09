# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Normalizer class."""

from __future__ import annotations

import pytest
import torch

from getiaction.data import Feature, NormalizationParameters
from getiaction.transforms import NormalizationMode, Normalizer


class TestNormalizerSingleEmbodiment:
    """Tests for single-embodiment normalizer."""

    @pytest.fixture
    def sample_features(self) -> dict[str, Feature]:
        """Create sample features with normalization data."""
        return {
            "state": Feature(
                normalization_data=NormalizationParameters(
                    mean=torch.tensor([1.0, 2.0, 3.0]),
                    std=torch.tensor([0.5, 1.0, 1.5]),
                    min=torch.tensor([0.0, 0.0, 0.0]),
                    max=torch.tensor([2.0, 4.0, 6.0]),
                ),
                name="state",
            ),
            "action": Feature(
                normalization_data=NormalizationParameters(
                    mean=torch.tensor([0.0, 0.0]),
                    std=torch.tensor([1.0, 2.0]),
                    min=torch.tensor([-1.0, -2.0]),
                    max=torch.tensor([1.0, 2.0]),
                ),
                name="action",
            ),
        }

    def test_normalize_mean_std(self, sample_features: dict[str, Feature]) -> None:
        """Test mean/std normalization."""
        normalizer = Normalizer(features=sample_features, norm_mode=NormalizationMode.MEAN_STD)

        batch = {
            "state": torch.tensor([[1.0, 2.0, 3.0]]),  # Equals mean
            "action": torch.tensor([[0.0, 0.0]]),
        }

        normalized = normalizer.normalize(batch)

        # When input equals mean, output should be 0
        assert isinstance(normalized, dict)
        torch.testing.assert_close(normalized["state"], torch.tensor([[0.0, 0.0, 0.0]]))
        torch.testing.assert_close(normalized["action"], torch.tensor([[0.0, 0.0]]))

    def test_denormalize_mean_std(self, sample_features: dict[str, Feature]) -> None:
        """Test mean/std denormalization."""
        normalizer = Normalizer(features=sample_features, norm_mode=NormalizationMode.MEAN_STD)

        normalized = {
            "state": torch.tensor([[0.0, 0.0, 0.0]]),
            "action": torch.tensor([[1.0, 1.0]]),  # 1 std above mean
        }

        denormalized = normalizer.denormalize(normalized)

        # Should recover original values
        torch.testing.assert_close(denormalized["state"], torch.tensor([[1.0, 2.0, 3.0]]))
        torch.testing.assert_close(denormalized["action"], torch.tensor([[1.0, 2.0]]))  # mean + 1*std

    def test_normalize_denormalize_roundtrip(self, sample_features: dict[str, Feature]) -> None:
        """Test that normalize -> denormalize recovers original values."""
        normalizer = Normalizer(features=sample_features, norm_mode=NormalizationMode.MEAN_STD)

        original = {
            "state": torch.tensor([[0.5, 1.5, 2.5], [1.5, 2.5, 3.5]]),
            "action": torch.tensor([[0.3, 0.7], [-0.3, -0.7]]),
        }

        normalized = normalizer.normalize(original)
        assert isinstance(normalized, dict)  # Single-embodiment returns dict
        recovered = normalizer.denormalize(normalized)

        torch.testing.assert_close(recovered["state"], original["state"])
        torch.testing.assert_close(recovered["action"], original["action"])

    def test_normalize_min_max(self, sample_features: dict[str, Feature]) -> None:
        """Test min/max normalization to [-1, 1]."""
        normalizer = Normalizer(features=sample_features, norm_mode=NormalizationMode.MIN_MAX)

        batch = {
            "state": torch.tensor([[1.0, 2.0, 3.0]]),  # Middle of range
            "action": torch.tensor([[0.0, 0.0]]),  # Middle of range
        }

        normalized = normalizer.normalize(batch)

        # Middle of range should map to 0
        assert isinstance(normalized, dict)
        torch.testing.assert_close(normalized["state"], torch.tensor([[0.0, 0.0, 0.0]]))
        torch.testing.assert_close(normalized["action"], torch.tensor([[0.0, 0.0]]))

    def test_normalize_identity(self, sample_features: dict[str, Feature]) -> None:
        """Test identity mode (no normalization)."""
        normalizer = Normalizer(features=sample_features, norm_mode=NormalizationMode.IDENTITY)

        batch = {
            "state": torch.tensor([[1.0, 2.0, 3.0]]),
            "action": torch.tensor([[0.5, 1.0]]),
        }

        normalized = normalizer.normalize(batch)

        # Should be unchanged
        assert isinstance(normalized, dict)
        torch.testing.assert_close(normalized["state"], batch["state"])
        torch.testing.assert_close(normalized["action"], batch["action"])

    def test_passthrough_unknown_keys(self, sample_features: dict[str, Feature]) -> None:
        """Test that unknown keys pass through unchanged."""
        normalizer = Normalizer(features=sample_features, norm_mode=NormalizationMode.MEAN_STD)

        batch = {
            "state": torch.tensor([[1.0, 2.0, 3.0]]),
            "unknown_key": torch.tensor([[100.0, 200.0]]),
        }

        normalized = normalizer.normalize(batch)

        # Unknown key should pass through unchanged
        assert isinstance(normalized, dict)
        torch.testing.assert_close(normalized["unknown_key"], batch["unknown_key"])

    def test_single_embodiment_properties(self, sample_features: dict[str, Feature]) -> None:
        """Test single-embodiment property accessors."""
        normalizer = Normalizer(features=sample_features)

        assert not normalizer.is_multi_embodiment
        assert normalizer.embodiments == ["default"]
        assert set(normalizer.get_feature_names()) == {"state", "action"}

    def test_state_dict_contains_buffers(self, sample_features: dict[str, Feature]) -> None:
        """Test that normalization stats are saved in state_dict."""
        normalizer = Normalizer(features=sample_features)

        state_dict = normalizer.state_dict()

        # Check that buffers are present
        assert "default_state_mean" in state_dict
        assert "default_state_std" in state_dict
        assert "default_action_mean" in state_dict
        assert "default_action_std" in state_dict

    def test_load_state_dict(self, sample_features: dict[str, Feature]) -> None:
        """Test loading state_dict restores normalization stats."""
        normalizer1 = Normalizer(features=sample_features)

        # Modify stats
        normalizer1.default_state_mean.fill_(99.0)

        # Create new normalizer and load state
        normalizer2 = Normalizer(features=sample_features)
        normalizer2.load_state_dict(normalizer1.state_dict())

        torch.testing.assert_close(
            normalizer2.default_state_mean,
            torch.tensor([99.0, 99.0, 99.0]),
        )

    def test_device_handling(self, sample_features: dict[str, Feature]) -> None:
        """Test that normalizer handles device placement correctly."""
        normalizer = Normalizer(features=sample_features)

        # Stats should be on CPU by default
        assert normalizer.default_state_mean.device.type == "cpu"

        # If CUDA available, test device transfer
        if torch.cuda.is_available():
            normalizer = normalizer.cuda()
            assert normalizer.default_state_mean.device.type == "cuda"

            # Normalize should work with CUDA tensors
            batch = {
                "state": torch.tensor([[1.0, 2.0, 3.0]], device="cuda"),
            }
            normalized = normalizer.normalize(batch)
            assert isinstance(normalized, dict)
            assert normalized["state"].device.type == "cuda"


class TestNormalizerMultiEmbodiment:
    """Tests for multi-embodiment normalizer."""

    @pytest.fixture
    def multi_embodiment_features(self) -> dict[str, dict[str, Feature]]:
        """Create features for multiple embodiments."""
        return {
            "franka": {
                "state": Feature(
                    normalization_data=NormalizationParameters(
                        mean=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),  # 7 DOF
                        std=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
                    ),
                    name="state",
                ),
            },
            "ur5": {
                "state": Feature(
                    normalization_data=NormalizationParameters(
                        mean=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),  # 6 DOF
                        std=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                    ),
                    name="state",
                ),
            },
        }

    def test_multi_embodiment_detection(
        self, multi_embodiment_features: dict[str, dict[str, Feature]]
    ) -> None:
        """Test that multi-embodiment mode is correctly detected."""
        normalizer = Normalizer(features=multi_embodiment_features)

        assert normalizer.is_multi_embodiment
        assert set(normalizer.embodiments) == {"franka", "ur5"}

    def test_multi_embodiment_normalize(
        self, multi_embodiment_features: dict[str, dict[str, Feature]]
    ) -> None:
        """Test normalization with embodiment_id."""
        normalizer = Normalizer(features=multi_embodiment_features)

        batch = {
            "state": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]),
        }

        # Normalize for franka embodiment
        normalized = normalizer.normalize(batch, embodiment_id="franka")

        # Without padding configured, should just return normalized dict
        assert isinstance(normalized, dict)
        torch.testing.assert_close(normalized["state"], torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))

    def test_multi_embodiment_with_padding(
        self, multi_embodiment_features: dict[str, dict[str, Feature]]
    ) -> None:
        """Test multi-embodiment with padding to max dimensions."""
        normalizer = Normalizer(
            features=multi_embodiment_features,
            max_state_dim=10,
        )

        batch = {
            "state": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),  # UR5 (6 DOF)
        }

        result = normalizer.normalize(batch, embodiment_id="ur5")

        # Should return tuple with masks
        assert isinstance(result, tuple)
        padded, masks = result

        # Check padding
        assert padded["state"].shape[-1] == 10  # Padded to max
        assert masks["state"].shape[-1] == 10
        assert masks["state"][:6].all()  # First 6 are valid
        assert not masks["state"][6:].any()  # Last 4 are padding

    def test_multi_embodiment_separate_stats(
        self, multi_embodiment_features: dict[str, dict[str, Feature]]
    ) -> None:
        """Test that each embodiment has separate normalization stats."""
        normalizer = Normalizer(features=multi_embodiment_features)

        # Check that separate buffers exist
        assert hasattr(normalizer, "franka_state_mean")
        assert hasattr(normalizer, "ur5_state_mean")

        # Check they have different dimensions
        assert normalizer.franka_state_mean.shape[0] == 7
        assert normalizer.ur5_state_mean.shape[0] == 6


class TestNormalizerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_features(self) -> None:
        """Test handling of empty features dict."""
        normalizer = Normalizer(features={})

        batch = {"some_key": torch.tensor([[1.0, 2.0]])}
        normalized = normalizer.normalize(batch)

        # Should pass through unchanged
        assert isinstance(normalized, dict)
        torch.testing.assert_close(normalized["some_key"], batch["some_key"])

    def test_feature_without_normalization_data(self) -> None:
        """Test handling of features without normalization data."""
        features = {
            "state": Feature(
                normalization_data=None,  # No normalization data
                name="state",
            ),
        }
        normalizer = Normalizer(features=features)

        batch = {"state": torch.tensor([[1.0, 2.0, 3.0]])}
        normalized = normalizer.normalize(batch)

        # Should pass through unchanged
        assert isinstance(normalized, dict)
        torch.testing.assert_close(normalized["state"], batch["state"])

    def test_zero_std_handling(self) -> None:
        """Test that zero std is handled (clamped to avoid division by zero)."""
        features = {
            "state": Feature(
                normalization_data=NormalizationParameters(
                    mean=torch.tensor([1.0, 2.0]),
                    std=torch.tensor([0.0, 1.0]),  # First element has zero std
                ),
                name="state",
            ),
        }
        normalizer = Normalizer(features=features)

        batch = {"state": torch.tensor([[1.0, 2.0]])}
        normalized = normalizer.normalize(batch)

        # Should not raise or produce inf/nan
        assert isinstance(normalized, dict)
        assert not torch.isnan(normalized["state"]).any()
        assert not torch.isinf(normalized["state"]).any()

    def test_forward_method(self) -> None:
        """Test that forward() is an alias for normalize()."""
        features = {
            "state": Feature(
                normalization_data=NormalizationParameters(
                    mean=torch.tensor([1.0]),
                    std=torch.tensor([1.0]),
                ),
                name="state",
            ),
        }
        normalizer = Normalizer(features=features)

        batch = {"state": torch.tensor([[2.0]])}

        # forward() and normalize() should produce same result
        result_forward = normalizer.forward(batch)
        result_normalize = normalizer.normalize(batch)

        assert isinstance(result_forward, dict)
        assert isinstance(result_normalize, dict)
        torch.testing.assert_close(result_forward["state"], result_normalize["state"])
