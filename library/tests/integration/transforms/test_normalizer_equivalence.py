# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Equivalence tests comparing getiaction Normalizer with LeRobot's formulas.

These tests verify that our Normalizer produces identical outputs to LeRobot's
normalization formulas for both MEAN_STD and MIN_MAX normalization modes.

LeRobot formulas (from lerobot/processor/normalize_processor.py):
- MEAN_STD normalize: (tensor - mean) / (std + eps)
- MEAN_STD denormalize: tensor * std + mean
- MIN_MAX normalize: 2 * (tensor - min) / denom - 1, where denom = max - min (with eps when == 0)
- MIN_MAX denormalize: (tensor + 1) / 2 * denom + min
"""

from __future__ import annotations

import pytest
import torch

from getiaction.data import Feature, NormalizationParameters
from getiaction.transforms import NormalizationMode, Normalizer


def lerobot_mean_std_normalize(
    tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    """LeRobot's MEAN_STD normalization formula."""
    denom = std + eps
    return (tensor - mean) / denom


def lerobot_mean_std_denormalize(
    tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
) -> torch.Tensor:
    """LeRobot's MEAN_STD denormalization formula."""
    return tensor * std + mean


def lerobot_min_max_normalize(
    tensor: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    """LeRobot's MIN_MAX normalization formula."""
    denom = max_val - min_val
    denom = torch.where(
        denom == 0, torch.tensor(eps, device=tensor.device, dtype=tensor.dtype), denom,
    )
    return 2 * (tensor - min_val) / denom - 1


def lerobot_min_max_denormalize(
    tensor: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    """LeRobot's MIN_MAX denormalization formula."""
    denom = max_val - min_val
    denom = torch.where(
        denom == 0, torch.tensor(eps, device=tensor.device, dtype=tensor.dtype), denom,
    )
    return (tensor + 1) / 2 * denom + min_val


class TestNormalizerEquivalenceWithLeRobot:
    """Tests verifying numerical equivalence with LeRobot's normalization formulas."""

    @pytest.fixture
    def sample_stats(self) -> dict[str, dict[str, torch.Tensor]]:
        """Create sample statistics for testing."""
        return {
            "state": {
                "mean": torch.tensor([0.5, 1.0, -0.5]),
                "std": torch.tensor([0.1, 0.5, 0.2]),
                "min": torch.tensor([0.0, 0.0, -1.0]),
                "max": torch.tensor([1.0, 2.0, 0.0]),
            },
            "action": {
                "mean": torch.tensor([0.0, 0.0]),
                "std": torch.tensor([1.0, 2.0]),
                "min": torch.tensor([-1.0, -2.0]),
                "max": torch.tensor([1.0, 2.0]),
            },
        }

    @pytest.fixture
    def getiaction_features(
        self, sample_stats: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, Feature]:
        """Create getiaction features from sample stats."""
        features = {}
        for name, stats in sample_stats.items():
            features[name] = Feature(
                shape=(len(stats["mean"]),),
                normalization_data=NormalizationParameters(
                    mean=stats["mean"],
                    std=stats["std"],
                    min=stats["min"],
                    max=stats["max"],
                ),
            )
        return features

    def test_mean_std_normalize_equivalence(
        self,
        getiaction_features: dict[str, Feature],
        sample_stats: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """Verify MEAN_STD normalization produces identical outputs to LeRobot formula."""
        eps = 1e-6
        our_normalizer = Normalizer(
            features=getiaction_features,
            norm_mode=NormalizationMode.MEAN_STD,
            eps=eps,
        )

        # Create test input
        batch = {
            "state": torch.tensor([[0.6, 1.5, -0.3]]),
            "action": torch.tensor([[0.5, 1.0]]),
        }

        # Normalize with our implementation
        our_result = our_normalizer.normalize(batch)

        # Normalize with LeRobot formula directly
        for key in batch:
            stats = sample_stats[key]
            lerobot_result = lerobot_mean_std_normalize(
                batch[key], stats["mean"], stats["std"], eps,
            )

            torch.testing.assert_close(
                our_result[key],
                lerobot_result,
                rtol=1e-5,
                atol=1e-7,
                msg=f"Mismatch in MEAN_STD normalization for key '{key}'",
            )

    def test_mean_std_denormalize_equivalence(
        self,
        getiaction_features: dict[str, Feature],
        sample_stats: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """Verify MEAN_STD denormalization produces identical outputs to LeRobot formula."""
        our_normalizer = Normalizer(
            features=getiaction_features,
            norm_mode=NormalizationMode.MEAN_STD,
        )

        # Create normalized input
        normalized_batch = {
            "state": torch.tensor([[1.0, -0.5, 0.2]]),
            "action": torch.tensor([[-0.5, 0.25]]),
        }

        # Denormalize with our implementation
        our_result = our_normalizer.denormalize(normalized_batch)

        # Denormalize with LeRobot formula directly
        for key in normalized_batch:
            stats = sample_stats[key]
            lerobot_result = lerobot_mean_std_denormalize(
                normalized_batch[key], stats["mean"], stats["std"],
            )

            torch.testing.assert_close(
                our_result[key],
                lerobot_result,
                rtol=1e-5,
                atol=1e-7,
                msg=f"Mismatch in MEAN_STD denormalization for key '{key}'",
            )

    def test_min_max_normalize_equivalence(
        self, sample_stats: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """Verify MIN_MAX normalization produces identical outputs to LeRobot formula."""
        eps = 1e-6

        # Create features for our normalizer
        features = {}
        for name, stats in sample_stats.items():
            features[name] = Feature(
                shape=(len(stats["mean"]),),
                normalization_data=NormalizationParameters(
                    mean=stats["mean"],
                    std=stats["std"],
                    min=stats["min"],
                    max=stats["max"],
                ),
            )

        our_normalizer = Normalizer(features=features, norm_mode=NormalizationMode.MIN_MAX, eps=eps)

        # Create test input
        batch = {
            "state": torch.tensor([[0.5, 1.0, -0.5]]),
            "action": torch.tensor([[0.0, 0.0]]),
        }

        # Normalize with our implementation
        our_result = our_normalizer.normalize(batch)

        # Normalize with LeRobot formula directly
        for key in batch:
            stats = sample_stats[key]
            lerobot_result = lerobot_min_max_normalize(
                batch[key], stats["min"], stats["max"], eps,
            )

            torch.testing.assert_close(
                our_result[key],
                lerobot_result,
                rtol=1e-5,
                atol=1e-7,
                msg=f"Mismatch in MIN_MAX normalization for key '{key}'",
            )

    def test_min_max_denormalize_equivalence(
        self, sample_stats: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """Verify MIN_MAX denormalization produces identical outputs to LeRobot formula."""
        eps = 1e-6

        # Create features for our normalizer
        features = {}
        for name, stats in sample_stats.items():
            features[name] = Feature(
                shape=(len(stats["mean"]),),
                normalization_data=NormalizationParameters(
                    mean=stats["mean"],
                    std=stats["std"],
                    min=stats["min"],
                    max=stats["max"],
                ),
            )

        our_normalizer = Normalizer(features=features, norm_mode=NormalizationMode.MIN_MAX, eps=eps)

        # Create normalized input (values in [-1, 1] range)
        normalized_batch = {
            "state": torch.tensor([[0.0, 0.0, 0.0]]),  # midpoint
            "action": torch.tensor([[-1.0, 1.0]]),  # extremes
        }

        # Denormalize with our implementation
        our_result = our_normalizer.denormalize(normalized_batch)

        # Denormalize with LeRobot formula directly
        for key in normalized_batch:
            stats = sample_stats[key]
            lerobot_result = lerobot_min_max_denormalize(
                normalized_batch[key], stats["min"], stats["max"], eps,
            )

            torch.testing.assert_close(
                our_result[key],
                lerobot_result,
                rtol=1e-5,
                atol=1e-7,
                msg=f"Mismatch in MIN_MAX denormalization for key '{key}'",
            )

    def test_roundtrip_equivalence(
        self, sample_stats: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """Verify normalize->denormalize roundtrip recovers original values."""
        eps = 1e-6

        # Create features for our normalizer
        features = {}
        for name, stats in sample_stats.items():
            features[name] = Feature(
                shape=(len(stats["mean"]),),
                normalization_data=NormalizationParameters(
                    mean=stats["mean"],
                    std=stats["std"],
                    min=stats["min"],
                    max=stats["max"],
                ),
            )

        our_normalizer = Normalizer(
            features=features, norm_mode=NormalizationMode.MEAN_STD, eps=eps,
        )

        # Original input
        original = {
            "state": torch.tensor([[0.6, 1.5, -0.3]]),
            "action": torch.tensor([[0.5, 1.0]]),
        }

        # Our roundtrip
        our_normalized = our_normalizer.normalize(original)
        our_roundtrip = our_normalizer.denormalize(our_normalized)

        # LeRobot roundtrip (using formulas directly)
        for key in original:
            stats = sample_stats[key]
            lerobot_normalized = lerobot_mean_std_normalize(
                original[key], stats["mean"], stats["std"], eps,
            )
            lerobot_roundtrip = lerobot_mean_std_denormalize(
                lerobot_normalized, stats["mean"], stats["std"],
            )

            # Compare roundtrip results
            torch.testing.assert_close(
                our_roundtrip[key],
                lerobot_roundtrip,
                rtol=1e-5,
                atol=1e-7,
                msg=f"Roundtrip mismatch for key '{key}'",
            )

            # Verify we recover original
            torch.testing.assert_close(
                our_roundtrip[key],
                original[key],
                rtol=1e-5,
                atol=1e-7,
                msg=f"Our roundtrip doesn't recover original for key '{key}'",
            )

    def test_zero_std_handling_equivalence(self) -> None:
        """Verify zero std handling matches LeRobot (uses eps in denominator)."""
        eps = 1e-6

        # Create feature with zero std for one dimension
        features = {
            "state": Feature(
                shape=(3,),
                normalization_data=NormalizationParameters(
                    mean=torch.tensor([0.5, 1.0, -0.5]),
                    std=torch.tensor([0.0, 0.5, 0.2]),  # First dim has zero std
                    min=torch.tensor([0.0, 0.0, -1.0]),
                    max=torch.tensor([1.0, 2.0, 0.0]),
                ),
            ),
        }

        our_normalizer = Normalizer(
            features=features, norm_mode=NormalizationMode.MEAN_STD, eps=eps,
        )

        # Test input
        batch = {"state": torch.tensor([[0.5, 1.5, -0.3]])}

        # Our result
        our_result = our_normalizer.normalize(batch)

        # LeRobot formula result
        mean = torch.tensor([0.5, 1.0, -0.5])
        std = torch.tensor([0.0, 0.5, 0.2])
        lerobot_result = lerobot_mean_std_normalize(batch["state"], mean, std, eps)

        torch.testing.assert_close(
            our_result["state"],
            lerobot_result,
            rtol=1e-5,
            atol=1e-7,
            msg="Zero std handling differs from LeRobot",
        )

    def test_zero_range_min_max_handling(self) -> None:
        """Verify handling when min == max (zero range) matches LeRobot."""
        eps = 1e-6

        features = {
            "state": Feature(
                shape=(3,),
                normalization_data=NormalizationParameters(
                    mean=torch.tensor([0.5, 0.5, 0.5]),
                    std=torch.tensor([0.1, 0.1, 0.1]),
                    min=torch.tensor([0.5, 0.0, -1.0]),  # First dim: min == max
                    max=torch.tensor([0.5, 1.0, 0.0]),
                ),
            ),
        }

        our_normalizer = Normalizer(
            features=features, norm_mode=NormalizationMode.MIN_MAX, eps=eps,
        )

        batch = {"state": torch.tensor([[0.5, 0.5, -0.5]])}

        our_result = our_normalizer.normalize(batch)

        # LeRobot formula
        min_val = torch.tensor([0.5, 0.0, -1.0])
        max_val = torch.tensor([0.5, 1.0, 0.0])
        lerobot_result = lerobot_min_max_normalize(batch["state"], min_val, max_val, eps)

        torch.testing.assert_close(
            our_result["state"],
            lerobot_result,
            rtol=1e-5,
            atol=1e-7,
            msg="Zero range handling differs from LeRobot",
        )

    def test_eps_parameter_effect(self) -> None:
        """Verify eps parameter is correctly applied in denominator."""
        features = {
            "state": Feature(
                shape=(2,),
                normalization_data=NormalizationParameters(
                    mean=torch.tensor([0.0, 0.0]),
                    std=torch.tensor([0.0, 1.0]),  # One zero std
                    min=torch.tensor([-1.0, -1.0]),
                    max=torch.tensor([1.0, 1.0]),
                ),
            ),
        }

        # Test with different eps values
        for eps in [1e-6, 1e-8, 1e-4]:
            our_normalizer = Normalizer(
                features=features,
                norm_mode=NormalizationMode.MEAN_STD,
                eps=eps,
            )

            batch = {"state": torch.tensor([[0.5, 0.5]])}
            our_result = our_normalizer.normalize(batch)

            # LeRobot formula
            mean = torch.tensor([0.0, 0.0])
            std = torch.tensor([0.0, 1.0])
            lerobot_result = lerobot_mean_std_normalize(batch["state"], mean, std, eps)

            torch.testing.assert_close(
                our_result["state"],
                lerobot_result,
                rtol=1e-7,
                atol=1e-9,
                msg=f"Results differ with eps={eps}",
            )
