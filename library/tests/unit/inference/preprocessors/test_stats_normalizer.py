# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from physicalai.inference.preprocessors import Preprocessor, StatsNormalizer

_EPS = 1e-8


@pytest.fixture()
def stats_dir(tmp_path: Path) -> Path:
    stats = {
        "observation.state/mean": np.array([1.0, 2.0]),
        "observation.state/std": np.array([0.5, 1.0]),
        "observation.state/min": np.array([0.0, 0.0]),
        "observation.state/max": np.array([2.0, 4.0]),
        "observation.state/q01": np.array([0.1, 0.2]),
        "observation.state/q99": np.array([1.9, 3.8]),
        "observation.image/mean": np.array([0.5]),
        "observation.image/std": np.array([0.25]),
        "observation.image/min": np.array([0.0]),
        "observation.image/max": np.array([1.0]),
        "observation.image/q01": np.array([0.05]),
        "observation.image/q99": np.array([0.95]),
    }
    stats_path = tmp_path / "stats.safetensors"
    save_file(stats, str(stats_path))
    return tmp_path


class TestStatsNormalizerInit:
    def test_invalid_mode_raises(self, stats_dir: Path) -> None:
        with pytest.raises(ValueError, match="Unknown normalization mode"):
            StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="bad")

    def test_valid_modes_accepted(self, stats_dir: Path) -> None:
        for mode in ("mean_std", "min_max", "quantiles", "identity"):
            normalizer = StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode=mode)
            assert normalizer._mode == mode

    def test_is_preprocessor(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"))
        assert isinstance(normalizer, Preprocessor)

    def test_neither_stats_path_nor_artifact_raises(self) -> None:
        with pytest.raises(ValueError, match="Either stats_path or artifact must be provided"):
            StatsNormalizer()

    def test_artifact_param_accepted(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(artifact=str(stats_dir / "stats.safetensors"))
        assert normalizer._stats_path == str(stats_dir / "stats.safetensors")

    def test_stats_path_takes_precedence_over_artifact(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(
            stats_path=str(stats_dir / "stats.safetensors"),
            artifact="/some/other/path.safetensors",
        )
        assert normalizer._stats_path == str(stats_dir / "stats.safetensors")


class TestStatsNormalizerMeanStd:
    def test_normalizes_all_features(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="mean_std")
        inputs = {
            "observation.state": np.array([1.0, 2.0]),
            "observation.image": np.array([0.5]),
        }
        result = normalizer(inputs)

        expected_state = (np.array([1.0, 2.0]) - np.array([1.0, 2.0])) / (np.array([0.5, 1.0]) + _EPS)
        expected_image = (np.array([0.5]) - np.array([0.5])) / (np.array([0.25]) + _EPS)
        np.testing.assert_allclose(result["observation.state"], expected_state)
        np.testing.assert_allclose(result["observation.image"], expected_image)

    def test_normalizes_filtered_features(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(
            stats_path=str(stats_dir / "stats.safetensors"),
            mode="mean_std",
            features=["observation.state"],
        )
        inputs = {
            "observation.state": np.array([2.0, 4.0]),
            "observation.image": np.array([0.75]),
        }
        result = normalizer(inputs)

        expected_state = (np.array([2.0, 4.0]) - np.array([1.0, 2.0])) / (np.array([0.5, 1.0]) + _EPS)
        np.testing.assert_allclose(result["observation.state"], expected_state)
        np.testing.assert_array_equal(result["observation.image"], np.array([0.75]))


class TestStatsNormalizerMinMax:
    def test_normalizes_all_features(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="min_max")
        inputs = {
            "observation.state": np.array([1.0, 2.0]),
        }
        result = normalizer(inputs)

        min_val = np.array([0.0, 0.0])
        max_val = np.array([2.0, 4.0])
        expected = 2.0 * (np.array([1.0, 2.0]) - min_val) / (max_val - min_val + _EPS) - 1.0
        np.testing.assert_allclose(result["observation.state"], expected)

    def test_normalizes_filtered_features(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(
            stats_path=str(stats_dir / "stats.safetensors"),
            mode="min_max",
            features=["observation.state"],
        )
        inputs = {
            "observation.state": np.array([0.0, 4.0]),
            "observation.image": np.array([0.5]),
        }
        result = normalizer(inputs)

        min_val = np.array([0.0, 0.0])
        max_val = np.array([2.0, 4.0])
        expected = 2.0 * (np.array([0.0, 4.0]) - min_val) / (max_val - min_val + _EPS) - 1.0
        np.testing.assert_allclose(result["observation.state"], expected)
        np.testing.assert_array_equal(result["observation.image"], np.array([0.5]))


class TestStatsNormalizerQuantiles:
    def test_normalizes_all_features(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="quantiles")
        inputs = {"observation.state": np.array([1.0, 2.0])}
        result = normalizer(inputs)

        q01 = np.array([0.1, 0.2])
        q99 = np.array([1.9, 3.8])
        expected = 2.0 * (np.array([1.0, 2.0]) - q01) / (q99 - q01) - 1.0
        np.testing.assert_allclose(result["observation.state"], expected)

    def test_normalizes_filtered_features(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(
            stats_path=str(stats_dir / "stats.safetensors"),
            mode="quantiles",
            features=["observation.state"],
        )
        inputs = {
            "observation.state": np.array([0.5, 1.0]),
            "observation.image": np.array([0.75]),
        }
        result = normalizer(inputs)

        q01 = np.array([0.1, 0.2])
        q99 = np.array([1.9, 3.8])
        expected = 2.0 * (np.array([0.5, 1.0]) - q01) / (q99 - q01) - 1.0
        np.testing.assert_allclose(result["observation.state"], expected)
        np.testing.assert_array_equal(result["observation.image"], np.array([0.75]))

    def test_boundary_values(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="quantiles")

        # q01 should map to -1
        result_min = normalizer({"observation.state": np.array([0.1, 0.2])})
        np.testing.assert_allclose(result_min["observation.state"], np.array([-1.0, -1.0]))

        # q99 should map to +1
        result_max = normalizer({"observation.state": np.array([1.9, 3.8])})
        np.testing.assert_allclose(result_max["observation.state"], np.array([1.0, 1.0]))


class TestStatsNormalizerIdentity:
    def test_identity_mode_passthrough(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="identity")
        inputs = {"observation.state": np.array([42.0, 99.0])}
        result = normalizer(inputs)
        np.testing.assert_array_equal(result["observation.state"], np.array([42.0, 99.0]))


class TestStatsNormalizerLazyLoading:
    def test_stats_not_loaded_at_init(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"))
        assert normalizer._stats is None

    def test_stats_loaded_on_first_call(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"))
        normalizer({"observation.state": np.array([1.0, 2.0])})
        assert normalizer._stats is not None

    def test_load_stats_eager(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"))
        normalizer.load_stats()
        assert normalizer._stats is not None
        assert "observation.state" in normalizer._stats
        assert "observation.image" in normalizer._stats


class TestStatsNormalizerEdgeCases:
    def test_unknown_keys_pass_through(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="mean_std")
        inputs = {
            "observation.state": np.array([1.0, 2.0]),
            "extra_key": np.array([7.0]),
        }
        result = normalizer(inputs)
        np.testing.assert_array_equal(result["extra_key"], np.array([7.0]))

    def test_missing_feature_in_inputs_skipped(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(
            stats_path=str(stats_dir / "stats.safetensors"),
            mode="mean_std",
            features=["observation.state", "nonexistent"],
        )
        inputs = {"observation.state": np.array([1.0, 2.0])}
        result = normalizer(inputs)
        assert "observation.state" in result
        assert "nonexistent" not in result

    def test_empty_inputs(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="mean_std")
        result = normalizer({})
        assert result == {}


class TestStatsNormalizerRepr:
    def test_repr_with_features(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(
            stats_path=str(stats_dir / "stats.safetensors"),
            mode="mean_std",
            features=["observation.state"],
        )
        r = repr(normalizer)
        assert "StatsNormalizer" in r
        assert "mean_std" in r
        assert "observation.state" in r

    def test_repr_all_features(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"))
        r = repr(normalizer)
        assert "all" in r


class TestStatsNormalizerArtifactParam:
    def test_artifact_normalizes_mean_std(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(artifact=str(stats_dir / "stats.safetensors"), mode="mean_std")
        inputs = {"observation.state": np.array([1.0, 2.0])}
        result = normalizer(inputs)

        expected = (np.array([1.0, 2.0]) - np.array([1.0, 2.0])) / (np.array([0.5, 1.0]) + _EPS)
        np.testing.assert_allclose(result["observation.state"], expected)

    def test_artifact_normalizes_min_max(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(artifact=str(stats_dir / "stats.safetensors"), mode="min_max")
        inputs = {"observation.state": np.array([1.0, 2.0])}
        result = normalizer(inputs)

        min_val = np.array([0.0, 0.0])
        max_val = np.array([2.0, 4.0])
        expected = 2.0 * (np.array([1.0, 2.0]) - min_val) / (max_val - min_val + _EPS) - 1.0
        np.testing.assert_allclose(result["observation.state"], expected)

    def test_artifact_lazy_load(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(artifact=str(stats_dir / "stats.safetensors"))
        assert normalizer._stats is None
        normalizer({"observation.state": np.array([1.0, 2.0])})
        assert normalizer._stats is not None

    def test_artifact_repr(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(artifact=str(stats_dir / "stats.safetensors"))
        r = repr(normalizer)
        assert "StatsNormalizer" in r
        assert "stats.safetensors" in r


class TestParseFlatStats:
    def test_skips_keys_without_slash(self, tmp_path: Path) -> None:
        stats = {
            "observation.state/mean": np.array([1.0]),
            "orphan_key": np.array([99.0]),
        }
        save_file(stats, str(tmp_path / "stats.safetensors"))
        normalizer = StatsNormalizer(stats_path=str(tmp_path / "stats.safetensors"))
        normalizer.load_stats()
        assert normalizer._stats is not None
        assert "orphan_key" not in normalizer._stats

    def test_groups_features_correctly(self, stats_dir: Path) -> None:
        normalizer = StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"))
        normalizer.load_stats()
        assert normalizer._stats is not None
        assert set(normalizer._stats.keys()) == {"observation.state", "observation.image"}
        assert set(normalizer._stats["observation.state"].keys()) == {"mean", "std", "min", "max", "q01", "q99"}
