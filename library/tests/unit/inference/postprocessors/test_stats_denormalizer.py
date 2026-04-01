# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from physicalai.inference.postprocessors import Postprocessor, StatsDenormalizer


@pytest.fixture()
def stats_dir(tmp_path: Path) -> Path:
    stats = {
        "action/mean": np.array([1.0, 2.0, 3.0]),
        "action/std": np.array([0.5, 1.0, 1.5]),
        "action/min": np.array([0.0, 0.0, 0.0]),
        "action/max": np.array([2.0, 4.0, 6.0]),
        "action.gripper/mean": np.array([0.5]),
        "action.gripper/std": np.array([0.1]),
        "action.gripper/min": np.array([0.0]),
        "action.gripper/max": np.array([1.0]),
    }
    stats_path = tmp_path / "stats.safetensors"
    save_file(stats, str(stats_path))
    return tmp_path


class TestStatsDenormalizerInit:
    def test_invalid_mode_raises(self, stats_dir: Path) -> None:
        with pytest.raises(ValueError, match="Unknown normalization mode"):
            StatsDenormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="bad")

    def test_valid_modes_accepted(self, stats_dir: Path) -> None:
        for mode in ("mean_std", "min_max", "identity"):
            denormalizer = StatsDenormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode=mode)
            assert denormalizer._mode == mode

    def test_is_postprocessor(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(stats_path=str(stats_dir / "stats.safetensors"))
        assert isinstance(denormalizer, Postprocessor)

    def test_neither_stats_path_nor_artifact_raises(self) -> None:
        with pytest.raises(ValueError, match="Either stats_path, artifact, or stats must be provided"):
            StatsDenormalizer()

    def test_artifact_param_accepted(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(artifact=str(stats_dir / "stats.safetensors"))
        assert denormalizer._stats_path == str(stats_dir / "stats.safetensors")

    def test_stats_path_takes_precedence_over_artifact(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(
            stats_path=str(stats_dir / "stats.safetensors"),
            artifact="/some/other/path.safetensors",
        )
        assert denormalizer._stats_path == str(stats_dir / "stats.safetensors")


class TestStatsDenormalizerMeanStd:
    def test_denormalizes_all_features(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="mean_std")
        outputs = {
            "action": np.array([0.0, 0.0, 0.0]),
            "action.gripper": np.array([0.0]),
        }
        result = denormalizer(outputs)

        # denorm(0) with mean_std: 0 * std + mean = mean
        np.testing.assert_allclose(result["action"], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result["action.gripper"], np.array([0.5]))

    def test_denormalizes_filtered_features(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(
            stats_path=str(stats_dir / "stats.safetensors"),
            mode="mean_std",
            features=["action"],
        )
        outputs = {
            "action": np.array([1.0, 1.0, 1.0]),
            "action.gripper": np.array([2.0]),
        }
        result = denormalizer(outputs)

        # denorm(1) with mean_std: 1 * std + mean
        expected_action = np.array([1.0, 1.0, 1.0]) * np.array([0.5, 1.0, 1.5]) + np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result["action"], expected_action)
        np.testing.assert_array_equal(result["action.gripper"], np.array([2.0]))

    def test_roundtrip_mean_std(self, stats_dir: Path) -> None:
        from physicalai.inference.preprocessors import StatsNormalizer

        original = {"action": np.array([1.5, 3.0, 4.5])}

        normalizer = StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="mean_std")
        denormalizer = StatsDenormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="mean_std")

        normalized = normalizer(dict(original))
        recovered = denormalizer(normalized)

        np.testing.assert_allclose(recovered["action"], original["action"], atol=1e-6)


class TestStatsDenormalizerMinMax:
    def test_denormalizes_all_features(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="min_max")

        # normalized -1 → min, +1 → max
        outputs = {"action": np.array([-1.0, -1.0, -1.0])}
        result = denormalizer(outputs)
        np.testing.assert_allclose(result["action"], np.array([0.0, 0.0, 0.0]))

        outputs_max = {"action": np.array([1.0, 1.0, 1.0])}
        result_max = denormalizer(outputs_max)
        np.testing.assert_allclose(result_max["action"], np.array([2.0, 4.0, 6.0]))

    def test_denormalizes_filtered_features(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(
            stats_path=str(stats_dir / "stats.safetensors"),
            mode="min_max",
            features=["action"],
        )
        outputs = {
            "action": np.array([0.0, 0.0, 0.0]),
            "action.gripper": np.array([0.5]),
        }
        result = denormalizer(outputs)

        # denorm(0) with min_max: (0 + 1) / 2 * (max - min) + min = midpoint
        expected = (np.array([0.0]) + 1.0) / 2.0 * (np.array([2.0, 4.0, 6.0]) - np.array([0.0, 0.0, 0.0])) + np.array([
            0.0,
            0.0,
            0.0,
        ])
        np.testing.assert_allclose(result["action"], expected)
        np.testing.assert_array_equal(result["action.gripper"], np.array([0.5]))

    def test_roundtrip_min_max(self, stats_dir: Path) -> None:
        from physicalai.inference.preprocessors import StatsNormalizer

        original = {"action": np.array([1.0, 2.0, 3.0])}

        normalizer = StatsNormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="min_max")
        denormalizer = StatsDenormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="min_max")

        normalized = normalizer(dict(original))
        recovered = denormalizer(normalized)

        np.testing.assert_allclose(recovered["action"], original["action"], atol=1e-6)


class TestStatsDenormalizerIdentity:
    def test_identity_mode_passthrough(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="identity")
        outputs = {"action": np.array([42.0, 99.0, 7.0])}
        result = denormalizer(outputs)
        np.testing.assert_array_equal(result["action"], np.array([42.0, 99.0, 7.0]))


class TestStatsDenormalizerLazyLoading:
    def test_stats_not_loaded_at_init(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(stats_path=str(stats_dir / "stats.safetensors"))
        assert denormalizer._stats is None

    def test_stats_loaded_on_first_call(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(stats_path=str(stats_dir / "stats.safetensors"))
        denormalizer({"action": np.array([0.0, 0.0, 0.0])})
        assert denormalizer._stats is not None

    def test_load_stats_eager(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(stats_path=str(stats_dir / "stats.safetensors"))
        denormalizer.load_stats()
        assert denormalizer._stats is not None
        assert "action" in denormalizer._stats
        assert "action.gripper" in denormalizer._stats


class TestStatsDenormalizerEdgeCases:
    def test_unknown_keys_pass_through(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="mean_std")
        outputs = {
            "action": np.array([0.0, 0.0, 0.0]),
            "extra_key": np.array([7.0]),
        }
        result = denormalizer(outputs)
        np.testing.assert_array_equal(result["extra_key"], np.array([7.0]))

    def test_missing_feature_in_outputs_skipped(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(
            stats_path=str(stats_dir / "stats.safetensors"),
            mode="mean_std",
            features=["action", "nonexistent"],
        )
        outputs = {"action": np.array([0.0, 0.0, 0.0])}
        result = denormalizer(outputs)
        assert "action" in result
        assert "nonexistent" not in result

    def test_empty_outputs(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(stats_path=str(stats_dir / "stats.safetensors"), mode="mean_std")
        result = denormalizer({})
        assert result == {}


class TestStatsDenormalizerRepr:
    def test_repr_with_features(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(
            stats_path=str(stats_dir / "stats.safetensors"),
            mode="mean_std",
            features=["action"],
        )
        r = repr(denormalizer)
        assert "StatsDenormalizer" in r
        assert "mean_std" in r
        assert "action" in r

    def test_repr_all_features(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(stats_path=str(stats_dir / "stats.safetensors"))
        r = repr(denormalizer)
        assert "all" in r


class TestStatsDenormalizerArtifactParam:
    def test_artifact_denormalizes_mean_std(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(artifact=str(stats_dir / "stats.safetensors"), mode="mean_std")
        outputs = {"action": np.array([0.0, 0.0, 0.0])}
        result = denormalizer(outputs)
        np.testing.assert_allclose(result["action"], np.array([1.0, 2.0, 3.0]))

    def test_artifact_denormalizes_min_max(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(artifact=str(stats_dir / "stats.safetensors"), mode="min_max")
        outputs = {"action": np.array([-1.0, -1.0, -1.0])}
        result = denormalizer(outputs)
        np.testing.assert_allclose(result["action"], np.array([0.0, 0.0, 0.0]))

    def test_artifact_lazy_load(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(artifact=str(stats_dir / "stats.safetensors"))
        assert denormalizer._stats is None
        denormalizer({"action": np.array([0.0, 0.0, 0.0])})
        assert denormalizer._stats is not None

    def test_artifact_repr(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(artifact=str(stats_dir / "stats.safetensors"))
        r = repr(denormalizer)
        assert "StatsDenormalizer" in r
        assert "stats.safetensors" in r

    def test_roundtrip_via_artifact(self, stats_dir: Path) -> None:
        from physicalai.inference.preprocessors import StatsNormalizer

        original = {"action": np.array([1.5, 3.0, 4.5])}
        normalizer = StatsNormalizer(artifact=str(stats_dir / "stats.safetensors"), mode="mean_std")
        denormalizer = StatsDenormalizer(artifact=str(stats_dir / "stats.safetensors"), mode="mean_std")

        normalized = normalizer(dict(original))
        recovered = denormalizer(normalized)
        np.testing.assert_allclose(recovered["action"], original["action"], atol=1e-6)


class TestParseFlatStats:
    def test_skips_keys_without_slash(self, tmp_path: Path) -> None:
        stats = {
            "action/mean": np.array([1.0]),
            "orphan_key": np.array([99.0]),
        }
        save_file(stats, str(tmp_path / "stats.safetensors"))
        denormalizer = StatsDenormalizer(stats_path=str(tmp_path / "stats.safetensors"))
        denormalizer.load_stats()
        assert denormalizer._stats is not None
        assert "orphan_key" not in denormalizer._stats

    def test_groups_features_correctly(self, stats_dir: Path) -> None:
        denormalizer = StatsDenormalizer(stats_path=str(stats_dir / "stats.safetensors"))
        denormalizer.load_stats()
        assert denormalizer._stats is not None
        assert set(denormalizer._stats.keys()) == {"action", "action.gripper"}
        assert set(denormalizer._stats["action"].keys()) == {"mean", "std", "min", "max"}
