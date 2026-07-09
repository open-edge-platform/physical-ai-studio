# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading regression tests for first-party policies.

Simple tests that validate export/import round-trips without downloading weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from physicalai.inference import InferenceModel
from physicalai.policies.pi05 import Pi05, Pi05Config
from physicalai.policies.smolvla import SmolVLA, SmolVLAConfig


def _minimal_export_stats() -> dict[str, dict[str, Any]]:
    """Return minimal dataset statistics required by export preprocessors."""
    return {
        "observation.state": {
            "name": "observation.state",
            "shape": (8,),
            "mean": [0.0] * 8,
            "std": [1.0] * 8,
            "q01": [-1.0] * 8,
            "q99": [1.0] * 8,
            "type": "STATE",
        },
        "observation.image": {
            "name": "observation.image",
            "shape": (3, 224, 224),
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "q01": [-1.0, -1.0, -1.0],
            "q99": [1.0, 1.0, 1.0],
            "type": "VISUAL",
        },
        "action": {
            "name": "action",
            "shape": (7,),
            "mean": [0.0] * 7,
            "std": [1.0] * 7,
            "q01": [-1.0] * 7,
            "q99": [1.0] * 7,
            "type": "ACTION",
        },
    }


@pytest.mark.parametrize(
    ("policy_cls", "config_cls", "mock_config"),
    [
        (
            Pi05,
            Pi05Config,
            Pi05Config(
                normalization_mode="MEAN_STD",
                empty_cameras=1,
                n_action_steps=1,
            ),
        ),
        (
            SmolVLA,
            SmolVLAConfig,
            SmolVLAConfig(
                n_action_steps=1,
                num_vlm_layers=0,
                load_vlm_weights=True,
                expert_width_multiplier=0.5,
                prefix_length=0,
                vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Instruct",
            ),
        ),
    ],
)
@pytest.mark.parametrize("backend", ["torch"])
def test_policy_export_successful(
    tmp_path: Path,
    policy_cls: type,
    config_cls: type,
    mock_config: Any,
    backend: str,
) -> None:
    """Policy should export successfully without downloading weights."""

    def _fake_from_hf(self: Any, *args: object, **kwargs: object) -> tuple[Any, None, None]:
        del self, args, kwargs
        return mock_config, None, None

    assert isinstance(mock_config, config_cls)

    with patch.object(policy_cls, "_from_hf", _fake_from_hf):
        # Create and export policy
        policy = policy_cls(pretrained_name_or_path="stub-repo")
        policy._dataset_stats = _minimal_export_stats()  # noqa: SLF001
        policy.eval()

        export_dir = tmp_path / f"{policy_cls.__name__.lower()}_{backend}"
        policy.export(export_dir, backend=backend)

        # Verify export files were created
        export_file = export_dir / f"{policy_cls.__name__.lower()}.pt"
        assert export_file.exists(), f"Export file not found: {export_file}"
        assert export_file.stat().st_size > 0, f"Export file is empty: {export_file}"


@pytest.mark.parametrize(
    ("policy_cls", "config_cls", "mock_config"),
    [
        (
            Pi05,
            Pi05Config,
            Pi05Config(
                normalization_mode="MEAN_STD",
                empty_cameras=1,
                n_action_steps=1,
            ),
        ),
        (
            SmolVLA,
            SmolVLAConfig,
            SmolVLAConfig(
                n_action_steps=1,
                num_vlm_layers=0,
                load_vlm_weights=True,
                expert_width_multiplier=0.5,
                prefix_length=0,
                vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Instruct",
            ),
        ),
    ],
)
@pytest.mark.parametrize("backend", ["torch"])
def test_policy_export_import_roundtrip(
    tmp_path: Path,
    policy_cls: type,
    config_cls: type,
    mock_config: Any,
    backend: str,
) -> None:
    """Policy should export and re-import successfully."""

    def _fake_from_hf(self: Any, *args: object, **kwargs: object) -> tuple[Any, None, None]:
        del self, args, kwargs
        return mock_config, None, None

    assert isinstance(mock_config, config_cls)

    with patch.object(policy_cls, "_from_hf", _fake_from_hf):
        # Create and export policy
        policy = policy_cls(pretrained_name_or_path="stub-repo")
        policy._dataset_stats = _minimal_export_stats()  # noqa: SLF001
        policy.eval()

        export_dir = tmp_path / f"{policy_cls.__name__.lower()}_{backend}"
        policy.export(export_dir, backend=backend)

        # Mock load_from_checkpoint to also set dataset_stats on the reloaded policy
        original_load_from_checkpoint = policy_cls.load_from_checkpoint  # type: ignore[attr-defined]

        def _load_with_stats(checkpoint_path: Path, **kwargs: Any) -> Any:
            loaded_policy = original_load_from_checkpoint(checkpoint_path, **kwargs)
            loaded_policy._dataset_stats = _minimal_export_stats()  # noqa: SLF001
            return loaded_policy

        with patch.object(policy_cls, "load_from_checkpoint", _load_with_stats):  # type: ignore[misc]
            # Re-import and verify
            loaded = InferenceModel(export_dir)
            assert loaded.backend == backend
            assert loaded.policy_name == policy_cls.__name__.lower()
