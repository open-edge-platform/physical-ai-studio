# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the vla-eval PhysicalAIHarness model server bridge.

Tests the ``_build_policy_observation`` helper which converts vla-eval
observations (HWC uint8 images) into the layout expected by Physical AI
Studio policies.

Image layout contract:
  - ``Policy`` subclasses: ``(B, C, H, W)`` float32 in ``[0, 1]``
  - ``InferenceModel``: ``(B, H, W, C)`` uint8 (exported preprocessors
    handle layout detection and normalisation internally).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# physicalai_harness.py lives outside the physicalai package (no __init__.py
# in the model_servers dir), so load it via importlib from its file path.
_HARNESS_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "vla-evaluation-harness"
    / "model_servers"
    / "physicalai_harness.py"
)

_spec = importlib.util.spec_from_file_location("physicalai_harness", _HARNESS_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PhysicalAIHarness = _mod.PhysicalAIHarness


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_harness(image_keys: dict[str, str] | None = None, state_key: str | None = "state") -> Any:
    """Build a PhysicalAIHarness with a mock policy (no real weights)."""
    mock_policy = MagicMock()
    mock_policy.image_keys = ["image", "image2"]
    mock_policy.chunk_size = 10
    return PhysicalAIHarness(
        _policy=mock_policy,
        image_keys=image_keys or {"agentview": "image", "wrist": "image2"},
        state_key=state_key,
        chunk_size=10,
    )


def _make_vla_eval_obs(
    *,
    h: int = 256,
    w: int = 256,
    c: int = 3,
    state_dim: int = 8,
    task: str = "pick up the bowl",
    include_wrist: bool = True,
) -> dict[str, Any]:
    """Build a minimal vla-eval observation dict."""
    images: dict[str, np.ndarray] = {"agentview": np.zeros((h, w, c), dtype=np.uint8)}
    if include_wrist:
        images["wrist"] = np.zeros((h, w, c), dtype=np.uint8)
    return {
        "images": images,
        "states": np.zeros(state_dim, dtype=np.float32),
        "task_description": task,
    }


# --------------------------------------------------------------------------- #
# Policy path — channels_first=True → (B, C, H, W) float32 [0, 1]
# --------------------------------------------------------------------------- #


class TestPolicyPathChannelsFirst:
    """Images converted to (B, C, H, W) float32 [0, 1] for Policy subclasses."""

    def test_single_camera_chw_layout(self) -> None:
        harness = _make_harness(image_keys={"agentview": "image"}, state_key=None)
        obs = _make_vla_eval_obs(include_wrist=False)
        result = harness._build_policy_observation(obs, channels_first=True)

        assert result.images is not None
        assert set(result.images) == {"image"}
        img = result.images["image"]
        assert img.shape == (1, 3, 256, 256)
        assert img.dtype == np.float32
        # [0, 1] range
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_two_cameras_chw_layout(self) -> None:
        harness = _make_harness()
        obs = _make_vla_eval_obs()
        result = harness._build_policy_observation(obs, channels_first=True)

        assert result.images is not None
        assert set(result.images) == {"image", "image2"}
        for key in ("image", "image2"):
            assert result.images[key].shape == (1, 3, 256, 256)
            assert result.images[key].dtype == np.float32

    def test_uint8_normalised_to_float32_0_1(self) -> None:
        harness = _make_harness(image_keys={"agentview": "image"}, state_key=None)
        obs = _make_vla_eval_obs(include_wrist=False)
        obs["images"]["agentview"] = np.full((256, 256, 3), 255, dtype=np.uint8)
        result = harness._build_policy_observation(obs, channels_first=True)

        img = result.images["image"]
        assert img.dtype == np.float32
        np.testing.assert_allclose(img, np.ones_like(img), atol=1e-6)


# --------------------------------------------------------------------------- #
# InferenceModel path — channels_first=False → (B, H, W, C) uint8
# --------------------------------------------------------------------------- #


class TestInferenceModelPathChannelsLast:
    """Images kept as (B, H, W, C) uint8 for InferenceModel."""

    def test_single_camera_hwc_layout(self) -> None:
        harness = _make_harness(image_keys={"agentview": "image"}, state_key=None)
        obs = _make_vla_eval_obs(include_wrist=False)
        result = harness._build_policy_observation(obs, channels_first=False)

        assert result.images is not None
        assert set(result.images) == {"image"}
        img = result.images["image"]
        assert img.shape == (1, 256, 256, 3)
        assert img.dtype == np.uint8

    def test_two_cameras_hwc_layout(self) -> None:
        harness = _make_harness()
        obs = _make_vla_eval_obs()
        result = harness._build_policy_observation(obs, channels_first=False)

        assert result.images is not None
        assert set(result.images) == {"image", "image2"}
        for key in ("image", "image2"):
            assert result.images[key].shape == (1, 256, 256, 3)
            assert result.images[key].dtype == np.uint8

    def test_uint8_preserved(self) -> None:
        harness = _make_harness(image_keys={"agentview": "image"}, state_key=None)
        obs = _make_vla_eval_obs(include_wrist=False)
        obs["images"]["agentview"] = np.full((256, 256, 3), 128, dtype=np.uint8)
        result = harness._build_policy_observation(obs, channels_first=False)

        img = result.images["image"]
        assert img.dtype == np.uint8
        np.testing.assert_array_equal(img, np.full((1, 256, 256, 3), 128, dtype=np.uint8))


# --------------------------------------------------------------------------- #
# State and task
# --------------------------------------------------------------------------- #


class TestStateAndTask:
    """State and task fields are batched and passed through."""

    def test_state_batched(self) -> None:
        harness = _make_harness(state_key="observation.state")
        obs = _make_vla_eval_obs()
        result = harness._build_policy_observation(obs, channels_first=True)

        assert result.state is not None
        assert result.state.shape == (1, 8)
        assert result.state.dtype == np.float32

    def test_state_none_when_key_disabled(self) -> None:
        harness = _make_harness(state_key=None)
        obs = _make_vla_eval_obs()
        result = harness._build_policy_observation(obs, channels_first=True)
        assert result.state is None

    def test_task_passed_through(self) -> None:
        harness = _make_harness(state_key=None)
        obs = _make_vla_eval_obs()
        result = harness._build_policy_observation(obs, channels_first=True)
        assert result.task == "pick up the bowl"

    def test_task_none_when_missing(self) -> None:
        harness = _make_harness(state_key=None)
        obs = _make_vla_eval_obs()
        del obs["task_description"]
        result = harness._build_policy_observation(obs, channels_first=True)
        assert result.task is None
