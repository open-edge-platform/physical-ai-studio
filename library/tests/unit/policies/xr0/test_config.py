# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the XR0 configuration dataclass."""

from __future__ import annotations

import pytest

from physicalai.data import Feature, FeatureType
from physicalai.policies.xr0.config import XR0Config


def test_defaults() -> None:
    """Default config constructs with the expected core values."""
    cfg = XR0Config()
    assert cfg.chunk_size == 30
    assert cfg.n_action_steps == 30
    assert cfg.max_action_dim == 32
    assert cfg.dit_num_layers == 16
    assert cfg.camera_views == ("base", "wrist_left")


def test_dict_roundtrip() -> None:
    """Config survives a to_dict / from_dict round trip."""
    cfg = XR0Config(chunk_size=16, n_action_steps=16, camera_views=("base",))
    restored = XR0Config.from_dict(cfg.to_dict())
    assert restored == cfg


def test_dict_roundtrip_with_features() -> None:
    """input_features / output_features survive a to_dict / from_dict round trip."""
    cfg = XR0Config(
        input_features=[Feature(name="state", ftype=FeatureType.STATE, shape=(8,))],
        output_features=[Feature(name="action", ftype=FeatureType.ACTION, shape=(6,))],
    )
    restored = XR0Config.from_dict(cfg.to_dict())
    assert restored == cfg
    assert restored.input_features is not None
    assert restored.input_features[0].ftype is FeatureType.STATE
    assert restored.output_features is not None
    assert restored.output_features[0].name == "action"


def test_n_action_steps_bound() -> None:
    """n_action_steps may not exceed chunk_size."""
    with pytest.raises(ValueError, match="n_action_steps"):
        XR0Config(chunk_size=10, n_action_steps=20)


def test_dit_head_divisibility() -> None:
    """dit_hidden_size must be divisible by dit_head_dim."""
    with pytest.raises(ValueError, match="divisible"):
        XR0Config(dit_hidden_size=100, dit_head_dim=32)


def test_dit_kv_heads_bound() -> None:
    """DiT num_heads must be >= dit_kv_heads."""
    with pytest.raises(ValueError, match="dit_kv_heads"):
        XR0Config(dit_hidden_size=64, dit_head_dim=32, dit_kv_heads=8)


def test_requires_camera_views() -> None:
    """At least one camera view is required."""
    with pytest.raises(ValueError, match="camera_views"):
        XR0Config(camera_views=())
