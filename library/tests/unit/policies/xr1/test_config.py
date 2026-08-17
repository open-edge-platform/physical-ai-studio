# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for XR1Config."""

from __future__ import annotations

import pytest
from physicalai.policies.xr1 import ALOHA_STATE_TO_XR1, XR1Config


class TestDefaults:
    """Defaults must match the reference implementation, except where documented."""

    def test_reference_defaults(self) -> None:
        """Hyperparameters carried over from the reference implementation."""
        config = XR1Config()

        assert config.vlm_model_id == "Qwen/Qwen3-VL-4B-Instruct"
        assert config.chunk_size == 30
        assert config.n_action_steps == 30
        assert config.state_len == 1
        assert config.dit_num_layers == 36
        assert config.dit_hidden_size == 1024
        assert config.dit_head_dim == 128
        assert config.dit_kv_heads == 8
        assert config.num_inference_steps == 5
        assert config.training_repeat == 4
        assert config.prefix_mask_prob == pytest.approx(0.5)
        assert config.flow_sampling == "beta"
        assert config.beta_alpha == pytest.approx(1.5)
        assert config.beta_beta == pytest.approx(1.0)
        assert config.freq_coefficient == pytest.approx(1.0)
        assert config.freq_excluded_dims == (17, 18, 19)
        assert config.n_choices == 5

    def test_documented_divergences(self) -> None:
        """The one default that intentionally differs from the reference.

        ``flash_attention_2`` is not a library dependency and is not exportable, so
        the backbone defaults to ``sdpa`` instead.
        """
        config = XR1Config()

        assert config.vlm_attn_implementation == "sdpa"

    def test_reference_training_recipe_is_the_default(self) -> None:
        """Both training-only branches of the reference recipe are on by default."""
        config = XR1Config()

        assert config.async_train is True
        assert config.enable_choice_head is True

    def test_is_frozen(self) -> None:
        """Config is immutable, like every other first-party policy config."""
        config = XR1Config()

        with pytest.raises(AttributeError):
            config.chunk_size = 10  # type: ignore[misc]


class TestOverrides:
    """The tiny preset used by the unit tests must be expressible."""

    def test_tiny_preset(self) -> None:
        """A small configuration that fits in unit-test memory."""
        config = XR1Config(
            dit_num_layers=4,
            dit_hidden_size=256,
            dit_head_dim=128,
            dit_kv_heads=2,
            chunk_size=4,
            n_action_steps=4,
            max_state_dim=8,
            max_action_dim=8,
            image_resolution=(64, 64),
            camera_views=("base",),
        )

        assert config.dit_num_layers == 4
        assert config.dit_hidden_size // config.dit_head_dim == 2
        assert config.camera_views == ("base",)


class TestValidation:
    """__post_init__ must reject configurations that would fail later."""

    def test_action_steps_exceeding_chunk(self) -> None:
        """Executing more steps than are predicted is impossible."""
        with pytest.raises(ValueError, match="cannot be greater than chunk_size"):
            XR1Config(chunk_size=10, n_action_steps=20)

    def test_hidden_size_not_divisible_by_head_dim(self) -> None:
        """Head dim must tile the hidden size."""
        with pytest.raises(ValueError, match="must be divisible by"):
            XR1Config(dit_hidden_size=300, dit_head_dim=128)

    def test_fewer_heads_than_kv_heads(self) -> None:
        """Grouped-query attention needs at least as many heads as kv heads."""
        with pytest.raises(ValueError, match="must be >= dit_kv_heads"):
            XR1Config(dit_hidden_size=128, dit_head_dim=128, dit_kv_heads=8)

    def test_heads_not_divisible_by_kv_heads(self) -> None:
        """Each kv head must serve a whole number of query heads."""
        with pytest.raises(ValueError, match="must be divisible by dit_kv_heads"):
            XR1Config(dit_hidden_size=384, dit_head_dim=128, dit_kv_heads=2)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"chunk_size": 0, "n_action_steps": 0}, "chunk_size"),
            ({"state_len": 0}, "state_len"),
            ({"dit_num_layers": 0}, "dit_num_layers"),
            ({"num_inference_steps": 0}, "num_inference_steps"),
            ({"training_repeat": 0}, "training_repeat"),
        ],
    )
    def test_positive_integers(self, kwargs: dict[str, int], match: str) -> None:
        """Structural sizes must be positive."""
        with pytest.raises(ValueError, match=match):
            XR1Config(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("prob", [-0.1, 1.1])
    def test_prefix_mask_prob_range(self, prob: float) -> None:
        """A masking probability outside [0, 1] is a configuration error."""
        with pytest.raises(ValueError, match="must be in"):
            XR1Config(prefix_mask_prob=prob)

    def test_choice_head_needs_multiple_choices(self) -> None:
        """A choice head with a single candidate cannot be trained."""
        with pytest.raises(ValueError, match="n_choices"):
            XR1Config(enable_choice_head=True, n_choices=1)

    def test_negative_freq_excluded_dims(self) -> None:
        """Excluded frequency dimensions index the action vector."""
        with pytest.raises(ValueError, match="freq_excluded_dims"):
            XR1Config(freq_excluded_dims=(-1,))

    def test_empty_camera_views(self) -> None:
        """At least one camera view must be present in the prompt."""
        with pytest.raises(ValueError, match="at least one view"):
            XR1Config(camera_views=())

    def test_duplicate_camera_views(self) -> None:
        """Duplicated views would duplicate images in the prompt."""
        with pytest.raises(ValueError, match="must be unique"):
            XR1Config(camera_views=("base", "base"))

    @pytest.mark.parametrize("resolution", [(0, 64), (64,), (64, 64, 64)])
    def test_invalid_image_resolution(self, resolution: tuple[int, ...]) -> None:
        """Resolution is exactly two positive integers."""
        with pytest.raises(ValueError, match="image_resolution"):
            XR1Config(image_resolution=resolution)  # type: ignore[arg-type]

    def test_features_must_be_paired(self) -> None:
        """A partial feature schema cannot be resolved."""
        with pytest.raises(ValueError, match="must be provided together"):
            XR1Config(input_features=[])


class TestSlotMaps:
    """Routing dataset dimensions onto XR1's fixed slot layout."""

    def test_absent_by_default(self) -> None:
        """Training from scratch needs no slot layout, so nothing is imposed."""
        config = XR1Config()

        assert config.state_slot_map is None
        assert config.action_slot_map is None

    def test_accepts_the_aloha_state_map(self) -> None:
        """The shipped ALOHA map must fit the 60-slot state vector."""
        config = XR1Config(max_state_dim=60, state_slot_map=ALOHA_STATE_TO_XR1)

        assert config.state_slot_map == ALOHA_STATE_TO_XR1

    def test_rejects_a_slot_past_the_state_width(self) -> None:
        """A map is only meaningful relative to the configured width."""
        with pytest.raises(ValueError, match="state_slot_map"):
            XR1Config(max_state_dim=14, state_slot_map=ALOHA_STATE_TO_XR1)

    def test_rejects_a_duplicated_action_slot(self) -> None:
        """Two action dimensions in one slot would lose one of them."""
        with pytest.raises(ValueError, match="action_slot_map"):
            XR1Config(action_slot_map=(0, 0))
