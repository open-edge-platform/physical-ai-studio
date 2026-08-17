# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the XR1 vision-language-action model."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest
import torch
from physicalai.policies.xr1 import XR1Config, XR1Model

if TYPE_CHECKING:
    from physicalai.policies.xr1.vlm import XR1Qwen3VL

BATCH = 2


@pytest.fixture
def model(tiny_config: XR1Config, tiny_vlm: XR1Qwen3VL) -> XR1Model:
    """Build the tiny model against the shared random backbone.

    Args:
        tiny_config: Small configuration.
        tiny_vlm: Small randomly initialized backbone.

    Returns:
        The model.
    """
    return XR1Model(tiny_config, vlm=tiny_vlm)


class TestBackboneCompatibility:
    """Geometry checks that the reference only catches inside attention."""

    def test_rejects_dit_deeper_than_backbone(self, tiny_config: XR1Config, tiny_vlm: XR1Qwen3VL) -> None:
        """Each DiT layer reads one cached VLM layer, so it cannot be deeper."""
        config = dataclasses.replace(tiny_config, dit_num_layers=99)

        with pytest.raises(ValueError, match="exceeds the backbone's"):
            XR1Model(config, vlm=tiny_vlm)

    def test_rejects_head_dim_mismatch(self, tiny_config: XR1Config, tiny_vlm: XR1Qwen3VL) -> None:
        """Rotary embeddings are applied to cached VLM keys, so head dims must agree."""
        config = dataclasses.replace(tiny_config, dit_hidden_size=256, dit_head_dim=128, dit_kv_heads=2)

        with pytest.raises(ValueError, match="must match the backbone head dim"):
            XR1Model(config, vlm=tiny_vlm)

    def test_rejects_kv_head_mismatch(self, tiny_config: XR1Config, tiny_vlm: XR1Qwen3VL) -> None:
        """The cache carries a fixed number of kv heads."""
        config = dataclasses.replace(tiny_config, dit_kv_heads=4, dit_hidden_size=256, dit_head_dim=32)

        with pytest.raises(ValueError, match="num_key_value_heads"):
            XR1Model(config, vlm=tiny_vlm)


class TestTraining:
    """The flow-matching training path."""

    def test_loss_and_metrics(self, model: XR1Model, model_batch: dict[str, torch.Tensor]) -> None:
        """Training returns a finite scalar loss plus its component terms."""
        model.train()

        loss, metrics = model.compute_loss(model_batch)

        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert {"loss", "loss_mse", "loss_freq"} <= set(metrics)

    def test_gradients_reach_action_expert(
        self,
        model: XR1Model,
        model_batch: dict[str, torch.Tensor],
    ) -> None:
        """Every DiT layer and the output head must train."""
        model.train()

        loss, _ = model.compute_loss(model_batch)
        loss.backward()

        assert model.action_output_layer.layers[0].weight.grad is not None
        for index, layer in enumerate(model.dit.layers):
            assert layer.attn.qkv_proj.weight.grad is not None, f"DiT layer {index} got no gradient"

    def test_training_repeat_expands_the_batch(
        self,
        tiny_config: XR1Config,
        tiny_vlm: XR1Qwen3VL,
        model_batch: dict[str, torch.Tensor],
    ) -> None:
        """Repeats let one backbone pass serve several timesteps."""
        model = XR1Model(dataclasses.replace(tiny_config, training_repeat=3), vlm=tiny_vlm)
        model.train()

        loss, _ = model.compute_loss(model_batch)

        assert torch.isfinite(loss)

    def test_metrics_are_detached(self, model: XR1Model, model_batch: dict[str, torch.Tensor]) -> None:
        """Logged metrics must not hold the graph alive."""
        model.train()

        _, metrics = model.compute_loss(model_batch)

        assert not metrics["loss"].requires_grad  # type: ignore[union-attr]


class TestInference:
    """The sampling path."""

    def test_chunk_shape(self, model: XR1Model, model_batch: dict[str, torch.Tensor]) -> None:
        """Predictions match the configured horizon and padded action width."""
        model.eval()

        chunk = model.predict_action_chunk(model_batch)

        assert chunk.shape == (BATCH, model.config.chunk_size, model.config.max_action_dim)
        assert torch.isfinite(chunk).all()

    def test_forward_dispatches_on_mode(self, model: XR1Model, model_batch: dict[str, torch.Tensor]) -> None:
        """forward() trains in train mode and predicts in eval mode."""
        model.train()
        assert isinstance(model(model_batch), tuple)

        model.eval()
        assert isinstance(model(model_batch), torch.Tensor)

    def test_validation_loss_scores_a_rollout(
        self,
        model: XR1Model,
        model_batch: dict[str, torch.Tensor],
    ) -> None:
        """Validation integrates the flow and compares actions directly."""
        model.eval()

        loss, metrics = model.compute_val_loss(model_batch)

        assert torch.isfinite(loss)
        assert "action_mse" in metrics

    def test_action_prefix_is_preserved(
        self,
        tiny_config: XR1Config,
        tiny_vlm: XR1Qwen3VL,
        model_batch: dict[str, torch.Tensor],
    ) -> None:
        """Asynchronous execution conditions on already-issued actions."""
        model = XR1Model(tiny_config, vlm=tiny_vlm)
        model.eval()
        prefix = torch.full((BATCH, tiny_config.chunk_size, tiny_config.max_action_dim), 0.5)
        batch = {**model_batch, "action_prefix": prefix, "prefix_length": 2}

        chunk = model.predict_action_chunk(batch)

        assert torch.allclose(chunk[:, :2], prefix[:, :2])


class TestFreezing:
    """Memory-saving options."""

    def test_freeze_vlm(self, tiny_config: XR1Config, tiny_vlm: XR1Qwen3VL) -> None:
        """Freezing the backbone is what makes a 24 GB card viable."""
        model = XR1Model(dataclasses.replace(tiny_config, freeze_vlm=True), vlm=tiny_vlm)

        assert not any(p.requires_grad for p in model.vlm.parameters())
        assert all(p.requires_grad for p in model.dit.parameters())

    def test_freeze_vision_encoder_only(self, tiny_config: XR1Config, tiny_vlm: XR1Qwen3VL) -> None:
        """The vision tower can be frozen while the language model trains."""
        model = XR1Model(dataclasses.replace(tiny_config, freeze_vision_encoder=True), vlm=tiny_vlm)

        assert not any(p.requires_grad for p in model.vlm.model.visual.parameters())
        assert any(p.requires_grad for p in model.vlm.model.language_model.parameters())


class TestChoiceHead:
    """The optional auxiliary head."""

    def test_absent_by_default(self, model: XR1Model) -> None:
        """Default configurations carry no choice projectors."""
        assert not hasattr(model, "action_projector_choice")

    def test_built_when_enabled(self, tiny_config: XR1Config, tiny_vlm: XR1Qwen3VL) -> None:
        """Enabling the head adds all three projectors."""
        model = XR1Model(dataclasses.replace(tiny_config, enable_choice_head=True), vlm=tiny_vlm)

        assert hasattr(model, "action_projector_choice")
        assert hasattr(model, "score_projector_choice")
        assert hasattr(model, "state_projector_choice")

    def test_explains_missing_supervision(
        self,
        tiny_config: XR1Config,
        tiny_vlm: XR1Qwen3VL,
        model_batch: dict[str, torch.Tensor],
    ) -> None:
        """LeRobot datasets lack choice targets; the error must say so."""
        model = XR1Model(dataclasses.replace(tiny_config, enable_choice_head=True), vlm=tiny_vlm)
        model.train()

        with pytest.raises(KeyError, match="LeRobot datasets do not provide"):
            model.compute_loss(model_batch)


class TestDeltaIndices:
    """Dataset indexing contract."""

    def test_action_chunk_indices(self, model: XR1Model) -> None:
        """One chunk spans chunk_size consecutive future actions."""
        assert model.action_delta_indices == list(range(model.config.chunk_size))

    def test_no_reward_or_history(self, model: XR1Model) -> None:
        """XR1 conditions on the current observation and uses no rewards."""
        assert model.reward_delta_indices is None
        assert model.observation_delta_indices is None
