# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the XR1 vision-language-action model."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest
import torch
from physicalai.policies.xr1 import XR1Config, XR1Model
from physicalai.policies.xr1.io import continue_text_position_ids

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
    """The Choice Policy auxiliary head."""

    def test_enabled_by_default(self, model: XR1Model) -> None:
        """The reference recipe trains the head, so the default builds it."""
        assert model.config.enable_choice_head is True
        assert hasattr(model, "action_projector_choice")
        assert hasattr(model, "score_projector_choice")
        assert hasattr(model, "state_projector_choice")
        assert model.action_query_embed.num_embeddings == model.config.chunk_size
        assert model.score_query_embed.num_embeddings == 1

    def test_absent_when_disabled(self, tiny_config: XR1Config, tiny_vlm: XR1Qwen3VL) -> None:
        """Disabling the head leaves no choice parameters behind."""
        model = XR1Model(dataclasses.replace(tiny_config, enable_choice_head=False), vlm=tiny_vlm)

        assert not hasattr(model, "action_projector_choice")
        assert not hasattr(model, "action_query_embed")

    def test_contributes_its_own_loss_terms(
        self,
        model: XR1Model,
        model_batch: dict[str, torch.Tensor],
    ) -> None:
        """Both auxiliary terms are reported and finite."""
        model.train()

        _, metrics = model.compute_loss(model_batch)

        assert {"loss_choice", "loss_score"} <= set(metrics)
        assert torch.isfinite(metrics["loss_choice"])  # type: ignore[arg-type]
        assert torch.isfinite(metrics["loss_score"])  # type: ignore[arg-type]

    def test_supervised_by_the_action_chunk(
        self,
        model: XR1Model,
        model_batch: dict[str, torch.Tensor],
    ) -> None:
        """No dataset fields beyond the action chunk are required."""
        model.train()
        batch = {key: value for key, value in model_batch.items() if key != "action_mask"}

        with pytest.raises(KeyError, match="action_mask"):
            model.compute_loss(batch)

    def test_query_tokens_stay_out_of_the_action_expert(
        self,
        model: XR1Model,
        model_batch: dict[str, torch.Tensor],
    ) -> None:
        """The paper requires the choice tokens be invisible to the action expert.

        Otherwise the expert could copy a candidate instead of learning the flow. The
        cache the expert reads must therefore still be exactly the prompt length after
        the choice branch has run.
        """
        model.train()
        prompt_length = model_batch["input_ids"].shape[1]
        outputs = model.encode_prompt(model_batch)

        model._choice_loss(model_batch, outputs)  # noqa: SLF001 - asserting on an internal invariant

        for index, (keys, values) in enumerate(outputs.past_key_values):
            assert keys.shape[-2] == prompt_length, f"layer {index} cache grew to {keys.shape[-2]}"
            assert values.shape[-2] == prompt_length

    def test_gradients_reach_the_head(
        self,
        model: XR1Model,
        model_batch: dict[str, torch.Tensor],
    ) -> None:
        """The query embeddings and both projectors must train."""
        model.train()

        loss, _ = model.compute_loss(model_batch)
        loss.backward()

        assert model.action_query_embed.weight.grad is not None
        assert model.score_query_embed.weight.grad is not None
        assert model.action_projector_choice[0].layers[0].weight.grad is not None
        assert model.score_projector_choice[0].layers[0].weight.grad is not None

    def test_winner_takes_all(self, model: XR1Model, model_batch: dict[str, torch.Tensor]) -> None:
        """Only the closest candidate is trained, so the loss is the minimum error."""
        model.train()
        outputs = model.encode_prompt(model_batch)
        target = model_batch["action"]
        mask = model_batch["action_mask"]

        queries = model._choice_queries(model_batch["state"], target.shape[1])  # noqa: SLF001
        hidden = model.vlm.continue_sequence(
            queries,
            cache=outputs.cache,
            prompt_attention_mask=outputs.attention_mask,
            position_ids=continue_text_position_ids(outputs.position_ids, queries.shape[1]),
        )
        candidates, _ = model._choice_predictions(hidden, 1, target.shape[1])  # noqa: SLF001
        per_choice = ((candidates - target[:, :, None, :]).abs() * mask[:, :, None, :]).sum(dim=(1, 3)) / mask.sum(
            dim=(1, 2)
        )[:, None]

        loss_choice, _ = model._choice_loss(model_batch, outputs)  # noqa: SLF001

        assert torch.allclose(loss_choice, per_choice.min(dim=-1).values.mean(), atol=1e-5)

    def test_masked_steps_do_not_contribute(
        self,
        model: XR1Model,
        model_batch: dict[str, torch.Tensor],
    ) -> None:
        """A chunk running past the end of an episode is masked, not learned from."""
        model.train()
        outputs = model.encode_prompt(model_batch)
        masked = dict(model_batch)
        masked["action_mask"] = model_batch["action_mask"].clone()
        masked["action_mask"][:, 2:] = 0.0
        perturbed = dict(masked)
        perturbed["action"] = model_batch["action"].clone()
        perturbed["action"][:, 2:] += 100.0

        baseline, _ = model._choice_loss(masked, outputs)  # noqa: SLF001
        changed, _ = model._choice_loss(perturbed, outputs)  # noqa: SLF001

        assert torch.allclose(baseline, changed)

    def test_rejects_a_horizon_past_the_query_table(self, model: XR1Model) -> None:
        """There is one learned query per step, so a longer chunk has nowhere to go."""
        state = torch.randn(BATCH, 1, model.config.max_state_dim)

        with pytest.raises(ValueError, match="exceeds chunk_size"):
            model._choice_queries(state, model.config.chunk_size + 1)  # noqa: SLF001


class TestChoiceTurnAttention:
    """The appended turn must behave exactly like tokens in the same chat turn.

    The two-pass construction is only equivalent to upstream's one-pass-then-truncate
    if the appended tokens attend to the whole prompt and causally to each other. These
    tests check that directly rather than trusting the cache plumbing.
    """

    @staticmethod
    def _continue(model: XR1Model, batch: dict[str, torch.Tensor], queries: torch.Tensor) -> torch.Tensor:
        """Run the appended tokens over a freshly encoded prompt.

        Args:
            model: The model.
            batch: Preprocessed batch.
            queries: Appended embeddings.

        Returns:
            Hidden states for the appended tokens.
        """
        outputs = model.encode_prompt(batch)
        return model.vlm.continue_sequence(
            queries,
            cache=outputs.cache,  # type: ignore[arg-type]
            prompt_attention_mask=outputs.attention_mask,
            position_ids=continue_text_position_ids(outputs.position_ids, queries.shape[1]),
        )

    def test_is_causal(self, model: XR1Model, model_batch: dict[str, torch.Tensor]) -> None:
        """Perturbing a later query must leave the earlier ones untouched."""
        torch.manual_seed(0)
        queries = torch.randn(BATCH, 4, model.vlm.config.text_config.hidden_size)
        perturbed = queries.clone()
        perturbed[:, 3] += 10.0

        base = self._continue(model, model_batch, queries)
        after = self._continue(model, model_batch, perturbed)

        assert torch.allclose(base[:, :3], after[:, :3], atol=1e-5)
        assert not torch.allclose(base[:, 3], after[:, 3], atol=1e-3)

    def test_score_token_sees_the_action_queries(
        self,
        model: XR1Model,
        model_batch: dict[str, torch.Tensor],
    ) -> None:
        """Scores must depend on the candidates they are scoring."""
        torch.manual_seed(0)
        queries = torch.randn(BATCH, 4, model.vlm.config.text_config.hidden_size)
        perturbed = queries.clone()
        perturbed[:, 1] += 10.0

        base = self._continue(model, model_batch, queries)
        after = self._continue(model, model_batch, perturbed)

        assert not torch.allclose(base[:, -1], after[:, -1], atol=1e-3)

    def test_conditions_on_the_prompt(self, model: XR1Model, model_batch: dict[str, torch.Tensor]) -> None:
        """A different instruction must produce different candidates."""
        torch.manual_seed(0)
        queries = torch.randn(BATCH, 4, model.vlm.config.text_config.hidden_size)
        other = dict(model_batch)
        other["input_ids"] = (model_batch["input_ids"] + 1) % model.vlm.config.text_config.vocab_size

        base = self._continue(model, model_batch, queries)
        after = self._continue(model, other, queries)

        assert not torch.allclose(base, after, atol=1e-3)


class TestDeltaIndices:
    """Dataset indexing contract."""

    def test_action_chunk_indices(self, model: XR1Model) -> None:
        """One chunk spans chunk_size consecutive future actions."""
        assert model.action_delta_indices == list(range(model.config.chunk_size))

    def test_no_reward_or_history(self, model: XR1Model) -> None:
        """XR1 conditions on the current observation and uses no rewards."""
        assert model.reward_delta_indices is None
        assert model.observation_delta_indices is None
