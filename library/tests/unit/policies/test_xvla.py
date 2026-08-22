# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the XVLA policy.

Fast and self-contained: the Florence-2 backbone is built from a miniature architecture and
the BART tokenizer is replaced with a stub, so the whole train/infer/checkpoint path runs
without any download.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from physicalai.data import Observation
from physicalai.policies import get_policy
from physicalai.policies.xvla import XVLA, XVLAConfig, XVLAModel
from physicalai.policies.xvla.action_hub import ACTION_REGISTRY, build_action_space
from physicalai.policies.xvla.preprocessor import (
    build_features,
    make_xvla_preprocessors,
    resize_with_pad,
    resolve_num_image_views,
)

pytest.importorskip("transformers")

IMAGE_SIZE = 64
STATE_DIM = 6
ACTION_DIM = 7
CHUNK_SIZE = 4
BATCH_SIZE = 2

TINY_FLORENCE: dict[str, Any] = {
    "vision_config": {
        "model_type": "florence_vision",
        "depths": [1, 1, 1, 1],
        "embed_dim": [8, 16, 24, 32],
        "num_heads": [1, 1, 1, 2],
        "num_groups": [1, 1, 1, 2],
        "window_size": 2,
        "projection_dim": 32,
        "max_temporal_embeddings": 4,
        "max_position_embeddings": 16,
    },
    "text_config": {
        "model_type": "bart",
        "d_model": 32,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "encoder_attention_heads": 2,
        "decoder_attention_heads": 2,
        "encoder_ffn_dim": 32,
        "decoder_ffn_dim": 32,
        "vocab_size": 99,
        "max_position_embeddings": 64,
    },
}

VOCAB_SIZE = TINY_FLORENCE["text_config"]["vocab_size"]

DATASET_STATS: dict[str, dict[str, Any]] = {
    "observation.state": {
        "name": "observation.state",
        "type": "STATE",
        "shape": (STATE_DIM,),
        "mean": [0.0] * STATE_DIM,
        "std": [1.0] * STATE_DIM,
        "q01": [-1.0] * STATE_DIM,
        "q99": [1.0] * STATE_DIM,
    },
    "action": {
        "name": "action",
        "type": "ACTION",
        "shape": (ACTION_DIM,),
        "mean": [0.0] * ACTION_DIM,
        "std": [1.0] * ACTION_DIM,
        "q01": [-1.0] * ACTION_DIM,
        "q99": [1.0] * ACTION_DIM,
    },
    "observation.images.top": {
        "name": "observation.images.top",
        "type": "VISUAL",
        "shape": (3, IMAGE_SIZE, IMAGE_SIZE),
    },
    "observation.images.wrist": {
        "name": "observation.images.wrist",
        "type": "VISUAL",
        "shape": (3, IMAGE_SIZE, IMAGE_SIZE),
    },
}


def tiny_kwargs(**overrides: Any) -> dict[str, Any]:
    """Build the constructor arguments of a miniature but complete XVLA.

    Args:
        **overrides: Fields to override on top of the miniature defaults.

    Returns:
        Keyword arguments for :class:`XVLA` or :class:`XVLAConfig`.
    """
    kwargs: dict[str, Any] = {
        "florence_config": TINY_FLORENCE,
        "tokenizer_max_length": 8,
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": CHUNK_SIZE,
        "hidden_size": 32,
        "depth": 1,
        "num_heads": 2,
        "num_domains": 3,
        "len_soft_prompts": 2,
        "dim_time": 8,
        "max_len_seq": 64,
        "max_state_dim": 8,
        "max_action_dim": 8,
        "num_denoising_steps": 2,
    }
    kwargs.update(overrides)
    return kwargs


class FakeTokenizer:
    """Deterministic stand-in for Florence-2's BART tokenizer."""

    def __call__(self, prompts: list[str], max_length: int = 8, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Tokenize by hashing each prompt into a fixed-length id sequence.

        Args:
            prompts: Prompts to encode.
            max_length: Fixed output length.
            **kwargs: Ignored padding/truncation options.

        Returns:
            Dict with ``input_ids`` of shape ``[len(prompts), max_length]``.
        """
        del kwargs
        ids = torch.tensor(
            [[(hash(prompt) + i) % VOCAB_SIZE for i in range(max_length)] for prompt in prompts],
            dtype=torch.long,
        )
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def build_policy(stats: dict[str, dict[str, Any]] | None = None, **overrides: Any) -> XVLA:
    """Build a miniature policy with its model and a stubbed tokenizer.

    Args:
        stats: Dataset statistics; defaults to :data:`DATASET_STATS`.
        **overrides: Config overrides.

    Returns:
        A ready-to-run policy.
    """
    policy = XVLA(**tiny_kwargs(**overrides))
    policy._initialize_model(DATASET_STATS if stats is None else stats)  # noqa: SLF001
    policy._preprocessor._tokenizer = FakeTokenizer()  # noqa: SLF001
    return policy


def make_observation(
    batch_size: int = BATCH_SIZE,
    action_dim: int = ACTION_DIM,
    chunk_size: int = CHUNK_SIZE,
) -> Observation:
    """Build a synthetic observation batch matching :data:`DATASET_STATS`.

    Args:
        batch_size: Number of samples.
        action_dim: Width of the ground-truth actions.
        chunk_size: Number of action steps.

    Returns:
        The observation batch.
    """
    return Observation(
        state=torch.randn(batch_size, STATE_DIM),
        action=torch.randn(batch_size, chunk_size, action_dim),
        task=["pick up the cube"] * batch_size,
        images={
            "top": torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE),
            "wrist": torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE),
        },
    )


# ============================================================================ #
# Configuration                                                                #
# ============================================================================ #


class TestXVLAConfig:
    """Tests for the configuration dataclass."""

    def test_defaults_match_upstream(self) -> None:
        """The published defaults are carried over from the reference implementation."""
        config = XVLAConfig()
        assert config.chunk_size == 32
        assert config.n_action_steps == 32
        assert config.hidden_size == 1024
        assert config.depth == 24
        assert config.num_domains == 30
        assert config.len_soft_prompts == 32
        assert config.max_action_dim == 20
        assert config.num_denoising_steps == 10
        assert config.tokenizer_name == "facebook/bart-large"

    def test_action_mode_defaults_to_auto(self) -> None:
        """Studio defaults to the embodiment-agnostic action space."""
        assert XVLAConfig().action_mode == "auto"

    def test_dim_proprio_follows_use_proprio(self) -> None:
        """Disabling proprioception collapses the state width to zero."""
        assert XVLAConfig().dim_proprio == 32
        assert XVLAConfig(use_proprio=False).dim_proprio == 0

    def test_n_action_steps_bounded_by_chunk(self) -> None:
        """More executed steps than predicted ones is rejected."""
        with pytest.raises(ValueError, match="cannot be greater than chunk_size"):
            XVLAConfig(chunk_size=8, n_action_steps=16)

    def test_unknown_action_mode(self) -> None:
        """An unregistered action space names the available ones."""
        with pytest.raises(ValueError, match="Unknown action_mode"):
            XVLAConfig(action_mode="nope")

    def test_invalid_dtype(self) -> None:
        """Only the two supported precisions are accepted."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            XVLAConfig(dtype="float64")

    def test_inheritance_and_serialization(self) -> None:
        """The config is a Studio ``Config`` and survives a dict round-trip."""
        from physicalai.config import Config  # noqa: PLC0415

        config = XVLAConfig(**tiny_kwargs())
        assert isinstance(config, Config)
        assert XVLAConfig.from_dict(config.to_dict()) == config

    def test_from_dict_coerces_json_lists(self) -> None:
        """A config parsed from JSON compares equal to one built from tuples."""
        config = XVLAConfig.from_dict({
            "optimizer_betas": [0.9, 0.99],
            "resize_imgs_with_padding": [224, 224],
        })
        assert config.optimizer_betas == (0.9, 0.99)
        assert config.resize_imgs_with_padding == (224, 224)

    def test_build_florence_config_from_defaults(self) -> None:
        """An empty ``florence_config`` falls back to the transformers defaults."""
        florence = XVLAConfig().build_florence_config()
        assert florence.vision_config.projection_dim > 0

    def test_build_florence_config_translates_legacy_format(self) -> None:
        """The legacy remote-code vision config is mapped onto the native field names."""
        config = XVLAConfig(
            florence_config={
                "vision_config": {
                    "model_type": "davit",
                    "dim_embed": [8, 16, 24, 32],
                    "projection_dim": 32,
                    "image_pos_embed": {"type": "learned_abs_2d", "max_pos_embeddings": 16},
                    "visual_temporal_embedding": {"type": "COSINE", "max_temporal_embeddings": 4},
                    "image_feature_source": ["spatial_avg_pool", "temporal_avg_pool"],
                },
                "text_config": {"d_model": 32, "vocab_size": 99},
            },
        )
        florence = config.build_florence_config()
        assert list(florence.vision_config.embed_dim) == [8, 16, 24, 32]
        assert florence.vision_config.max_position_embeddings == 16
        assert florence.vision_config.max_temporal_embeddings == 4
        # The legacy language config is BART, field for field.
        assert florence.text_config.model_type == "bart"

    def test_build_florence_config_rejects_unknown_backbone(self) -> None:
        """A vision backbone the native implementation cannot reproduce is refused."""
        config = XVLAConfig(florence_config={"vision_config": {"model_type": "convnext"}, "text_config": {}})
        with pytest.raises(ValueError, match="Unsupported Florence-2 vision backbone"):
            config.build_florence_config()


# ============================================================================ #
# Action spaces                                                                #
# ============================================================================ #


class TestXVLAActionSpaces:
    """Tests for the action-space registry."""

    def test_registry_contents(self) -> None:
        """Every upstream action space is registered."""
        assert set(ACTION_REGISTRY) == {
            "ee6d",
            "joint",
            "agibot_ee6d",
            "franka_joint7",
            "auto",
            "so101_bimanual",
        }

    def test_unknown_space(self) -> None:
        """Building an unregistered space lists the available ones."""
        with pytest.raises(KeyError, match="Unknown action space"):
            build_action_space("nope")

    def test_auto_pads_and_trims(self) -> None:
        """``auto`` widens the dataset's actions for the model and narrows them back."""
        space = build_action_space("auto", real_dim=7, max_dim=20)
        assert space.dim_action == 20

        _, padded = space.preprocess(torch.zeros(2, 32), torch.randn(2, 4, 7))
        assert padded.shape == (2, 4, 20)
        assert torch.count_nonzero(padded[..., 7:]) == 0
        assert space.postprocess(torch.randn(2, 4, 20)).shape == (2, 4, 7)

    def test_auto_loss_ignores_padding(self) -> None:
        """Only the dataset's real channels are supervised."""
        space = build_action_space("auto", real_dim=3, max_dim=8)
        pred = torch.zeros(1, 2, 8)
        target = torch.zeros(1, 2, 3)
        pred[..., 3:] = 100.0  # padding channels must not enter the loss
        assert space.compute_loss(pred, target)["joints_loss"].item() == pytest.approx(0.0)

    def test_auto_rejects_impossible_widths(self) -> None:
        """A dataset wider than the model's action vector is refused up front."""
        with pytest.raises(ValueError, match="real_dim must be in"):
            build_action_space("auto", real_dim=32, max_dim=20)

    def test_ee6d_layout(self) -> None:
        """``ee6d`` supervises position, rotation and grippers separately."""
        space = build_action_space("ee6d")
        assert space.dim_action == 20
        assert space.gripper_idx == (9, 19)
        losses = space.compute_loss(torch.randn(2, 4, 20), torch.rand(2, 4, 20))
        assert set(losses) == {"position_loss", "rotate6D_loss", "gripper_loss"}

    def test_ee6d_postprocess_squashes_grippers(self) -> None:
        """Gripper logits become probabilities; the other channels are untouched."""
        space = build_action_space("ee6d")
        raw = torch.zeros(1, 1, 20)
        raw[..., 3] = 7.0
        decoded = space.postprocess(raw)
        assert decoded[0, 0, 9].item() == pytest.approx(0.5)
        assert decoded[0, 0, 19].item() == pytest.approx(0.5)
        assert decoded[0, 0, 3].item() == pytest.approx(7.0)

    def test_ee6d_masks_grippers_from_the_transformer_input(self) -> None:
        """The noised gripper value is not fed back into the transformer."""
        space = build_action_space("ee6d")
        proprio, action = space.preprocess(torch.ones(2, 20), torch.ones(2, 4, 20))
        assert torch.count_nonzero(action[..., list(space.gripper_idx)]) == 0
        assert torch.count_nonzero(proprio[..., list(space.gripper_idx)]) == 0

    def test_so101_trims_to_twelve_channels(self) -> None:
        """The bimanual SO-101 space emits a 12-dim control vector."""
        space = build_action_space("so101_bimanual")
        assert space.postprocess(torch.randn(2, 4, 20)).shape == (2, 4, 12)

    def test_shape_mismatch_is_reported(self) -> None:
        """A fixed-layout space refuses mismatched prediction and target shapes."""
        space = build_action_space("ee6d")
        with pytest.raises(ValueError, match="shapes must match"):
            space.compute_loss(torch.randn(2, 4, 20), torch.randn(2, 4, 14))


# ============================================================================ #
# Registration                                                                 #
# ============================================================================ #


class TestXVLARegistration:
    """Tests for the policy factory and the lazy construction path."""

    def test_get_policy(self) -> None:
        """XVLA is reachable through the first-party factory."""
        assert isinstance(get_policy("xvla"), XVLA)

    def test_get_policy_is_case_insensitive(self) -> None:
        """Policy names are matched case-insensitively."""
        assert isinstance(get_policy("XVLA"), XVLA)

    def test_package_exports(self) -> None:
        """The family's three public names are exported from ``physicalai.policies``."""
        import physicalai.policies as policies  # noqa: PLC0415

        assert {"XVLA", "XVLAConfig", "XVLAModel"} <= set(policies.__all__)
        assert policies.XVLAModel is XVLAModel

    def test_lazy_initialization(self) -> None:
        """Constructing without statistics defers the model to ``setup()``."""
        policy = XVLA(**tiny_kwargs())
        assert policy.model is None
        with pytest.raises(RuntimeError, match="before the model was initialized"):
            _ = policy.inner_model

    def test_hyperparameters_saved(self) -> None:
        """The resolved config is recorded for checkpointing."""
        policy = XVLA(**tiny_kwargs(chunk_size=8, n_action_steps=8))
        assert policy.hparams["chunk_size"] == 8
        assert policy.hparams["config"]["chunk_size"] == 8

    def test_action_queue_sized_from_config(self) -> None:
        """The base action queue is sized from ``n_action_steps``."""
        policy = XVLA(**tiny_kwargs(chunk_size=8, n_action_steps=3))
        assert policy._action_queue.maxlen == 3  # noqa: SLF001


# ============================================================================ #
# Preprocessing                                                                #
# ============================================================================ #


class TestXVLAPreprocessing:
    """Tests for the observation preprocessor."""

    def test_resize_with_pad_keeps_aspect_ratio(self) -> None:
        """Images are resized without distortion and padded to the target size."""
        resized = resize_with_pad(torch.rand(2, 3, 32, 64), 64, 64)
        assert resized.shape == (2, 3, 64, 64)
        # Padding goes on the left and top (the XVLA convention).
        assert torch.count_nonzero(resized[:, :, :32, :]) == 0

    def test_resize_with_pad_requires_bchw(self) -> None:
        """A non-4D input is reported rather than silently reshaped."""
        with pytest.raises(ValueError, match=r"\(B, C, H, W\) expected"):
            resize_with_pad(torch.rand(3, 32, 32), 64, 64)

    def test_resolve_num_image_views(self) -> None:
        """Camera slots come from the dataset, the config override and the empty slots."""
        assert resolve_num_image_views(DATASET_STATS) == 2
        assert resolve_num_image_views(DATASET_STATS, empty_cameras=2) == 4
        assert resolve_num_image_views(DATASET_STATS, num_image_views=5) == 5
        # Never fewer than one, even without statistics.
        assert resolve_num_image_views(None) == 1

    def test_images_are_stacked_per_view(self) -> None:
        """Cameras are stacked into one tensor with a per-view validity mask."""
        policy = build_policy()
        batch = policy._preprocess(make_observation())  # noqa: SLF001
        assert batch["images"].shape == (BATCH_SIZE, 2, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert batch["image_masks"].shape == (BATCH_SIZE, 2)
        assert bool(batch["image_masks"].all())

    def test_empty_cameras_are_masked_out(self) -> None:
        """Extra camera slots are appended as zero images the model must ignore."""
        policy = build_policy(empty_cameras=2)
        batch = policy._preprocess(make_observation())  # noqa: SLF001
        assert batch["images"].shape[1] == 4
        assert bool(batch["image_masks"][:, :2].all())
        assert not bool(batch["image_masks"][:, 2:].any())
        assert torch.count_nonzero(batch["images"][:, 2:]) == 0

    def test_images_are_imagenet_normalized(self) -> None:
        """Cameras leave the preprocessor in Florence-2's expected distribution."""
        policy = build_policy()
        observation = make_observation()
        observation.images = {k: torch.full_like(v, 0.485) for k, v in observation.images.items()}
        batch = policy._preprocess(observation)  # noqa: SLF001
        # The first channel is exactly the ImageNet mean, so it normalizes to zero.
        torch.testing.assert_close(
            batch["images"][:, :, 0],
            torch.zeros_like(batch["images"][:, :, 0]),
            atol=1e-6,
            rtol=0,
        )

    def test_uint8_images_are_rescaled(self) -> None:
        """Byte images are divided by 255 before normalization."""
        policy = build_policy()
        float_observation = make_observation(batch_size=1)
        byte_images = {k: (v * 255).to(torch.uint8) for k, v in float_observation.images.items()}
        float_observation.images = {k: v.float() / 255.0 for k, v in byte_images.items()}
        expected = policy._preprocess(float_observation)["images"]  # noqa: SLF001

        byte_observation = make_observation(batch_size=1)
        byte_observation.images = byte_images
        torch.testing.assert_close(policy._preprocess(byte_observation)["images"], expected)  # noqa: SLF001

    def test_temporal_clips_use_the_latest_frame(self) -> None:
        """A ``[B, T, C, H, W]`` camera collapses to its most recent frame."""
        policy = build_policy()
        observation = make_observation(batch_size=1)
        clip = torch.rand(1, 3, 3, IMAGE_SIZE, IMAGE_SIZE)
        observation.images = {"top": clip, "wrist": clip}
        batch = policy._preprocess(observation)  # noqa: SLF001
        assert batch["images"].shape == (1, 2, 3, IMAGE_SIZE, IMAGE_SIZE)

    def test_state_is_padded_to_the_model_width(self) -> None:
        """Proprioception is zero-padded to the model's fixed width."""
        policy = build_policy()
        batch = policy._preprocess(make_observation())  # noqa: SLF001
        assert batch["state"].shape == (BATCH_SIZE, 8)
        assert torch.count_nonzero(batch["state"][:, STATE_DIM:]) == 0

    def test_state_is_truncated_when_too_wide(self) -> None:
        """A dataset wider than ``max_state_dim`` is truncated, not rejected."""
        policy = build_policy(max_state_dim=4)
        batch = policy._preprocess(make_observation())  # noqa: SLF001
        assert batch["state"].shape == (BATCH_SIZE, 4)

    def test_proprioception_can_be_disabled(self) -> None:
        """``use_proprio=False`` drops the state from the action tokens entirely."""
        policy = build_policy(use_proprio=False)
        assert policy.inner_model.dim_proprio == 0
        batch = policy._preprocess(make_observation())  # noqa: SLF001
        assert batch["state"].shape == (BATCH_SIZE, 0)

    def test_domain_id_defaults_to_the_config(self) -> None:
        """A batch without a domain index falls back to the configured one."""
        policy = build_policy(domain_id=2)
        batch = policy._preprocess(make_observation())  # noqa: SLF001
        assert torch.equal(batch["domain_id"], torch.full((BATCH_SIZE,), 2, dtype=torch.long))

    def test_domain_id_read_from_the_batch(self) -> None:
        """A per-sample domain index in ``extra`` wins over the configured default."""
        policy = build_policy(domain_id=2)
        observation = make_observation()
        observation.extra = {"domain_id": torch.tensor([1, 0])}
        batch = policy._preprocess(observation)  # noqa: SLF001
        assert torch.equal(batch["domain_id"], torch.tensor([1, 0]))

    def test_domain_id_from_a_custom_key(self) -> None:
        """``domain_feature_key`` points at an arbitrary batch entry."""
        policy = build_policy(domain_feature_key="extra.embodiment")
        observation = make_observation()
        observation.extra = {"embodiment": torch.tensor([2, 2])}
        batch = policy._preprocess(observation)  # noqa: SLF001
        assert torch.equal(batch["domain_id"], torch.tensor([2, 2]))

    def test_prompt_is_tokenized_to_a_fixed_length(self) -> None:
        """The prompt is padded and truncated so the sequence length never varies."""
        policy = build_policy()
        batch = policy._preprocess(make_observation())  # noqa: SLF001
        assert batch["tokenized_prompt"].shape == (BATCH_SIZE, 8)

    def test_missing_cameras_are_reported(self) -> None:
        """A batch without any camera names the keys it did carry."""
        policy = build_policy()
        observation = make_observation()
        observation.images = None
        with pytest.raises(ValueError, match="requires at least one camera"):
            policy._preprocess(observation)  # noqa: SLF001

    def test_identity_normalization_leaves_actions_alone(self) -> None:
        """The default mode matches the units the published checkpoints train on."""
        pre, post = make_xvla_preprocessors(DATASET_STATS)
        actions = torch.randn(1, CHUNK_SIZE, ACTION_DIM)
        torch.testing.assert_close(post({"action": actions})["action"], actions)

    def test_normalization_round_trip(self) -> None:
        """Opting into quantile normalization keeps pre- and postprocessing inverse."""
        stats = {
            "action": {"name": "action", "type": "ACTION", "shape": (3,), "q01": [-2.0, -1.0, 0.0], "q99": [2.0, 3.0, 1.0]},
        }
        pre, post = make_xvla_preprocessors(stats, normalization_mode="QUANTILES")
        actions = torch.tensor([[[1.0, 0.5, 0.25]]])
        normalized = pre._normalizer({"action": actions.clone()})["action"]  # noqa: SLF001
        assert normalized.abs().max() <= 1.0
        torch.testing.assert_close(post({"action": normalized})["action"], actions)

    def test_build_features_maps_dataset_names(self) -> None:
        """Dataset feature names are mapped onto Studio's flattened batch keys."""
        features = build_features(DATASET_STATS)
        assert set(features) == {"state", "action"}
        assert features["action"].shape == (ACTION_DIM,)


# ============================================================================ #
# Inference                                                                    #
# ============================================================================ #


class TestXVLAInference:
    """Shape and behaviour tests for the denoising inference path."""

    def test_predict_action_chunk_shape(self) -> None:
        """A chunk covers the configured horizon at the dataset's action width."""
        policy = build_policy()
        policy.eval()
        actions = policy.predict_action_chunk(make_observation())
        assert actions.shape == (BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)
        assert actions.dtype == torch.float32
        assert bool(torch.isfinite(actions).all())

    def test_forward_predicts_in_eval_mode(self) -> None:
        """``forward`` switches between the loss and the inference path."""
        policy = build_policy()
        policy.eval()
        assert policy(make_observation()).shape == (BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)

    def test_select_action_returns_single_action(self) -> None:
        """``select_action`` hands back one action per call."""
        policy = build_policy()
        policy.eval()
        assert policy.select_action(make_observation(batch_size=1)).shape == (1, ACTION_DIM)

    def test_select_action_refills_the_queue(self) -> None:
        """Executing past the end of a chunk transparently predicts the next one."""
        policy = build_policy()
        policy.eval()
        observation = make_observation(batch_size=1)
        actions = [policy.select_action(observation) for _ in range(2 * CHUNK_SIZE + 1)]
        assert all(action.shape == (1, ACTION_DIM) for action in actions)
        assert all(bool(torch.isfinite(action).all()) for action in actions)

    def test_reset_clears_the_queue(self) -> None:
        """Resetting drops the actions left over from the previous episode."""
        policy = build_policy()
        policy.eval()
        policy.select_action(make_observation(batch_size=1))
        assert len(policy._action_queue) > 0  # noqa: SLF001
        policy.reset()
        assert len(policy._action_queue) == 0  # noqa: SLF001

    def test_n_action_steps_bounds_the_queue(self) -> None:
        """Only the first ``n_action_steps`` of a chunk are executed."""
        policy = build_policy(chunk_size=8, n_action_steps=2, max_len_seq=96)
        policy.eval()
        policy.select_action(make_observation(batch_size=1, chunk_size=8))
        # One action was returned, so one of the two executed steps remains queued.
        assert len(policy._action_queue) == 1  # noqa: SLF001

    def test_fixed_action_space_emits_its_own_width(self) -> None:
        """``ee6d`` emits its 20-channel layout regardless of the dataset."""
        policy = build_policy(action_mode="ee6d", max_action_dim=20)
        policy.eval()
        assert policy.predict_action_chunk(make_observation()).shape == (BATCH_SIZE, CHUNK_SIZE, 20)

    def test_actions_are_denormalized(self) -> None:
        """Predictions come back in the dataset's units, not the model's space."""
        stats = {
            "action": {
                "name": "action",
                "type": "ACTION",
                "shape": (ACTION_DIM,),
                "q01": [0.0] * ACTION_DIM,
                "q99": [10.0] * ACTION_DIM,
            },
        }
        policy = build_policy(stats=stats, normalization_mode="QUANTILES")
        policy.eval()
        observation = make_observation()

        torch.manual_seed(0)
        raw = policy.inner_model.predict_action_chunk(policy._preprocess(observation))  # noqa: SLF001
        torch.manual_seed(0)
        actions = policy.predict_action_chunk(observation)

        # q01=0, q99=10 maps the model's [-1, 1] space onto [0, 10].
        torch.testing.assert_close(actions, (raw + 1.0) * 5.0)

    def test_masked_views_do_not_change_the_valid_ones(self) -> None:
        """A masked-out camera contributes zero features, not noise."""
        policy = build_policy()
        policy.eval()
        model = policy.inner_model
        batch = policy._preprocess(make_observation(batch_size=1))  # noqa: SLF001

        encoded = model.encode_observation(batch["tokenized_prompt"], batch["images"], batch["image_masks"])
        masked = batch["image_masks"].clone()
        masked[:, 1] = False
        encoded_masked = model.encode_observation(batch["tokenized_prompt"], batch["images"], masked)

        # The primary view is unaffected; only the auxiliary stream is zeroed.
        torch.testing.assert_close(encoded["vlm_features"], encoded_masked["vlm_features"])
        assert torch.count_nonzero(encoded_masked["aux_visual_inputs"]) == 0

    def test_all_views_masked_is_reported(self) -> None:
        """An observation with no valid camera is refused."""
        policy = build_policy()
        model = policy.inner_model
        batch = policy._preprocess(make_observation(batch_size=1))  # noqa: SLF001
        with pytest.raises(ValueError, match="At least one image view must be valid"):
            model.encode_observation(
                batch["tokenized_prompt"],
                batch["images"],
                torch.zeros_like(batch["image_masks"]),
            )

    def test_sequence_overflow_is_reported(self) -> None:
        """A sequence longer than the positional embedding names the knobs to turn."""
        policy = build_policy(max_len_seq=8)
        policy.eval()
        with pytest.raises(ValueError, match="exceeds max_len_seq"):
            policy.predict_action_chunk(make_observation())


# ============================================================================ #
# Training                                                                     #
# ============================================================================ #


class TestXVLATraining:
    """Tests for the training path."""

    def test_training_step_reports_every_loss_component(self) -> None:
        """The action space's loss terms are logged alongside the total."""
        policy = build_policy()
        policy.train()
        loss, loss_dict = policy(make_observation())
        assert loss.requires_grad
        assert loss_dict["loss"].item() == pytest.approx(loss.item(), rel=1e-5)
        assert "joints_loss" in loss_dict

    def test_ee6d_loss_components(self) -> None:
        """A fixed layout reports its position, rotation and gripper terms."""
        policy = build_policy(action_mode="ee6d", max_action_dim=20)
        policy.train()
        _, loss_dict = policy(make_observation(action_dim=20))
        assert {"position_loss", "rotate6D_loss", "gripper_loss", "loss"} == set(loss_dict)

    def test_gradients_reach_the_backbone_and_the_prompts(self) -> None:
        """Backpropagation touches the VLM, the transformer and the soft prompts."""
        policy = build_policy()
        policy.train()
        loss, _ = policy(make_observation())
        loss.backward()

        named = dict(policy.named_parameters())
        assert named["model.transformer.soft_prompt_hub.weight"].grad is not None
        assert named["model.transformer.action_decoder.fc.weight"].grad is not None
        assert any(p.grad is not None for n, p in named.items() if n.startswith("model.vlm."))

    def test_missing_actions_are_reported(self) -> None:
        """Training without action targets fails with a clear message."""
        policy = build_policy()
        policy.train()
        observation = make_observation()
        observation.action = None
        with pytest.raises(ValueError, match="missing the action targets"):
            policy(observation)

    def test_short_action_chunks_are_padded(self) -> None:
        """A dataset with fewer action steps than the chunk is zero-padded."""
        policy = build_policy()
        targets = policy.inner_model.prepare_targets(torch.randn(2, 2, ACTION_DIM))
        assert targets.shape == (2, CHUNK_SIZE, policy.inner_model.dim_action)
        assert torch.count_nonzero(targets[:, 2:]) == 0

    def test_single_step_actions_are_accepted(self) -> None:
        """A ``[B, D]`` action tensor is treated as a one-step chunk."""
        policy = build_policy()
        targets = policy.inner_model.prepare_targets(torch.randn(2, ACTION_DIM))
        assert targets.shape == (2, CHUNK_SIZE, policy.inner_model.dim_action)

    def test_val_loss_is_action_prediction_mse(self) -> None:
        """Validation runs the full denoising loop and compares against the ground truth."""
        policy = build_policy()
        policy.eval()
        loss, loss_dict = policy.compute_val_loss(make_observation())
        assert loss.ndim == 0
        assert not loss.requires_grad
        assert loss_dict["loss"] is loss

    def test_optimizer_param_groups(self) -> None:
        """The VLM trains at a tenth of the base rate; the other groups at their own."""
        policy = build_policy(optimizer_lr=1e-4, optimizer_weight_decay=0.1, optimizer_soft_prompt_lr_scale=0.5)
        policy.trainer = _FakeTrainer()
        configured = policy.configure_optimizers()

        optimizer = configured["optimizer"]
        assert isinstance(optimizer, torch.optim.AdamW)
        groups = {group["name"]: group for group in optimizer.param_groups}
        # LambdaLR applies its warmup factor on construction, so the peak LR lives in initial_lr.
        assert groups["vlm"]["initial_lr"] == pytest.approx(1e-5)
        assert groups["vlm"]["weight_decay"] == pytest.approx(0.01)
        assert groups["soft_prompts"]["initial_lr"] == pytest.approx(5e-5)
        assert groups["other"]["initial_lr"] == pytest.approx(1e-4)
        assert groups["other"]["weight_decay"] == pytest.approx(0.1)
        # The one shared schedule keeps every group's relative rate.
        assert groups["vlm"]["lr"] / groups["other"]["lr"] == pytest.approx(0.1)
        assert configured["lr_scheduler"]["interval"] == "step"

    def test_optimizer_skips_frozen_parameters(self) -> None:
        """Frozen components do not reach the optimizer at all."""
        policy = build_policy(
            freeze_vision_encoder=True,
            freeze_language_encoder=True,
            train_soft_prompts=False,
        )
        policy.trainer = _FakeTrainer()
        optimizer = policy.configure_optimizers()["optimizer"]
        names = {group["name"] for group in optimizer.param_groups}
        assert "soft_prompts" not in names
        assert "other" in names

    def test_freezing_flags(self) -> None:
        """The freezing flags reach the parameters they name."""
        policy = build_policy(freeze_vision_encoder=True, train_policy_transformer=False)
        model = policy.inner_model
        assert not any(p.requires_grad for p in model.vlm.vision_tower.parameters())
        assert not model.transformer.action_decoder.fc.weight.requires_grad
        # Soft prompts stay trainable when the backbone is frozen.
        assert model.transformer.soft_prompt_hub.weight.requires_grad

    def test_gradient_clipping_uses_the_config(self) -> None:
        """The configured clip norm is applied when the trainer supplies none."""
        policy = build_policy(optimizer_grad_clip_norm=3.0)
        recorded: dict[str, Any] = {}
        policy.clip_gradients = lambda optimizer, gradient_clip_val, gradient_clip_algorithm: recorded.update(  # type: ignore[method-assign]
            value=gradient_clip_val,
            algorithm=gradient_clip_algorithm,
        )
        policy.configure_gradient_clipping(torch.optim.AdamW([torch.zeros(1, requires_grad=True)]))
        assert recorded == {"value": 3.0, "algorithm": "norm"}

    def test_set_action_dim_refits_the_auto_space(self) -> None:
        """Finetuning on a narrower dataset re-slices the supervised channels."""
        policy = build_policy()
        model = policy.inner_model
        model.set_action_dim(3)
        assert model.action_space.real_dim == 3
        assert model.dim_action == 8
        policy.eval()
        assert policy.inner_model.predict_action_chunk(
            policy._preprocess(make_observation()),  # noqa: SLF001
        ).shape == (BATCH_SIZE, CHUNK_SIZE, 3)

    def test_set_action_dim_leaves_fixed_spaces_alone(self) -> None:
        """A fixed layout keeps its published width; the mismatch is only reported."""
        policy = build_policy(action_mode="ee6d", max_action_dim=20)
        policy.inner_model.set_action_dim(7)
        assert policy.inner_model.dim_action == 20

    def test_delta_indices(self) -> None:
        """The model asks the dataset for a full chunk of actions and one observation."""
        model = build_policy().inner_model
        assert model.action_delta_indices == list(range(CHUNK_SIZE))
        assert model.observation_delta_indices is None
        assert model.reward_delta_indices is None


class _FakeTrainer:
    """Minimal stand-in for the Lightning trainer's stepping-budget query."""

    estimated_stepping_batches = 100


# ============================================================================ #
# Checkpoint loading                                                           #
# ============================================================================ #


def write_fake_checkpoint(directory: Any, action_dim: int = ACTION_DIM) -> Any:
    """Write an XVLA checkpoint in the published LeRobot layout.

    Args:
        directory: Directory to write into.
        action_dim: Width of the checkpoint's action space.

    Returns:
        The directory that was written to.
    """
    from safetensors.torch import save_file  # noqa: PLC0415

    config = XVLAConfig(**tiny_kwargs())
    payload = config.to_dict()
    payload.update({
        "type": "xvla",
        "device": "cpu",
        "use_amp": False,
        "push_to_hub": True,
        "tokenizer_padding_side": "right",
        "pad_language_to": "max_length",
        "optimizer_soft_prompt_warmup_lr_scale": 0.01,
        "input_features": {
            "observation.images.top": {"type": "VISUAL", "shape": [3, IMAGE_SIZE, IMAGE_SIZE]},
            "observation.images.wrist": {"type": "VISUAL", "shape": [3, IMAGE_SIZE, IMAGE_SIZE]},
            "observation.state": {"type": "STATE", "shape": [STATE_DIM]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [action_dim]}},
        "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "IDENTITY", "ACTION": "IDENTITY"},
    })
    payload.pop("num_image_views", None)
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    model = XVLAModel(config, action_dim=action_dim)
    # LeRobot nests the network one level deeper than Studio does.
    state_dict = {f"model.{k}": v.contiguous() for k, v in model.state_dict().items()}
    # safetensors deduplicates tied tensors on save, so a published checkpoint carries only
    # one alias of the shared token embedding.
    state_dict.pop("model.vlm.language_model.encoder.embed_tokens.weight", None)
    save_file(state_dict, str(directory / "model.safetensors"))
    return directory


class TestXVLACheckpointLoading:
    """Tests for loading checkpoints published in the LeRobot layout."""

    def test_resolve_local_checkpoint(self, tmp_path: Any) -> None:
        """A local directory resolves the published artefacts."""
        from physicalai.policies.xvla.pretrained_utils import resolve_checkpoint  # noqa: PLC0415

        write_fake_checkpoint(tmp_path)
        files = resolve_checkpoint(tmp_path)
        assert files.config_file.name == "config.json"
        assert files.weights_file.name == "model.safetensors"
        assert files.processor_files == ()

    def test_load_config_drops_lerobot_only_keys(self, tmp_path: Any) -> None:
        """Feature specs and hub metadata do not leak into the Studio config."""
        from physicalai.policies.xvla.pretrained_utils import load_config  # noqa: PLC0415

        write_fake_checkpoint(tmp_path)
        config = load_config(tmp_path / "config.json", {"dtype": "bfloat16"})
        assert config.depth == 1
        assert config.dtype == "bfloat16"
        assert not hasattr(config, "input_features")
        assert not hasattr(config, "tokenizer_padding_side")

    def test_load_config_derives_camera_count(self, tmp_path: Any) -> None:
        """A checkpoint keeps the camera count it declares in its input features."""
        from physicalai.policies.xvla.pretrained_utils import load_config  # noqa: PLC0415

        write_fake_checkpoint(tmp_path)
        assert load_config(tmp_path / "config.json").num_image_views == 2

    def test_load_config_ignores_none_overrides(self, tmp_path: Any) -> None:
        """``None`` overrides mean "unset" and keep the checkpoint's value."""
        from physicalai.policies.xvla.pretrained_utils import load_config  # noqa: PLC0415

        write_fake_checkpoint(tmp_path)
        assert load_config(tmp_path / "config.json", {"dtype": None}).dtype == "float32"

    def test_read_action_dim(self, tmp_path: Any) -> None:
        """The checkpoint's action width is read from its output features."""
        from physicalai.policies.xvla.pretrained_utils import read_action_dim  # noqa: PLC0415

        write_fake_checkpoint(tmp_path, action_dim=5)
        assert read_action_dim(tmp_path / "config.json") == 5

    def test_end_to_end_pretrained_load(self, tmp_path: Any) -> None:
        """A published checkpoint builds a ready-to-run policy in one call."""
        write_fake_checkpoint(tmp_path, action_dim=3)

        policy = XVLA(pretrained_name_or_path=tmp_path)
        assert policy.model is not None
        assert policy.config.depth == 1
        assert policy.inner_model.action_space.real_dim == 3

        policy._preprocessor._tokenizer = FakeTokenizer()  # noqa: SLF001
        policy.eval()
        assert policy.select_action(make_observation(batch_size=1)).shape == (1, 3)

    def test_pretrained_weights_are_actually_loaded(self, tmp_path: Any) -> None:
        """The ``model.`` prefix is stripped and the weights land in the network."""
        from safetensors.torch import load_file  # noqa: PLC0415

        write_fake_checkpoint(tmp_path)
        policy = XVLA(pretrained_name_or_path=tmp_path)
        published = load_file(str(tmp_path / "model.safetensors"))
        torch.testing.assert_close(
            policy.inner_model.transformer.pos_emb,
            published["model.transformer.pos_emb"],
        )

        # safetensors dropped one alias of the tied token embedding on save; it is restored.
        assert "model.vlm.language_model.encoder.embed_tokens.weight" not in published
        torch.testing.assert_close(
            policy.inner_model.vlm.language_model.encoder.embed_tokens.weight,
            published["model.vlm.language_model.shared.weight"],
        )

    def test_pretrained_load_honours_explicit_overrides(self, tmp_path: Any) -> None:
        """Caller arguments win over the checkpoint's published values."""
        write_fake_checkpoint(tmp_path)
        policy = XVLA(pretrained_name_or_path=tmp_path, num_denoising_steps=4, optimizer_lr=7e-6)
        assert policy.config.num_denoising_steps == 4
        assert policy.config.optimizer_lr == 7e-6
        # Untouched fields keep the checkpoint's values.
        assert policy.config.depth == 1

    def test_vendored_florence_keys_are_remapped(self) -> None:
        """The legacy Florence-2 module tree is mapped onto the native layout."""
        from physicalai.policies.xvla.pretrained_utils import (  # noqa: PLC0415
            is_vendored_florence_state_dict,
            remap_vendored_florence_state_dict,
        )

        state_dict = {
            "vlm.image_projection": torch.randn(4, 8),
            "vlm.image_proj_norm.weight": torch.randn(8),
            "vlm.language_model.model.encoder.layers.0.fc1.weight": torch.randn(4, 4),
            "vlm.vision_tower.convs.0.proj.weight": torch.randn(4, 3, 7, 7),
            "vlm.vision_tower.blocks.0.0.spatial_block.ffn.fn.net.fc1.weight": torch.randn(4, 4),
            "vlm.language_model.final_logits_bias": torch.zeros(1, 8),
            "transformer.pos_emb": torch.randn(1, 8, 4),
        }
        assert is_vendored_florence_state_dict(state_dict)

        remapped = remap_vendored_florence_state_dict(state_dict)
        assert "vlm.multi_modal_projector.image_projection.weight" in remapped
        # The vendored parameter is used as `x @ p`, so the native Linear holds its transpose.
        assert remapped["vlm.multi_modal_projector.image_projection.weight"].shape == (8, 4)
        assert "vlm.multi_modal_projector.image_proj_norm.weight" in remapped
        assert "vlm.language_model.encoder.layers.0.fc1.weight" in remapped
        assert "vlm.vision_tower.convs.0.conv.weight" in remapped
        assert "vlm.vision_tower.blocks.0.0.spatial_block.ffn.fc1.weight" in remapped
        # The generation-only buffer has no counterpart in BartModel.
        assert not any("final_logits_bias" in key for key in remapped)
        # Keys outside the VLM pass through untouched.
        assert "transformer.pos_emb" in remapped
