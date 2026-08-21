# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the VLA-JEPA policy.

Fast, self-contained tests with no external dependencies (no HuggingFace model downloads), so the
Qwen3-VL backbone is never built here. The ported submodules (action head, world model) and the
pre/postprocessing pipeline are exercised directly on synthetic tensors.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest
import torch
from physicalai.config import Config
from physicalai.policies import get_policy
from physicalai.policies.vla_jepa import VLAJEPA, VLAJEPAConfig
from physicalai.policies.vla_jepa.components.action_head import VLAJEPAActionHead
from physicalai.policies.vla_jepa.preprocessor import (
    RelativeActionTransform,
    make_vla_jepa_preprocessors,
    prepare_images,
)
from physicalai.policies.vla_jepa.components.world_model import ActionConditionedVideoPredictor

if TYPE_CHECKING:
    from pathlib import Path

ACTION_DIM = 3
STATE_DIM = 4
CHUNK_SIZE = 4


def tiny_config(**overrides: object) -> VLAJEPAConfig:
    """Build a config small enough to instantiate the ported heads in a unit test.

    Args:
        **overrides: Fields to override on top of the tiny defaults.

    Returns:
        A VLAJEPAConfig sized for CPU tests.
    """
    kwargs: dict[str, object] = {
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": CHUNK_SIZE,
        "action_dim": ACTION_DIM,
        "state_dim": STATE_DIM,
        "action_model_type": "DiT-test",
        "action_hidden_size": 16,
        "action_num_layers": 2,
        "action_dropout": 0.0,
        "num_embodied_action_tokens_per_instruction": 4,
        "num_inference_timesteps": 2,
        "repeated_diffusion_steps": 2,
        "action_max_seq_len": 32,
        "enable_world_model": False,
    }
    kwargs.update(overrides)
    return VLAJEPAConfig(**kwargs)  # type: ignore[arg-type]


def tiny_stats() -> dict[str, dict[str, object]]:
    """Build dataset statistics for a 3-dim action and 4-dim state.

    Returns:
        Stats dict in the format produced by ``Dataset.stats``.
    """
    return {
        "observation.state": {
            "name": "state",
            "type": "STATE",
            "shape": (STATE_DIM,),
            "mean": [0.0] * STATE_DIM,
            "std": [2.0] * STATE_DIM,
            "min": [-1.0] * STATE_DIM,
            "max": [1.0] * STATE_DIM,
        },
        "action": {
            "name": "action",
            "type": "ACTION",
            "shape": (ACTION_DIM,),
            "mean": [0.0] * ACTION_DIM,
            "std": [1.0] * ACTION_DIM,
            "min": [-2.0] * ACTION_DIM,
            "max": [2.0] * ACTION_DIM,
        },
    }


# ============================================================================ #
# Configuration Tests                                                          #
# ============================================================================ #


class TestVLAJEPAConfig:
    """Tests for the VLAJEPAConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = VLAJEPAConfig()
        assert config.qwen_model_name == "Qwen/Qwen3-VL-2B-Instruct"
        assert config.jepa_encoder_name == "facebook/vjepa2-vitl-fpc64-256"
        assert config.action_model_type == "DiT-B"
        assert config.enable_world_model is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = VLAJEPAConfig(chunk_size=16, n_action_steps=8, optimizer_lr=2e-4, action_num_layers=4)
        assert config.chunk_size == 16
        assert config.n_action_steps == 8
        assert config.optimizer_lr == 2e-4
        assert config.action_num_layers == 4

    def test_n_action_steps_validation(self) -> None:
        """Test n_action_steps cannot exceed chunk_size."""
        with pytest.raises(ValueError, match="chunk size is the upper bound"):
            VLAJEPAConfig(chunk_size=4, n_action_steps=8)

    def test_num_video_frames_validation(self) -> None:
        """Test the world model needs both a context and a ground-truth temporal position."""
        with pytest.raises(ValueError, match="must be >= 2"):
            VLAJEPAConfig(num_video_frames=2, jepa_tubelet_size=2)

    def test_freeze_qwen_disables_world_model(self) -> None:
        """Test freezing the backbone turns the world model off: no gradient would reach it."""
        config = VLAJEPAConfig(freeze_qwen=True, enable_world_model=True)
        assert config.enable_world_model is False

    def test_resolved_gripper_dim_from_feature_names(self) -> None:
        """Test the gripper index is resolved from dataset feature names."""
        config = VLAJEPAConfig(action_feature_names=["x", "y", "gripper_pos"], gripper_dim=6)
        assert config.resolved_gripper_dim == 2

    def test_resolved_gripper_dim_falls_back(self) -> None:
        """Test the raw gripper_dim is used when no feature names are known."""
        assert VLAJEPAConfig(gripper_dim=5).resolved_gripper_dim == 5

    def test_num_world_model_views(self) -> None:
        """Test the view count falls back to the tubelet size the checkpoints encode."""
        assert VLAJEPAConfig(world_model_num_views=None, jepa_tubelet_size=2).num_world_model_views == 2
        assert VLAJEPAConfig(world_model_num_views=3).num_world_model_views == 3

    def test_inheritance_and_serialization(self) -> None:
        """Test the config inherits from base Config and round-trips."""
        config = VLAJEPAConfig(chunk_size=16, optimizer_lr=2e-4)
        assert isinstance(config, Config)

        config_dict = config.to_dict()
        assert config_dict["chunk_size"] == 16

        restored = VLAJEPAConfig.from_dict(config_dict)
        assert restored.chunk_size == 16
        assert restored.optimizer_lr == 2e-4

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Test published config.json keys the dataclass no longer declares are ignored."""
        payload = {**VLAJEPAConfig().to_dict(), "num_target_vision_tokens": 32}
        assert VLAJEPAConfig.from_dict(payload, strict=False).chunk_size == 7


class TestDeltaIndices:
    """Tests for the dataset delta indices the datamodule is reformatted with."""

    def test_observation_delta_indices_without_world_model(self) -> None:
        """Test only the current frame is loaded when the world model is off."""
        assert tiny_config().observation_delta_indices == [0]

    def test_observation_delta_indices_with_world_model(self) -> None:
        """Test the full frame window is loaded when the world model is on."""
        config = VLAJEPAConfig(chunk_size=8, n_action_steps=8, num_video_frames=8)
        assert config.observation_delta_indices == list(range(8))

    def test_observation_delta_indices_stride_long_chunks(self) -> None:
        """Test frames are strided across the chunk rather than clustered at its start."""
        config = VLAJEPAConfig(chunk_size=20, n_action_steps=20, num_video_frames=4)
        assert config.observation_delta_indices == [0, 6, 12, 18]

    def test_action_delta_indices(self) -> None:
        """Test one action index per chunk step."""
        assert tiny_config().action_delta_indices == list(range(CHUNK_SIZE))


# ============================================================================ #
# Action Head Tests                                                            #
# ============================================================================ #


class TestActionHead:
    """Tests for the flow-matching DiT action head."""

    @staticmethod
    def build_head() -> tuple[VLAJEPAActionHead, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a tiny head with matching conditioning, action and state tensors.

        Returns:
            Tuple of (head, conditioning tokens, actions, state).
        """
        torch.manual_seed(0)
        cross_attention_dim = 8
        head = VLAJEPAActionHead(tiny_config(), cross_attention_dim=cross_attention_dim)
        batch = 2
        conditioning = torch.randn(batch, 4, cross_attention_dim)
        actions = torch.randn(batch, CHUNK_SIZE, ACTION_DIM)
        state = torch.randn(batch, 1, STATE_DIM)
        return head, conditioning, actions, state

    def test_forward_returns_scalar_loss(self) -> None:
        """Test the training forward returns a scalar loss with a gradient."""
        head, conditioning, actions, state = self.build_head()
        loss = head(conditioning, actions, state)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_forward_per_sample_reduction(self) -> None:
        """Test reduction='none' returns one loss per sample."""
        head, conditioning, actions, state = self.build_head()
        loss = head(conditioning, actions, state, reduction="none")
        assert loss.shape == (actions.shape[0],)

    def test_forward_masks_padded_steps(self) -> None:
        """Test fully padded action chunks contribute nothing to the loss."""
        head, conditioning, actions, state = self.build_head()
        action_is_pad = torch.ones(actions.shape[:2], dtype=torch.bool)
        loss = head(conditioning, actions, state, action_is_pad)
        assert loss.item() == pytest.approx(0.0)

    def test_predict_action_shape(self) -> None:
        """Test inference returns a full action chunk."""
        head, conditioning, _, state = self.build_head()
        actions = head.predict_action(conditioning, state)
        assert actions.shape == (conditioning.shape[0], CHUNK_SIZE, ACTION_DIM)

    def test_predict_action_without_state(self) -> None:
        """Test the head works on datasets without a robot state."""
        head, conditioning, _, _ = self.build_head()
        assert head.predict_action(conditioning).shape == (conditioning.shape[0], CHUNK_SIZE, ACTION_DIM)


# ============================================================================ #
# World Model Tests                                                            #
# ============================================================================ #


class TestWorldModel:
    """Tests for the action-conditioned V-JEPA predictor."""

    def test_predicts_next_frame_embeddings(self) -> None:
        """Test the predictor emits one embedding per input frame token."""
        torch.manual_seed(0)
        batch, num_frames, tokens_per_step = 2, 2, 2
        embed_dim, predictor_embed_dim, action_embed_dim = 16, 24, 32
        predictor = ActionConditionedVideoPredictor(
            num_frames=num_frames,
            img_size=(32, 32),
            patch_size=16,
            tubelet_size=1,
            embed_dim=embed_dim,
            action_embed_dim=action_embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=1,
            num_heads=2,
            mlp_ratio=2.0,
            num_action_tokens_per_step=tokens_per_step,
        )
        grid_tokens = predictor.grid_height * predictor.grid_width
        frame_tokens = torch.randn(batch, num_frames * grid_tokens, embed_dim)
        action_tokens = torch.randn(batch, num_frames * tokens_per_step, action_embed_dim)

        predicted = predictor(frame_tokens, action_tokens)

        assert predicted.shape == frame_tokens.shape


# ============================================================================ #
# Preprocessing Tests                                                          #
# ============================================================================ #


class TestImagePrep:
    """Tests for the shared image preparation."""

    def test_expands_grayscale_and_resizes(self) -> None:
        """Test a single channel is expanded to three and the image is resized."""
        image = torch.rand(2, 1, 8, 8)
        prepared = prepare_images(image, resize_to=(4, 4))
        assert prepared.shape == (2, 3, 4, 4)

    def test_is_idempotent(self) -> None:
        """Test re-running the prep is a no-op, so the model can guard with the same call."""
        image = torch.rand(2, 3, 4, 4)
        once = prepare_images(image, resize_to=(4, 4))
        twice = prepare_images(once, resize_to=(4, 4))
        assert torch.equal(once, twice)

    def test_handles_frame_windows(self) -> None:
        """Test video windows keep their time dimension."""
        prepared = prepare_images(torch.rand(2, 5, 1, 8, 8), resize_to=(4, 4))
        assert prepared.shape == (2, 5, 3, 4, 4)


class TestPreprocessors:
    """Tests for the pre/postprocessor pair."""

    def test_normalization_round_trip(self) -> None:
        """Test actions survive normalize -> denormalize unchanged."""
        pre, post = make_vla_jepa_preprocessors(tiny_config(), tiny_stats())
        action = torch.tensor([[[0.5, -1.5, 2.0], [0.0, 1.0, -2.0]]])
        batch = {"state": torch.zeros(1, STATE_DIM), "action": action.clone()}

        processed = pre(batch)
        assert torch.allclose(processed["action"], action / 2.0)

        restored = post({"action": processed["action"]})["action"]
        assert torch.allclose(restored, action, atol=1e-5)

    def test_state_is_normalized_with_mean_std(self) -> None:
        """Test the state uses mean/std normalization."""
        pre, _ = make_vla_jepa_preprocessors(tiny_config(), tiny_stats())
        processed = pre({"state": torch.full((1, STATE_DIM), 4.0)})
        assert torch.allclose(processed["state"], torch.full((1, STATE_DIM), 2.0), atol=1e-5)

    def test_clip_is_refused_under_mean_std(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test clipping to [-1, 1] is refused when it would truncate at 1 sigma."""
        config = tiny_config(action_normalization="MEAN_STD", clip_normalized_actions=True)
        with caplog.at_level(logging.WARNING):
            _, post = make_vla_jepa_preprocessors(config, tiny_stats())
        assert post.clip_normalized_actions is False
        assert "clip_normalized_actions" in caplog.text

    def test_clip_applies_under_min_max(self) -> None:
        """Test out-of-range normalized actions are clipped before denormalization."""
        _, post = make_vla_jepa_preprocessors(tiny_config(clip_normalized_actions=True), tiny_stats())
        restored = post({"action": torch.tensor([[[5.0, -5.0, 0.0]]])})["action"]
        assert torch.allclose(restored, torch.tensor([[[2.0, -2.0, 0.0]]]), atol=1e-5)

    def test_missing_stats_for_normalization_raise(self) -> None:
        """Test MIN_MAX actions without min/max statistics fail with a clear message."""
        stats = tiny_stats()
        del stats["action"]["min"]
        del stats["action"]["max"]
        with pytest.raises(ValueError, match="min/max statistics"):
            make_vla_jepa_preprocessors(tiny_config(), stats)

    def test_gripper_pre_snap_and_binarize(self) -> None:
        """Test the LIBERO gripper steps snap the gripper to a binary command."""
        config = tiny_config(
            pre_snap_gripper_action=True,
            binarize_gripper_action=True,
            gripper_dim=2,
            clip_normalized_actions=False,
        )
        _, post = make_vla_jepa_preprocessors(config, tiny_stats())
        # dim 2 = 0.8 -> snapped to 1.0 -> denormalized to 2.0 -> binarized to -1.0.
        restored = post({"action": torch.tensor([[[0.0, 0.0, 0.8]]])})["action"]
        assert restored[0, 0, 2].item() == pytest.approx(-1.0)

    def test_relative_actions_round_trip(self) -> None:
        """Test the relative conversion is exactly reversed by the postprocessor."""
        config = tiny_config(
            use_relative_actions=True,
            relative_exclude_joints=[],
            state_normalization="IDENTITY",
            action_normalization="IDENTITY",
            clip_normalized_actions=False,
        )
        pre, post = make_vla_jepa_preprocessors(config, tiny_stats())
        action = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        state = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

        processed = pre({"state": state, "action": action.clone()})
        assert torch.allclose(processed["action"][0, 0], torch.tensor([0.0, 1.0, 2.0]))

        restored = post({"action": processed["action"]})["action"]
        assert torch.allclose(restored, action)

    def test_relative_actions_respect_excluded_joints(self) -> None:
        """Test excluded joints stay absolute."""
        transform = RelativeActionTransform(
            enabled=True,
            exclude_joints=["gripper"],
            action_names=["x", "y", "gripper"],
        )
        action = torch.tensor([[[1.0, 2.0, 3.0]]])
        state = torch.tensor([[1.0, 1.0, 1.0]])
        relative = transform.to_relative(action, state)
        assert torch.allclose(relative[0, 0], torch.tensor([0.0, 1.0, 3.0]))

    def test_absolute_without_cached_state_raises(self) -> None:
        """Test reversing the conversion before the preprocessor ran is an error."""
        transform = RelativeActionTransform(enabled=True)
        with pytest.raises(RuntimeError, match="no state has been cached"):
            transform.to_absolute(torch.zeros(1, 1, ACTION_DIM))


# ============================================================================ #
# Policy Tests                                                                 #
# ============================================================================ #


class TestVLAJEPAPolicy:
    """Tests for the Lightning policy wrapper."""

    def test_lazy_construction_defers_model(self) -> None:
        """Test the model is only built once dataset stats are known."""
        policy = VLAJEPA(chunk_size=8, n_action_steps=4, enable_world_model=False)
        assert policy.model is None
        assert policy.config.chunk_size == 8
        assert policy.config.n_action_steps == 4

    def test_config_is_saved_to_hparams(self) -> None:
        """Test the resolved config is stored for checkpoint restoration."""
        policy = VLAJEPA(chunk_size=8, n_action_steps=8, optimizer_lr=3e-4)
        assert policy.hparams["config"]["chunk_size"] == 8
        assert policy.hparams["optimizer_lr"] == 3e-4

    def test_predict_before_setup_raises(self) -> None:
        """Test inference before the model exists fails loudly."""
        policy = VLAJEPA(enable_world_model=False)
        with pytest.raises(ValueError, match="Model is not initialized"):
            policy.predict_action_chunk(None)  # type: ignore[arg-type]

    def test_get_policy_factory(self) -> None:
        """Test the policy is reachable through the physicalai factory."""
        policy = get_policy("vla_jepa", enable_world_model=False)
        assert isinstance(policy, VLAJEPA)

    def test_action_queue_uses_n_action_steps(self) -> None:
        """Test the base-class action queue is sized from n_action_steps."""
        policy = VLAJEPA(chunk_size=8, n_action_steps=3, enable_world_model=False)
        chunk = torch.arange(8 * 2, dtype=torch.float32).reshape(1, 8, 2)
        first = policy._queue_actions(chunk)  # noqa: SLF001
        assert torch.equal(first, chunk[:, 0])
        assert len(policy._action_queue) == 2  # noqa: SLF001


# ============================================================================ #
# Pretrained config merge                                                      #
# ============================================================================ #


class TestPretrainedConfigMerge:
    """Tests for how `_from_hf` merges a published config with constructor arguments.

    A published checkpoint records inference-critical settings - input resolution and gripper
    handling - that differ from the from-scratch defaults. Letting those defaults through silently
    reconfigured the LIBERO checkpoint's preprocessing and drove its success rate to zero, so the
    merge must ignore every argument the caller did not pass explicitly.
    """

    @staticmethod
    def _checkpoint(tmp_path: Path) -> Path:
        """Write a minimal published-checkpoint directory.

        Args:
            tmp_path: Directory the checkpoint files are written into.

        Returns:
            The checkpoint directory, ready to pass to `_from_hf`.
        """
        config = {
            "type": "vla_jepa",
            "input_features": {"observation.state": {"type": "STATE", "shape": [8]}},
            "output_features": {"action": {"type": "ACTION", "shape": [7]}},
            "chunk_size": 7,
            "n_action_steps": 7,
            "resize_images_to": [224, 224],
            "binarize_gripper_action": True,
            "pre_snap_gripper_action": True,
            "optimizer_lr": 1e-4,
        }
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (tmp_path / "model.safetensors").touch()
        return tmp_path

    def test_defaults_do_not_override_checkpoint(self, tmp_path: Path) -> None:
        """Test unpassed constructor arguments leave the checkpoint's values intact."""
        config, _, _ = VLAJEPA._from_hf(  # noqa: SLF001
            self._checkpoint(tmp_path),
            {"resize_images_to": None, "binarize_gripper_action": False, "pre_snap_gripper_action": False},
            frozenset({"pretrained_name_or_path"}),
        )
        assert config.resize_images_to == (224, 224)
        assert config.binarize_gripper_action is True
        assert config.pre_snap_gripper_action is True

    def test_explicit_argument_overrides_checkpoint(self, tmp_path: Path) -> None:
        """Test an explicitly passed argument still wins, even when it equals the default."""
        config, _, _ = VLAJEPA._from_hf(  # noqa: SLF001
            self._checkpoint(tmp_path),
            {"binarize_gripper_action": False, "optimizer_lr": 3e-5},
            frozenset({"binarize_gripper_action", "optimizer_lr"}),
        )
        assert config.binarize_gripper_action is False
        assert config.optimizer_lr == pytest.approx(3e-5)

    def test_architecture_field_cannot_be_overridden(self, tmp_path: Path) -> None:
        """Test shape-baked fields keep the checkpoint's value even when passed explicitly."""
        config, _, _ = VLAJEPA._from_hf(  # noqa: SLF001
            self._checkpoint(tmp_path),
            {"chunk_size": 99},
            frozenset({"chunk_size"}),
        )
        assert config.chunk_size == 7

    def test_explicit_args_are_tracked(self) -> None:
        """Test the constructor records exactly which arguments the caller passed."""
        policy = VLAJEPA(chunk_size=8, n_action_steps=4, enable_world_model=False)
        assert policy._explicit_args == frozenset({"chunk_size", "n_action_steps", "enable_world_model"})  # noqa: SLF001


# ============================================================================ #
# Export                                                                       #
# ============================================================================ #


class TestExport:
    """Tests for the ExportablePolicyMixin surface.

    Uses a lightweight stub instead of constructing the full model, so the Qwen3-VL backbone is
    never downloaded.
    """

    @staticmethod
    def _stub(dataset_stats: dict, **config_overrides: object) -> VLAJEPA:
        """Build a stand-in exposing only what the export properties read."""

        class _Stub:
            def __init__(self) -> None:
                self._dataset_stats = dataset_stats
                self.model = torch.nn.Linear(1, 1)
                self.config = tiny_config(**config_overrides)

        return _Stub()  # type: ignore[return-value]

    @staticmethod
    def _stats() -> dict:
        """Dataset stats in the shape the policy stores them."""
        return {
            "observation.state": {"name": "observation.state", "shape": (STATE_DIM,), "type": "STATE"},
            # Real datasets carry the dotted feature path in `name`; the schema must strip the
            # `observation.`/`images.` prefixes rather than nest them (`images.images.top`).
            "observation.images.top": {"name": "images.top", "shape": (3, 480, 640), "type": "VISUAL"},
            "observation.images.wrist": {"name": "images.wrist", "shape": (3, 480, 640), "type": "VISUAL"},
            "action": {"name": "action", "shape": (ACTION_DIM,), "type": "ACTION"},
        }

    def test_only_torch_backend_is_supported(self) -> None:
        """Test tracing backends are not advertised."""
        from physicalai.export import ExportBackend

        assert VLAJEPA.get_supported_export_backends() == [ExportBackend.TORCH]

    def test_schemas_are_none_before_setup(self) -> None:
        """Test the schemas stay None until the model and stats exist."""
        policy = VLAJEPA(enable_world_model=False)
        assert policy.inputs_schema is None
        assert policy.outputs_schema is None

    def test_inputs_schema_covers_state_images_and_task(self) -> None:
        """Test every observation modality is described once."""
        from physicalai.data.observation import IMAGES, STATE, TASK

        schema = VLAJEPA.inputs_schema.fget(self._stub(self._stats()))  # type: ignore[attr-defined]
        names = [feature.name for feature in schema]
        assert names == [STATE, f"{IMAGES}.top", f"{IMAGES}.wrist", TASK]

    def test_inputs_schema_reports_the_resized_resolution(self) -> None:
        """Test the visual features advertise the resolution the backbone actually sees."""
        stub = self._stub(self._stats(), resize_images_to=(224, 224))
        schema = VLAJEPA.inputs_schema.fget(stub)  # type: ignore[attr-defined]
        visual = [feature for feature in schema if "images" in feature.name]
        assert all(feature.shape == (3, 224, 224) for feature in visual)

    def test_outputs_schema_is_the_action_chunk(self) -> None:
        """Test the output feature carries the full chunk horizon."""
        from physicalai.data.observation import ACTION

        schema = VLAJEPA.outputs_schema.fget(self._stub(self._stats()))  # type: ignore[attr-defined]
        assert len(schema) == 1
        assert schema[0].name == ACTION
        assert schema[0].shape == (CHUNK_SIZE, ACTION_DIM)

    def test_sample_input_is_built_from_the_schema(self) -> None:
        """Test the inherited sample_input turns the schema into traceable tensors."""
        from physicalai.data.observation import STATE, TASK

        stub = self._stub(self._stats())
        stub.inputs_schema = VLAJEPA.inputs_schema.fget(stub)  # type: ignore[attr-defined]
        sample = VLAJEPA.sample_input.fget(stub)  # type: ignore[attr-defined]
        assert sample[STATE].shape == (1, STATE_DIM)
        assert isinstance(sample[TASK], str)

    def test_extra_export_args_trims_the_chunk_when_needed(self) -> None:
        """Test the torch manifest carries a trimmer only when the horizons differ."""
        trimmed = VLAJEPA.extra_export_args.fget(  # type: ignore[attr-defined]
            self._stub(self._stats(), n_action_steps=CHUNK_SIZE - 1),
        )
        assert [spec.type for spec in trimmed["torch"].postprocessors_specs] == ["action_chunk_trimmer"]

        untrimmed = VLAJEPA.extra_export_args.fget(self._stub(self._stats()))  # type: ignore[attr-defined]
        assert untrimmed["torch"].postprocessors_specs == []
        assert [spec.type for spec in untrimmed["torch"].preprocessors_specs] == ["to_float_tensor"]

    def test_extra_export_args_requires_dataset_stats(self) -> None:
        """Test export fails loudly when normalization statistics are missing."""
        with pytest.raises(ValueError, match="Dataset stats are required"):
            VLAJEPA.extra_export_args.fget(self._stub(None))  # type: ignore[attr-defined,arg-type]
