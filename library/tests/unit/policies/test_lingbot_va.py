# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LingBot-VA policy.

Fast and self-contained: the frozen VAE / UMT5 stack (~20 GB) is replaced with tiny stubs
so the autoregressive streaming path, the action queue and the (de)normalization can be
exercised without any download.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from physicalai.config import Config
from physicalai.inference.data import InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import Manifest
from torch import nn

from physicalai.data import Observation
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.policies import get_policy
from physicalai.policies.lingbot_va import LingBotVA, LingBotVAConfig, LingBotVAModel
from physicalai.policies.lingbot_va.preprocessor import (
    camera_basename,
    make_lingbot_va_preprocessors,
    resolve_camera_keys,
)

pytest.importorskip("diffusers")

LATENT_CHANNELS = 4
VAE_DOWNSAMPLE = 16
VAE_TEMPORAL_DOWNSAMPLE = 4


def tiny_config(**overrides: Any) -> LingBotVAConfig:
    """Build a miniature LingBot-VA config that still exercises every code path.

    Args:
        **overrides: Fields to override on top of the miniature defaults.

    Returns:
        The miniature configuration.
    """
    kwargs: dict[str, Any] = {
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "in_channels": LATENT_CHANNELS,
        "out_channels": LATENT_CHANNELS,
        "action_dim": 8,
        "text_dim": 32,
        "freq_dim": 16,
        "ffn_dim": 32,
        "num_layers": 1,
        "height": 32,
        "width": 32,
        "frame_chunk_size": 2,
        "action_per_frame": 4,
        "attn_window": 8,
        "num_inference_steps": 2,
        "action_num_inference_steps": 2,
        "max_sequence_length": 8,
        "used_action_channel_ids": (0, 1, 2),
        "guidance_scale": 1.0,
        "action_guidance_scale": 1.0,
        "obs_cam_keys": ("observation.images.image", "observation.images.image2"),
        "dtype": "float32",
    }
    kwargs.update(overrides)
    return LingBotVAConfig(**kwargs)


class _FakeVAEConfig:
    """Minimal stand-in for the diffusers ``AutoencoderKLWan`` config."""

    latents_mean = [0.0] * LATENT_CHANNELS
    latents_std = [1.0] * LATENT_CHANNELS
    z_dim = LATENT_CHANNELS
    patch_size = 2


class _FakeLatentDist:
    """Stand-in for a diffusers latent distribution."""

    def __init__(self, sample: torch.Tensor) -> None:
        self._sample = sample

    def mode(self) -> torch.Tensor:
        """Return the distribution mode.

        Returns:
            The wrapped tensor.
        """
        return self._sample


class _FakeEncodeOutput:
    """Stand-in for the output of ``AutoencoderKLWan.encode``."""

    def __init__(self, sample: torch.Tensor) -> None:
        self.latent_dist = _FakeLatentDist(sample)


class _FakeVAE(nn.Module):
    """Deterministic tiny stand-in for the frozen Wan2.2 VAE."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeVAEConfig()
        self.dtype = torch.float32
        self._marker = nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode(self, x: torch.Tensor) -> _FakeEncodeOutput:
        """Encode a clip into latents.

        Args:
            x: Video clip of shape ``[B, C, F, H, W]``.

        Returns:
            A fake encoder output holding ``[B, z_dim, F // 4, H // 16, W // 16]``.
        """
        return _FakeEncodeOutput(self._latents(x))

    def decode(self, latents: torch.Tensor, return_dict: bool = False) -> tuple[torch.Tensor]:  # noqa: FBT001, FBT002
        """Decode latents back into a video.

        Args:
            latents: Latents of shape ``[B, z_dim, F, h, w]``.
            return_dict: Ignored; present for signature parity.

        Returns:
            A one-tuple holding the decoded video.
        """
        del return_dict
        batch, _, frames, height, width = latents.shape
        video = torch.zeros(
            batch,
            3,
            frames * VAE_TEMPORAL_DOWNSAMPLE,
            height * VAE_DOWNSAMPLE,
            width * VAE_DOWNSAMPLE,
        )
        return (video,)

    @staticmethod
    def _latents(x: torch.Tensor) -> torch.Tensor:
        """Downsample a clip to latent resolution.

        Args:
            x: Video clip of shape ``[B, C, F, H, W]``.

        Returns:
            Latents of shape ``[B, z_dim, F // 4, H // 16, W // 16]``.
        """
        batch, _, frames, height, width = x.shape
        return torch.zeros(
            batch,
            LATENT_CHANNELS,
            max(1, frames // VAE_TEMPORAL_DOWNSAMPLE),
            height // VAE_DOWNSAMPLE,
            width // VAE_DOWNSAMPLE,
        )


class _FakeStreamingVAE:
    """Stand-in for the causal streaming VAE encoder wrapper."""

    def __init__(self) -> None:
        self.calls = 0

    def clear_cache(self) -> None:
        """Reset the (nonexistent) causal cache."""
        self.calls = 0

    def encode_chunk(self, x_chunk: torch.Tensor) -> torch.Tensor:
        """Encode one chunk into a mu/logvar stack.

        Args:
            x_chunk: Video chunk of shape ``[B, C, F, H, W]``.

        Returns:
            Encoder output of shape ``[B, 2 * z_dim, F // 4, H // 16, W // 16]``.
        """
        self.calls += 1
        latents = _FakeVAE._latents(x_chunk)  # noqa: SLF001
        return torch.cat([latents, torch.zeros_like(latents)], dim=1)


class _FakeTokenizerOutput:
    """Stand-in for a tokenizer's batch encoding."""

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class _FakeTokenizer:
    """Stand-in for the UMT5 tokenizer."""

    def __call__(self, prompts: list[str], max_length: int = 8, **kwargs: Any) -> _FakeTokenizerOutput:
        """Tokenize prompts into fixed-length ids.

        Args:
            prompts: Prompt strings.
            max_length: Padded length.
            **kwargs: Ignored tokenizer options.

        Returns:
            A fake batch encoding.
        """
        del kwargs
        batch = len(prompts)
        ids = torch.zeros(batch, max_length, dtype=torch.long)
        mask = torch.ones(batch, max_length, dtype=torch.long)
        return _FakeTokenizerOutput(ids, mask)


class _FakeTextEncoderOutput:
    """Stand-in for a transformers encoder output."""

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _FakeTextEncoder(nn.Module):
    """Stand-in for the frozen UMT5-XXL encoder."""

    def __init__(self, text_dim: int) -> None:
        super().__init__()
        self.text_dim = text_dim
        self._marker = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> _FakeTextEncoderOutput:
        """Encode token ids into hidden states.

        Args:
            input_ids: Token ids of shape ``[B, L]``.
            attention_mask: Attention mask of shape ``[B, L]``.

        Returns:
            A fake encoder output of shape ``[B, L, text_dim]``.
        """
        del attention_mask
        return _FakeTextEncoderOutput(torch.zeros(*input_ids.shape, self.text_dim))


def install_fake_frozen_modules(model: LingBotVAModel) -> None:
    """Replace the model's frozen sub-models with tiny deterministic stubs.

    Args:
        model: The model to patch in place.
    """
    model._frozen = {  # noqa: SLF001
        "vae": _FakeVAE().eval(),
        "streaming_vae": _FakeStreamingVAE(),
        "text_encoder": _FakeTextEncoder(model.config.text_dim).eval(),
        "tokenizer": _FakeTokenizer(),
    }


def make_observation(config: LingBotVAConfig, batch_size: int = 1) -> Observation:
    """Build a synthetic observation matching the configured cameras.

    Args:
        config: The policy configuration.
        batch_size: Leading batch dimension.

    Returns:
        An ``Observation`` with the configured cameras and a task string.
    """
    images = {
        camera_basename(key): torch.rand(batch_size, 3, config.height, config.width)
        for key in config.obs_cam_keys
    }
    return Observation(images=images, task=["pick up the block"] * batch_size)


def build_policy(**overrides: Any) -> LingBotVA:
    """Build a tiny policy with fake frozen modules and identity action statistics.

    Args:
        **overrides: Config overrides forwarded to :func:`tiny_config`.

    Returns:
        A ready-to-run policy.
    """
    config = tiny_config(**overrides)
    policy = LingBotVA(**config.to_dict())
    stats = {
        "action": {
            "name": "action",
            "shape": (config.output_action_dim,),
            "q01": [-1.0] * config.output_action_dim,
            "q99": [1.0] * config.output_action_dim,
        },
    }
    policy._initialize_model(stats)  # noqa: SLF001
    install_fake_frozen_modules(policy.inner_model)
    policy.eval()
    return policy


# ============================================================================ #
# Configuration                                                                #
# ============================================================================ #


class TestLingBotVAConfig:
    """Tests for the LingBot-VA configuration dataclass."""

    def test_defaults_match_released_checkpoints(self) -> None:
        """Default values match the upstream LIBERO configuration."""
        config = LingBotVAConfig()
        assert config.num_layers == 30
        assert config.action_dim == 30
        assert config.height == 128
        assert config.width == 128
        assert config.attn_mode == "torch"
        assert config.used_action_channel_ids == (0, 1, 2, 3, 4, 5, 6)

    def test_derived_action_geometry(self) -> None:
        """Chunk size and output width derive from the frame/action geometry."""
        config = LingBotVAConfig()
        assert config.chunk_size == config.frame_chunk_size * config.action_per_frame
        assert config.n_action_steps == config.chunk_size
        assert config.output_action_dim == 7

    def test_latent_grid_width_scales_with_cameras(self) -> None:
        """Per-camera latents are concatenated on width."""
        one_camera = LingBotVAConfig(obs_cam_keys=("observation.images.image",))
        two_cameras = LingBotVAConfig()
        assert two_cameras.latent_hw[1] == 2 * one_camera.latent_hw[1]
        assert two_cameras.latent_hw[0] == one_camera.latent_hw[0]

    def test_delta_indices(self) -> None:
        """Delta indices cover one full chunk of observations and actions."""
        config = LingBotVAConfig()
        assert len(config.action_delta_indices) == config.chunk_size
        assert len(config.observation_delta_indices) == config.frame_chunk_size * 4
        assert config.reward_delta_indices is None

    def test_invalid_attn_mode(self) -> None:
        """An unknown attention backend is rejected."""
        with pytest.raises(ValueError, match="attn_mode must be one of"):
            LingBotVAConfig(attn_mode="sdpa")  # type: ignore[arg-type]

    def test_tshape_requires_three_cameras(self) -> None:
        """The RoboTwin T-shape layout needs a head plus two wrist cameras."""
        with pytest.raises(ValueError, match="expects exactly 3 cameras"):
            LingBotVAConfig(camera_layout="robotwin_tshape")

    def test_action_channels_must_fit_action_space(self) -> None:
        """Used action channels must index into the model's action space."""
        with pytest.raises(ValueError, match="used_action_channel_ids must be within"):
            LingBotVAConfig(used_action_channel_ids=(0, 99))

    def test_inheritance_and_serialization(self) -> None:
        """The config is a Studio ``Config`` and round-trips through a dict."""
        config = LingBotVAConfig(num_layers=2, optimizer_lr=3e-5)
        assert isinstance(config, Config)

        restored = LingBotVAConfig.from_dict(config.to_dict())
        assert restored == config
        assert restored.num_layers == 2
        assert restored.optimizer_lr == 3e-5

    def test_from_dict_coerces_json_lists(self) -> None:
        """Checkpoint JSON lists are coerced into the dataclass' tuple fields."""
        config = LingBotVAConfig.from_dict(
            {"patch_size": [1, 2, 2], "used_action_channel_ids": [0, 1, 2], "obs_cam_keys": ["a", "b"]},
            strict=False,
        )
        assert config.patch_size == (1, 2, 2)
        assert config.used_action_channel_ids == (0, 1, 2)
        assert config.obs_cam_keys == ("a", "b")


# ============================================================================ #
# Registration                                                                 #
# ============================================================================ #


class TestLingBotVARegistration:
    """Tests for factory and package-level registration."""

    def test_get_policy(self) -> None:
        """``get_policy`` resolves the family by name."""
        policy = get_policy("lingbot_va")
        assert isinstance(policy, LingBotVA)

    def test_get_policy_is_case_insensitive(self) -> None:
        """Policy lookup is case-insensitive."""
        assert isinstance(get_policy("LingBot_VA"), LingBotVA)

    def test_lazy_initialization(self) -> None:
        """Lazy construction defers the model until ``setup()``."""
        policy = LingBotVA()
        assert policy.model is None
        assert policy.config.n_action_steps == 16

    def test_hyperparameters_saved(self) -> None:
        """Constructor arguments land in ``hparams`` for checkpointing."""
        policy = LingBotVA(num_layers=2, optimizer_lr=3e-5)
        assert policy.hparams.num_layers == 2
        assert policy.hparams.optimizer_lr == 3e-5
        assert policy.hparams["config"]["num_layers"] == 2


# ============================================================================ #
# Preprocessing                                                                #
# ============================================================================ #


class TestLingBotVAPreprocessing:
    """Tests for camera-key resolution and action (de)normalization."""

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("observation.images.image", "image"),
            ("images.image", "image"),
            ("image", "image"),
            ("observation.state", "state"),
        ],
    )
    def test_camera_basename(self, key: str, expected: str) -> None:
        """Dataset-specific prefixes are stripped from camera keys."""
        assert camera_basename(key) == expected

    def test_resolve_camera_keys_accepts_studio_batches(self) -> None:
        """LeRobot-named cameras resolve against Studio's flattened batch keys."""
        batch = {"images.image": torch.rand(1, 3, 8, 8), "images.image2": torch.rand(1, 3, 8, 8)}
        resolved = resolve_camera_keys(batch, ("observation.images.image", "observation.images.image2"))
        assert resolved == ["images.image", "images.image2"]

    def test_resolve_camera_keys_preserves_order(self) -> None:
        """Camera order follows the config, not the batch's insertion order."""
        batch = {"images.wrist": torch.rand(1, 3, 8, 8), "images.top": torch.rand(1, 3, 8, 8)}
        assert resolve_camera_keys(batch, ("top", "wrist")) == ["images.top", "images.wrist"]

    def test_resolve_camera_keys_reports_missing(self) -> None:
        """A missing camera names the keys that were tried."""
        with pytest.raises(KeyError, match="not found in batch"):
            resolve_camera_keys({"images.top": torch.rand(1, 3, 8, 8)}, ("wrist",))

    def test_action_normalization_round_trip(self) -> None:
        """The pre- and postprocessor are exact inverses."""
        stats = {"action": {"name": "action", "shape": (3,), "q01": [-2.0, -1.0, 0.0], "q99": [2.0, 3.0, 1.0]}}
        pre, post = make_lingbot_va_preprocessors(stats, used_action_channel_ids=(0, 1, 2))

        actions = torch.tensor([[[1.0, 0.5, 0.25]]])
        normalized = pre({"action": actions})["action"]
        assert normalized.abs().max() <= 1.0
        torch.testing.assert_close(post({"action": normalized})["action"], actions)

    def test_action_stats_are_sliced_to_used_channels(self) -> None:
        """Wider dataset statistics are sliced down to the policy's action channels."""
        stats = {"action": {"name": "action", "shape": (5,), "q01": [-1.0] * 5, "q99": [1.0, 2.0, 3.0, 4.0, 5.0]}}
        pre, _ = make_lingbot_va_preprocessors(stats, used_action_channel_ids=(0, 2))
        normalized = pre({"action": torch.zeros(1, 1, 2)})["action"]
        assert normalized.shape == (1, 1, 2)


# ============================================================================ #
# Inference                                                                    #
# ============================================================================ #


class TestLingBotVAInference:
    """Shape and streaming-behaviour tests for the autoregressive inference path."""

    def test_predict_action_chunk_shape(self) -> None:
        """The first chunk drops the conditioning frame's actions."""
        policy = build_policy()
        config = policy.config
        actions = policy.predict_action_chunk(make_observation(config))

        expected_steps = config.chunk_size - config.action_per_frame
        assert actions.shape == (1, expected_steps, config.output_action_dim)
        assert actions.dtype == torch.float32
        assert torch.isfinite(actions).all()

    def test_later_chunks_use_the_full_horizon(self) -> None:
        """After the first chunk every frame contributes actions."""
        policy = build_policy()
        config = policy.config
        observation = make_observation(config)

        policy.select_action(observation)
        for _ in range(config.chunk_size - config.action_per_frame - 1):
            policy.select_action(observation)

        # The queue is now empty, so the next call predicts a full-length chunk.
        actions = policy._predict_chunk(None)  # noqa: SLF001
        assert actions.shape == (1, config.chunk_size, config.output_action_dim)

    def test_select_action_returns_single_action(self) -> None:
        """``select_action`` hands back one action per call."""
        policy = build_policy()
        config = policy.config
        action = policy.select_action(make_observation(config))
        assert action.shape == (1, config.output_action_dim)

    def test_select_action_refills_the_queue(self) -> None:
        """Executing past the end of a chunk transparently predicts the next one."""
        policy = build_policy()
        config = policy.config
        observation = make_observation(config)

        steps = 2 * config.chunk_size
        actions = [policy.select_action(observation) for _ in range(steps)]
        assert len(actions) == steps
        assert all(a.shape == (1, config.output_action_dim) for a in actions)
        assert all(torch.isfinite(a).all() for a in actions)

    def test_observations_are_buffered_as_keyframes(self) -> None:
        """Every executed step feeds an observation back into the keyframe buffer."""
        policy = build_policy()
        model = policy.inner_model
        observation = make_observation(policy.config)

        policy.select_action(observation)
        assert model._obs_buffer == []  # noqa: SLF001

        policy.select_action(observation)
        assert len(model._obs_buffer) == 1  # noqa: SLF001

    def test_reset_clears_streaming_state(self) -> None:
        """``reset`` returns the policy to its start-of-episode state."""
        policy = build_policy()
        observation = make_observation(policy.config)
        policy.select_action(observation)
        policy.select_action(observation)

        policy.reset()
        model = policy.inner_model
        assert len(policy._action_queue) == 0  # noqa: SLF001
        assert model.streaming_started is False
        assert model._first_chunk is True  # noqa: SLF001
        assert model._frame_st_id == 0  # noqa: SLF001

        # A fresh episode runs from scratch.
        assert policy.select_action(observation).shape == (1, policy.config.output_action_dim)

    def test_classifier_free_guidance_path(self) -> None:
        """The CFG path doubles the batch and still yields finite actions."""
        policy = build_policy(guidance_scale=5.0, action_guidance_scale=2.0)
        assert policy.inner_model._use_cfg is True  # noqa: SLF001
        action = policy.select_action(make_observation(policy.config))
        assert action.shape == (1, policy.config.output_action_dim)
        assert torch.isfinite(action).all()

    def test_actions_are_denormalized(self) -> None:
        """Predictions come back in physical units, not the model's [-1, 1] space."""
        config = tiny_config()
        policy = LingBotVA(**config.to_dict())
        stats = {
            "action": {
                "name": "action",
                "shape": (3,),
                "q01": [0.0, 0.0, 0.0],
                "q99": [10.0, 10.0, 10.0],
            },
        }
        policy._initialize_model(stats)  # noqa: SLF001
        install_fake_frozen_modules(policy.inner_model)
        policy.eval()
        observation = make_observation(config)

        torch.manual_seed(0)
        policy.reset()
        raw = policy.inner_model.predict_action_chunk(policy._preprocess(observation))  # noqa: SLF001

        torch.manual_seed(0)
        policy.reset()
        actions = policy.predict_action_chunk(observation)

        # q01=0, q99=10 maps the model's [-1, 1] space onto [0, 10].
        torch.testing.assert_close(actions, (raw + 1.0) * 5.0)


# ============================================================================ #
# Training                                                                     #
# ============================================================================ #


class TestLingBotVATraining:
    """Tests for the training path's preconditions."""

    def test_training_requires_flex_attention(self) -> None:
        """The inference-only SDPA backend cannot build the block-causal masks."""
        policy = build_policy()
        observation = make_observation(policy.config)
        observation.action = torch.zeros(1, policy.config.chunk_size, policy.config.output_action_dim)

        policy.train()
        with pytest.raises(ValueError, match="requires attn_mode='flex'"):
            policy.training_step(observation, 0)

    def test_optimizer_and_scheduler(self) -> None:
        """AdamW is paired with the upstream warmup-then-constant schedule."""
        policy = build_policy(scheduler_warmup_steps=10)
        configured = policy.configure_optimizers()

        optimizer = configured["optimizer"]
        assert isinstance(optimizer, torch.optim.AdamW)
        # LambdaLR applies its factor on construction, so the peak LR lives in initial_lr.
        assert optimizer.param_groups[0]["initial_lr"] == policy.config.optimizer_lr

        scheduler = configured["lr_scheduler"]["scheduler"]
        assert configured["lr_scheduler"]["interval"] == "step"
        assert scheduler.lr_lambdas[0](0) == pytest.approx(0.1)
        assert scheduler.lr_lambdas[0](9) == pytest.approx(1.0)
        assert scheduler.lr_lambdas[0](1000) == pytest.approx(1.0)

    def test_only_the_transformer_is_trainable(self) -> None:
        """The frozen sub-models are not parameters of the model."""
        policy = build_policy()
        model = policy.inner_model
        assert model.get_optim_params()
        assert all(name.startswith("transformer.") for name, _ in model.named_parameters())


# ============================================================================ #
# Checkpoint loading                                                           #
# ============================================================================ #


def write_fake_checkpoint(directory: Any, action_dim: int = 7) -> Any:
    """Write a checkpoint in the published LeRobot layout.

    Args:
        directory: Directory to write into.
        action_dim: Width of the checkpoint's action space.

    Returns:
        The directory that was written to.
    """
    import json  # noqa: PLC0415

    from safetensors.torch import save_file  # noqa: PLC0415

    config = tiny_config()
    payload = config.to_dict()
    payload.update({
        "type": "lingbot_va",
        "device": "cpu",
        "use_amp": False,
        "push_to_hub": True,
        "input_features": {k: {"type": "VISUAL", "shape": [3, 256, 256]} for k in config.obs_cam_keys},
        "output_features": {"action": {"type": "ACTION", "shape": [action_dim]}},
        "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "IDENTITY", "ACTION": "IDENTITY"},
    })
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    (directory / "policy_postprocessor.json").write_text(
        json.dumps({
            "name": "policy_postprocessor",
            "steps": [
                {
                    "registry_name": "unnormalizer_processor",
                    "config": {
                        "features": {"action": {"type": "ACTION", "shape": [action_dim]}},
                        "norm_map": {"ACTION": "QUANTILES"},
                    },
                    "state_file": "postprocessor.safetensors",
                },
                {"registry_name": "device_processor", "config": {"device": "cpu"}},
            ],
        }),
        encoding="utf-8",
    )
    save_file(
        {"action.q01": -torch.ones(action_dim), "action.q99": torch.ones(action_dim)},
        str(directory / "postprocessor.safetensors"),
    )

    model = LingBotVAModel(config)
    save_file(dict(model.state_dict()), str(directory / "model.safetensors"))
    return directory


class TestLingBotVACheckpointLoading:
    """Tests for loading checkpoints published in the LeRobot layout."""

    def test_resolve_local_checkpoint(self, tmp_path: Any) -> None:
        """A local directory resolves every published artefact."""
        from physicalai.policies.lingbot_va.pretrained_utils import resolve_checkpoint  # noqa: PLC0415

        write_fake_checkpoint(tmp_path)
        files = resolve_checkpoint(tmp_path)
        assert files.config_file.name == "config.json"
        assert files.weights_file.name == "model.safetensors"
        assert files.postprocessor_file is not None
        assert files.postprocessor_dir == tmp_path

    def test_load_config_drops_lerobot_only_keys(self, tmp_path: Any) -> None:
        """Feature specs and hub metadata do not leak into the Studio config."""
        from physicalai.policies.lingbot_va.pretrained_utils import load_config  # noqa: PLC0415

        write_fake_checkpoint(tmp_path)
        config = load_config(tmp_path / "config.json", {"attn_mode": "flex"})
        assert config.num_layers == 1
        assert config.attn_mode == "flex"
        assert not hasattr(config, "input_features")

    def test_load_config_ignores_none_overrides(self, tmp_path: Any) -> None:
        """``None`` overrides mean "unset" and keep the checkpoint's value."""
        from physicalai.policies.lingbot_va.pretrained_utils import load_config  # noqa: PLC0415

        write_fake_checkpoint(tmp_path)
        config = load_config(tmp_path / "config.json", {"attn_mode": None})
        assert config.attn_mode == "torch"

    def test_extract_action_stats(self, tmp_path: Any) -> None:
        """Action quantiles are read out of the postprocessor state file."""
        from physicalai.policies.lingbot_va.pretrained_utils import (  # noqa: PLC0415
            detect_normalization_mode,
            extract_action_stats,
        )

        write_fake_checkpoint(tmp_path, action_dim=5)
        assert detect_normalization_mode(tmp_path / "policy_postprocessor.json") == "QUANTILES"

        stats = extract_action_stats(tmp_path / "policy_postprocessor.json", tmp_path)
        assert stats["action"]["shape"] == (5,)
        assert len(stats["action"]["q01"]) == 5
        assert len(stats["action"]["q99"]) == 5

    def test_end_to_end_pretrained_load(self, tmp_path: Any) -> None:
        """A published checkpoint builds a ready-to-run policy in one call."""
        write_fake_checkpoint(tmp_path, action_dim=3)

        policy = LingBotVA(pretrained_name_or_path=tmp_path)
        assert policy.model is not None
        assert policy.config.num_layers == 1
        assert policy.config.output_action_dim == 3
        assert policy._dataset_stats is not None  # noqa: SLF001

        install_fake_frozen_modules(policy.inner_model)
        policy.eval()
        action = policy.select_action(make_observation(policy.config))
        assert action.shape == (1, 3)

    def test_pretrained_load_honours_explicit_overrides(self, tmp_path: Any) -> None:
        """Caller arguments win over the checkpoint's published values."""
        write_fake_checkpoint(tmp_path)
        policy = LingBotVA(pretrained_name_or_path=tmp_path, attn_mode="flex", optimizer_lr=7e-6)
        assert policy.config.attn_mode == "flex"
        assert policy.config.optimizer_lr == 7e-6
        # Untouched fields keep the checkpoint's values.
        assert policy.config.num_layers == 1


# ============================================================================ #
# Export                                                                       #
# ============================================================================ #


class TestLingBotVAExport:
    """Tests for the Torch export path."""

    def test_only_the_torch_backend_is_supported(self) -> None:
        """Tracing backends are deliberately not offered."""
        policy = build_policy()
        assert isinstance(policy, ExportablePolicyMixin)
        assert policy.get_supported_export_backends() == [ExportBackend.TORCH]

    @pytest.mark.parametrize("backend", [ExportBackend.ONNX, ExportBackend.OPENVINO, ExportBackend.EXECUTORCH])
    def test_tracing_backends_are_rejected(self, backend: ExportBackend, tmp_path: Any) -> None:
        """Asking for a tracing backend fails instead of emitting a broken artifact."""
        policy = build_policy()
        with pytest.raises(NotImplementedError):
            policy.export(tmp_path, backend=backend)

    def test_schemas_describe_cameras_task_and_action_chunk(self) -> None:
        """Every configured camera, the task prompt and the action chunk are described."""
        policy = build_policy()
        config = policy.config

        inputs = policy.inputs_schema
        assert inputs is not None
        assert [feature.name for feature in inputs] == ["images.image", "images.image2", "task"]
        assert [feature.ftype for feature in inputs] == [
            InferenceFeatureType.VISUAL,
            InferenceFeatureType.VISUAL,
            InferenceFeatureType.LANGUAGE,
        ]
        assert inputs[0].shape == (3, config.height, config.width)
        assert inputs[-1].dtype is InferenceFeatureDtype.STRING

        outputs = policy.outputs_schema
        assert outputs is not None
        assert len(outputs) == 1
        assert outputs[0].name == "action"
        assert outputs[0].shape == (config.chunk_size, config.output_action_dim)

    def test_camera_shape_prefers_dataset_statistics(self) -> None:
        """A dataset's own camera resolution wins over the VAE input resolution."""
        policy = build_policy()
        policy._dataset_stats = {  # noqa: SLF001
            "observation.images.image": {"type": "VISUAL", "shape": (3, 96, 128)},
        }
        inputs = policy.inputs_schema
        assert inputs is not None
        assert inputs[0].shape == (3, 96, 128)
        # The camera missing from the stats falls back to the configured VAE resolution.
        assert inputs[1].shape == (3, policy.config.height, policy.config.width)

    def test_schemas_are_none_before_initialization(self) -> None:
        """A lazily-constructed policy has nothing to describe yet."""
        policy = LingBotVA(**tiny_config().to_dict())
        assert policy.inputs_schema is None
        assert policy.outputs_schema is None

    def test_export_writes_checkpoint_and_manifest(self, tmp_path: Any) -> None:
        """``to_torch`` writes both artifacts the export contract requires."""
        policy = build_policy()
        policy.export(tmp_path, backend=ExportBackend.TORCH)

        assert (tmp_path / "lingbotva.pt").exists()
        manifest = Manifest.load(tmp_path / "manifest.json")
        assert manifest.model.artifacts == {"torch": "lingbotva.pt"}
        assert manifest.policy.source.class_path.endswith("lingbot_va.policy.LingBotVA")
        assert [spec.type for spec in manifest.model.preprocessors] == ["to_float_tensor"]
        # n_action_steps == chunk_size, so nothing has to trim the chunk.
        assert manifest.model.postprocessors == []
        assert [spec.init_args["name"] for spec in manifest.model.output_features] == ["action"]

    def test_frozen_stack_stays_out_of_the_checkpoint(self, tmp_path: Any) -> None:
        """The ~20 GB VAE/UMT5 stack is not serialized with the transformer."""
        policy = build_policy()
        policy.to_torch(tmp_path)

        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        checkpoint = torch.load(tmp_path / "lingbotva.pt", map_location="cpu", weights_only=False)  # nosec B614
        owners = {key.split(".")[0] + "." + key.split(".")[1] for key in checkpoint["state_dict"]}
        assert owners == {
            "model.transformer",
            "_preprocessor._action_normalizer",
            "_postprocessor._action_denormalizer",
        }

    def test_exported_checkpoint_round_trips(self, tmp_path: Any) -> None:
        """The exported checkpoint rebuilds an identical policy."""
        policy = build_policy()
        policy.to_torch(tmp_path)

        restored = LingBotVA.load_from_checkpoint(
            tmp_path / "lingbotva.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert restored.config == policy.config
        assert restored._dataset_stats == policy._dataset_stats  # noqa: SLF001

        original_state = policy.state_dict()
        restored_state = restored.state_dict()
        assert set(original_state) == set(restored_state)
        assert all(torch.equal(original_state[key], restored_state[key]) for key in original_state)

    def test_restored_policy_matches_numerically(self, tmp_path: Any) -> None:
        """The restored policy predicts the same chunk as the exported one."""
        policy = build_policy()
        policy.to_torch(tmp_path)

        restored = LingBotVA.load_from_checkpoint(
            tmp_path / "lingbotva.pt",
            map_location="cpu",
            weights_only=False,
        )
        install_fake_frozen_modules(restored.inner_model)
        restored.eval()

        observation = make_observation(policy.config)
        torch.manual_seed(0)
        expected = policy.predict_action_chunk(observation)
        torch.manual_seed(0)
        actual = restored.predict_action_chunk(observation)
        assert torch.allclose(expected, actual, atol=1e-6)
