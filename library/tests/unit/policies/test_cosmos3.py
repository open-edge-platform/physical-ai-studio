# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Cosmos3 policy.

Fast, self-contained tests with no external network calls or heavy model downloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.utils.outputs import BaseOutput

from physicalai.config import Config
from physicalai.data.observation import ACTION, IMAGES, STATE, Observation
from physicalai.policies import Cosmos3, Cosmos3Config, Cosmos3Model, get_physicalai_policy_class, get_policy
from physicalai.policies.cosmos3 import Cosmos3Preprocessor, compose_horizontal_views, compose_t_views
from physicalai.policies.cosmos3.flow_matching import _unpack_transformer_output, flow_matching_step

# ============================================================================ #
# Configuration Tests                                                          #
# ============================================================================ #


class TestCosmos3Config:
    """Tests for Cosmos3Config dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = Cosmos3Config(embodiment="pusht")
        assert config.pretrained_model_name_or_path == "nvidia/Cosmos3-Edge"
        assert config.revision is None
        assert config.mode == "peft"
        assert config.paradigm == "policy"
        assert config.rank == 32
        assert config.alpha_scale == 1.0
        assert not config.dora
        assert config.head_lr_mult == 2.0
        assert config.action_weight == 10.0
        assert config.chunk_size == 32
        assert config.n_action_steps == 32
        assert config.resolution_tier == 256
        assert config.fps == 10
        assert config.grad_checkpoint is True
        assert config.embodiment == "pusht"
        assert config.action_space is None
        assert config.view_point is None
        assert config.dtype == "bfloat16"
        assert config.optimizer_lr == 1e-4

    def test_full_mode_defaults(self) -> None:
        """Test mode='full' default head_lr_mult is 10.0."""
        config = Cosmos3Config(embodiment="pusht", mode="full")
        assert config.mode == "full"
        assert config.head_lr_mult == 10.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = Cosmos3Config(
            embodiment="droid_lerobot",
            mode="full",
            paradigm="joint",
            head_lr_mult=5.0,
            chunk_size=16,
            n_action_steps=16,
            view_point="concat_view",
            revision="abc1234",
        )
        assert config.mode == "full"
        assert config.paradigm == "joint"
        assert config.head_lr_mult == 5.0
        assert config.chunk_size == 16
        assert config.n_action_steps == 16
        assert config.embodiment == "droid_lerobot"
        assert config.view_point == "concat_view"
        assert config.revision == "abc1234"

    def test_n_action_steps_validation(self) -> None:
        """Test n_action_steps cannot exceed chunk_size."""
        with pytest.raises(ValueError, match="cannot exceed chunk_size"):
            Cosmos3Config(embodiment="pusht", chunk_size=16, n_action_steps=32)

    def test_invalid_mode(self) -> None:
        """Test invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            Cosmos3Config(embodiment="pusht", mode="invalid")  # type: ignore[arg-type]

    def test_invalid_paradigm(self) -> None:
        """Test invalid paradigm raises ValueError."""
        with pytest.raises(ValueError, match="Invalid paradigm"):
            Cosmos3Config(embodiment="pusht", paradigm="invalid")  # type: ignore[arg-type]

    def test_invalid_resolution_tier(self) -> None:
        """Test invalid resolution_tier raises ValueError."""
        with pytest.raises(ValueError, match="Invalid resolution_tier"):
            Cosmos3Config(embodiment="pusht", resolution_tier=128)  # type: ignore[arg-type]

    def test_invalid_dtype(self) -> None:
        """Test invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            Cosmos3Config(embodiment="pusht", dtype="int8")  # type: ignore[arg-type]

    def test_serialization(self) -> None:
        """Test to_dict and from_dict round-trip."""
        config = Cosmos3Config(embodiment="pusht", rank=64, chunk_size=16, n_action_steps=16)
        assert isinstance(config, Config)
        cfg_dict = config.to_dict()
        assert cfg_dict["rank"] == 64
        assert cfg_dict["chunk_size"] == 16

        restored = Cosmos3Config.from_dict(cfg_dict)
        assert restored.rank == 64
        assert restored.chunk_size == 16

    def test_frozen_dataclass(self) -> None:
        """Test config dataclass is immutable."""
        config = Cosmos3Config(embodiment="pusht")
        with pytest.raises(AttributeError):
            config.chunk_size = 10  # type: ignore[misc]


# ============================================================================ #
# Policy Tests                                                                 #
# ============================================================================ #


class TestCosmos3Policy:
    """Tests for Cosmos3 Lightning policy wrapper."""

    def test_lazy_initialization(self) -> None:
        """Test policy initialization does not eagerly instantiate model."""
        policy = Cosmos3(embodiment="pusht")
        assert policy.model is None

    def test_hyperparameters_saved(self) -> None:
        """Test hyperparameters and config dictionary are saved."""
        policy = Cosmos3(embodiment="pusht", chunk_size=16, n_action_steps=16, mode="peft")
        assert policy.hparams.chunk_size == 16
        assert policy.hparams.mode == "peft"
        assert "config" in policy.hparams
        assert policy.hparams["config"]["chunk_size"] == 16

    def test_from_config(self) -> None:
        """Test instantiation via from_config classmethod."""
        config = Cosmos3Config(embodiment="pusht", chunk_size=16, n_action_steps=16)
        policy = Cosmos3.from_config(config)
        assert policy.model is None
        assert policy.config.chunk_size == 16

    def test_methods_raise_without_model(self) -> None:
        """Test forward and predict_action_chunk raise before setup()."""
        policy = Cosmos3(embodiment="pusht")
        obs = Observation(images=torch.randn(1, 3, 224, 224))
        with pytest.raises(RuntimeError, match="not initialized"):
            policy.forward(obs)
        with pytest.raises(RuntimeError, match="not initialized"):
            policy.predict_action_chunk(obs)
        with pytest.raises(RuntimeError, match="not initialized"):
            policy.configure_optimizers()

    def test_policy_factory_registration(self) -> None:
        """Test get_policy and get_physicalai_policy_class factory dispatch."""
        cls = get_physicalai_policy_class("cosmos3")
        assert cls is Cosmos3

        policy = get_policy("cosmos3", embodiment="pusht")
        assert isinstance(policy, Cosmos3)
        assert policy.model is None


# ============================================================================ #
# Mocked Model & Pipeline Tests                                                #
# ============================================================================ #


class TestMockedCosmos3Model:
    """Tests for Cosmos3Model wrapping mock pipeline primitives."""

    def _create_mock_pipeline(self) -> MagicMock:
        """Create mock diffusers pipeline with required attributes."""
        pipe = MagicMock()
        pipe.transformer = MagicMock()
        pipe.transformer.device = torch.device("cpu")
        pipe.transformer.dtype = torch.float32
        pipe.transformer.config.action_dim = 64

        # Action projection head mock layers for init_domain_action_head
        num_domains, in_size, out_size = 32, 64, 64
        for proj_name in ("action_proj_in", "action_proj_out"):
            proj = MagicMock()
            proj.num_domains = num_domains
            proj.input_size = in_size
            proj.output_size = out_size
            proj.fc.weight = torch.nn.Parameter(torch.randn(num_domains * in_size * out_size))
            proj.bias.weight = torch.nn.Parameter(torch.randn(num_domains, out_size))
            setattr(pipe.transformer, proj_name, proj)

        pipe.transformer.action_modality_embed = torch.nn.Parameter(torch.randn(1, 64))

        pipe.transformer.parameters.return_value = [torch.nn.Parameter(torch.zeros(2, 2))]
        pipe.transformer.named_parameters.return_value = [
            ("to_q.weight", torch.nn.Parameter(torch.zeros(2, 2))),
            ("action_proj_in.weight", torch.nn.Parameter(torch.zeros(2, 2))),
        ]

        pipe.vae = MagicMock()
        pipe.video_processor = MagicMock()
        pipe.scheduler = MagicMock()
        pipe.scheduler.config = {}

        return pipe

    def test_delta_indices(self) -> None:
        """Test model exposes expected action and observation delta indices."""
        config = Cosmos3Config(embodiment="pusht", chunk_size=32)
        pipe = self._create_mock_pipeline()
        model = Cosmos3Model(config, pipeline=pipe)

        assert model.reward_delta_indices is None
        assert model.action_delta_indices == list(range(32))
        assert model.observation_delta_indices == list(range(33))

    def test_action_space_resolution(self) -> None:
        """Embodiment maps to a default action space; an explicit override wins."""
        pipe = self._create_mock_pipeline()

        default_model = Cosmos3Model(Cosmos3Config(embodiment="pusht", chunk_size=4), pipeline=pipe)
        assert default_model.action_space == "identity"

        droid_model = Cosmos3Model(Cosmos3Config(embodiment="droid_lerobot", chunk_size=4), pipeline=pipe)
        assert droid_model.action_space == "joint_pos"

        override_model = Cosmos3Model(
            Cosmos3Config(embodiment="pusht", action_space="joint_pos", chunk_size=4),
            pipeline=pipe,
        )
        assert override_model.action_space == "joint_pos"

    def test_unsupported_action_space_raises(self) -> None:
        """An unsupported action_space override is rejected at model build time."""
        pipe = self._create_mock_pipeline()
        with pytest.raises(ValueError, match="Unsupported action_space"):
            Cosmos3Model(Cosmos3Config(embodiment="pusht", action_space="droid_ee", chunk_size=4), pipeline=pipe)

    def test_set_dataset_stats(self) -> None:
        """Test updating normalization bounds via dataset stats."""
        config = Cosmos3Config(embodiment="pusht", chunk_size=16)
        pipe = self._create_mock_pipeline()
        model = Cosmos3Model(config, pipeline=pipe)

        stats = {
            "action": {
                "min": [10.0, 20.0],
                "max": [50.0, 80.0],
            },
        }
        model.set_dataset_stats(stats)
        assert model.raw_dim == 2
        # minmax -> offset=(min+max)/2, scale=(max-min)/2
        torch.testing.assert_close(model.norm_offset, torch.tensor([30.0, 50.0]))
        torch.testing.assert_close(model.norm_scale, torch.tensor([20.0, 30.0]))

    def test_predict_action_chunk_mocked(self) -> None:
        """Test predict_action_chunk flow with mock pipeline output."""
        config = Cosmos3Config(embodiment="pusht", chunk_size=4)
        pipe = self._create_mock_pipeline()

        # Mock result: action tensor of shape [1, chunk_size + 1, 64]
        mock_actions = torch.zeros(1, 5, 64)
        # Action tokens 1..4 in normalized [-1, 1] space
        mock_actions[0, 1:5, :2] = 0.0  # midpoint maps to (min + max)/2 = (0 + 10)/2 = 5.0
        pipe.return_value = MagicMock(action=mock_actions)

        model = Cosmos3Model(config, pipeline=pipe)
        # min=[0,0], max=[10,10] -> offset=[5,5], scale=[5,5]
        model.norm_offset = torch.tensor([5.0, 5.0])
        model.norm_scale = torch.tensor([5.0, 5.0])

        batch = {
            IMAGES: torch.zeros(1, 3, 224, 224),
            "state": torch.tensor([[5.0, 5.0]]),
        }
        preds = model.predict_action_chunk(batch)

        assert preds.shape == (1, 4, 2)
        torch.testing.assert_close(preds, torch.full((1, 4, 2), 5.0))

    def test_compute_loss_mocked(self) -> None:
        """Test compute_loss flow with mock pipeline and pack caching."""
        from unittest.mock import patch

        config = Cosmos3Config(embodiment="pusht", chunk_size=4)
        pipe = self._create_mock_pipeline()

        pipe._prepare_action_video_conditioning.return_value = (
            torch.zeros(1, 3, 5, 224, 224),
            torch.tensor([1, 3, 224, 224]),
            224,
            224,
        )
        pipe._remove_action_video_padding_from_latent.return_value = torch.zeros(1, 16, 2, 14, 14)
        pipe._encode_video.return_value = torch.zeros(1, 16, 2, 14, 14)

        model = Cosmos3Model(config, pipeline=pipe)
        model._get_or_build_pack = MagicMock(return_value={})

        batch = {
            IMAGES: torch.zeros(1, 3, 224, 224),
            ACTION: torch.zeros(1, 4, 2),
            STATE: torch.zeros(1, 2),
        }
        with patch("physicalai.policies.cosmos3.model.flow_matching_step") as mock_step:
            mock_step.return_value = (torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.05))
            loss, loss_dict = model.compute_loss(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 1.0
        assert "loss" in loss_dict
        assert "loss_vision" in loss_dict
        assert "loss_action" in loss_dict

    def test_on_save_checkpoint_filtering(self) -> None:
        """Test policy on_save_checkpoint removes frozen parameters and keeps trainable ones."""
        pipe = self._create_mock_pipeline()
        policy = Cosmos3(chunk_size=4, embodiment="pusht", pipeline=pipe)

        # Register trainable and frozen parameters to simulate LoRA and base weights
        assert policy.model is not None
        policy.model.trainable_lora = torch.nn.Parameter(torch.randn(2, 2), requires_grad=True)
        policy.model.frozen_backbone = torch.nn.Parameter(torch.randn(10, 10), requires_grad=False)

        # Mock state_dict with frozen and trainable keys
        checkpoint = {
            "state_dict": {
                "model.trainable_lora": torch.randn(2, 2),
                "model.frozen_backbone": torch.randn(10, 10),
                "model.norm_offset": torch.tensor([0.0, 0.0]),
                "model.norm_scale": torch.tensor([1.0, 1.0]),
                "model.domain_id": torch.tensor([4]),
            },
        }
        policy.on_save_checkpoint(checkpoint)

        saved_keys = checkpoint["state_dict"].keys()
        assert "model.trainable_lora" in saved_keys
        assert "model.norm_offset" in saved_keys
        assert "model.norm_scale" in saved_keys
        assert "model.domain_id" in saved_keys
        assert "model.frozen_backbone" not in saved_keys

    def test_pretrained_action_head_detection(self, tmp_path: Path) -> None:
        """Test _has_pretrained_action_head detection for local files and remote repos."""
        from physicalai.policies.cosmos3.model import _has_pretrained_action_head

        # 1. Non-existent path returns False
        assert not _has_pretrained_action_head(str(tmp_path / "nonexistent"), "droid_lerobot")

        # 2. Local folder with <domain>_head.pt
        head_dir = tmp_path / "ckpt_head"
        head_dir.mkdir()
        (head_dir / "droid_lerobot_head.pt").touch()
        assert _has_pretrained_action_head(str(head_dir), "droid_lerobot")
        assert not _has_pretrained_action_head(str(head_dir), "pusht")

        # 3. Local folder with checkpoint.json
        json_dir = tmp_path / "ckpt_json"
        json_dir.mkdir()
        (json_dir / "checkpoint.json").touch()
        assert _has_pretrained_action_head(str(json_dir), "any_domain")

        # 4. Remote repository checks file_exists with revision
        with patch("physicalai.policies.cosmos3.model.file_exists", return_value=True) as mock_file_exists:
            assert _has_pretrained_action_head("nvidia/cosmos3-repo", "droid_lerobot", revision="sha_123")
            mock_file_exists.assert_called_once_with(
                repo_id="nvidia/cosmos3-repo",
                filename="checkpoint.json",
                revision="sha_123",
            )

    def test_model_from_pretrained_passes_revision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Cosmos3Model passes revision to PolicyPipelineWithState.from_pretrained."""
        from unittest.mock import patch

        pipe = self._create_mock_pipeline()
        with patch(
            "physicalai.policies.cosmos3.model.PolicyPipelineWithState.from_pretrained",
            return_value=pipe,
        ) as mock_from_pretrained, patch(
            "physicalai.policies.cosmos3.model._has_pretrained_action_head",
            return_value=True,
        ) as mock_has_head:
            config = Cosmos3Config(
                embodiment="pusht",
                pretrained_model_name_or_path="nvidia/Cosmos3-Edge",
                revision="pin_sha_abc",
            )
            Cosmos3Model(config)

            assert mock_from_pretrained.called
            assert mock_from_pretrained.call_args.kwargs.get("revision") == "pin_sha_abc"
            assert mock_has_head.called
            assert mock_has_head.call_args.kwargs.get("revision") == "pin_sha_abc"

    def test_compute_loss_multi_camera_viewpoint(self) -> None:
        """Test compute_loss with multi-camera DROID input composes views and passes concat_view."""
        from unittest.mock import patch

        config = Cosmos3Config(embodiment="droid_lerobot", chunk_size=4)
        pipe = self._create_mock_pipeline()

        pipe._prepare_action_video_conditioning.return_value = (
            torch.zeros(1, 3, 5, 224, 224),
            torch.tensor([1, 3, 224, 224]),
            224,
            224,
        )
        pipe._remove_action_video_padding_from_latent.return_value = torch.zeros(1, 16, 2, 14, 14)
        pipe._encode_video.return_value = torch.zeros(1, 16, 2, 14, 14)

        model = Cosmos3Model(config, pipeline=pipe)
        model._get_or_build_pack = MagicMock(return_value={})

        batch = {
            "images.wrist_image_left": torch.zeros(1, 3, 224, 224),
            "images.exterior_image_1_left": torch.zeros(1, 3, 224, 224),
            "images.exterior_image_2_left": torch.zeros(1, 3, 224, 224),
            ACTION: torch.zeros(1, 4, 10),
            STATE: torch.zeros(1, 10),
        }
        with patch("physicalai.policies.cosmos3.model.flow_matching_step") as mock_step:
            mock_step.return_value = (torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.05))
            model.compute_loss(batch)

        assert model._get_or_build_pack.called
        call_kwargs = model._get_or_build_pack.call_args.kwargs
        assert call_kwargs.get("view_point") == "concat_view"

        # Verify the conditioning clip passed to diffusers had composed T-shape height (3 * 224 // 2 = 336)
        prep_call_args = pipe._prepare_action_video_conditioning.call_args[0]
        clip_arg = prep_call_args[0]  # [T, C, H, W]
        assert clip_arg.shape[-2] == 336
        assert clip_arg.shape[-1] == 224

    def test_predict_action_chunk_multi_camera_viewpoint(self) -> None:
        """Test predict_action_chunk composes DROID views and passes concat_view to condition."""
        config = Cosmos3Config(embodiment="droid_lerobot", chunk_size=4)
        pipe = self._create_mock_pipeline()

        mock_actions = torch.zeros(1, 5, 64)
        pipe.return_value = MagicMock(action=mock_actions)

        model = Cosmos3Model(config, pipeline=pipe)
        batch = {
            IMAGES: {
                "wrist_image_left": torch.zeros(3, 224, 224),
                "exterior_image_1_left": torch.zeros(3, 224, 224),
                "exterior_image_2_left": torch.zeros(3, 224, 224),
            },
        }
        model.predict_action_chunk(batch)

        assert pipe.called
        call_kwargs = pipe.call_args.kwargs
        action_cond = call_kwargs["action"]
        assert action_cond.view_point == "concat_view"
        # PIL image height should be 336 (3 * 224 / 2)
        assert action_cond.image.size == (224, 336)  # PIL size is (W, H)

    def test_predict_action_chunk_pusht_viewpoint(self) -> None:
        """Test predict_action_chunk sets top_down_2d_view for pusht domain."""
        config = Cosmos3Config(embodiment="pusht", chunk_size=4)
        pipe = self._create_mock_pipeline()

        mock_actions = torch.zeros(1, 5, 64)
        pipe.return_value = MagicMock(action=mock_actions)

        model = Cosmos3Model(config, pipeline=pipe)
        batch = {IMAGES: torch.zeros(3, 224, 224)}
        model.predict_action_chunk(batch)

        assert pipe.called
        call_kwargs = pipe.call_args.kwargs
        action_cond = call_kwargs["action"]
        assert action_cond.view_point == "top_down_2d_view"


# ============================================================================ #
# View Composition & Preprocessor Tests                                        #
# ============================================================================ #


class TestViewComposition:
    """Tests for compose_t_views and compose_horizontal_views functions."""

    def test_compose_t_views_3d(self) -> None:
        """Test T-shape composition with 3D (C, H, W) tensors."""
        top = torch.zeros(3, 100, 100)
        left = torch.zeros(3, 80, 80)
        right = torch.zeros(3, 90, 90)

        out = compose_t_views(top, left, right)
        assert out.shape == (3, 150, 100)

    def test_compose_t_views_4d_batched(self) -> None:
        """Test T-shape composition with 4D (B, C, H, W) tensors."""
        top = torch.zeros(2, 3, 224, 224)
        left = torch.zeros(2, 3, 224, 224)
        right = torch.zeros(2, 3, 224, 224)

        out = compose_t_views(top, left, right)
        assert out.shape == (2, 3, 336, 224)

    def test_compose_t_views_5d_temporal(self) -> None:
        """Test T-shape composition with 5D (B, T, C, H, W) tensors."""
        top = torch.zeros(2, 4, 3, 224, 224)
        left = torch.zeros(2, 4, 3, 224, 224)
        right = torch.zeros(2, 4, 3, 224, 224)

        out = compose_t_views(top, left, right)
        assert out.shape == (2, 4, 3, 336, 224)

    def test_compose_t_views_odd_dimension(self) -> None:
        """Test T-shape composition with odd width and height."""
        top = torch.zeros(3, 101, 101)
        left = torch.zeros(3, 60, 60)
        right = torch.zeros(3, 60, 60)

        out = compose_t_views(top, left, right)
        assert out.shape == (3, 151, 101)

    def test_compose_horizontal_views_3d(self) -> None:
        """Test horizontal side-by-side composition with 3D (C, H, W) tensors."""
        left = torch.zeros(3, 128, 128)
        right = torch.zeros(3, 128, 128)

        out = compose_horizontal_views(left, right)
        assert out.shape == (3, 128, 256)

    def test_compose_horizontal_views_different_resolution(self) -> None:
        """Test horizontal side-by-side composition resizes right to match left."""
        left = torch.zeros(2, 3, 128, 128)
        right = torch.zeros(2, 3, 256, 256)

        out = compose_horizontal_views(left, right)
        assert out.shape == (2, 3, 128, 256)


class TestCosmos3Preprocessor:
    """Tests for Cosmos3Preprocessor."""

    def test_droid_t_shape_composition(self) -> None:
        """Test DROID 3-camera input triggers T-shape composition and concat_view."""
        preprocessor = Cosmos3Preprocessor(embodiment="droid_lerobot")
        batch = {
            IMAGES: {
                "wrist_image_left": torch.zeros(3, 224, 224),
                "exterior_image_1_left": torch.zeros(3, 224, 224),
                "exterior_image_2_left": torch.zeros(3, 224, 224),
            },
        }
        result = preprocessor(batch)

        assert result["view_point"] == "concat_view"
        assert result["viewpoint"] == "concat_view"
        assert result[IMAGES].shape == (3, 336, 224)

    @pytest.mark.parametrize(
        ("embodiment", "expected_viewpoint"),
        [
            ("pusht", "top_down_2d_view"),
            ("aloha", None),
        ],
    )
    def test_single_camera_embodiments(self, embodiment: str, expected_viewpoint: str | None) -> None:
        """Test single-camera embodiments set expected default viewpoints."""
        preprocessor = Cosmos3Preprocessor(embodiment=embodiment)
        batch = {IMAGES: torch.zeros(3, 224, 224)}
        result = preprocessor(batch)

        assert result["view_point"] == expected_viewpoint
        assert result[IMAGES].shape == (3, 224, 224)

    def test_explicit_viewpoint_override(self) -> None:
        """Test explicit view_point override takes precedence."""
        preprocessor = Cosmos3Preprocessor(embodiment="pusht", view_point="custom_view")
        batch = {IMAGES: torch.zeros(3, 224, 224)}
        result = preprocessor(batch)

        assert result["view_point"] == "custom_view"

    def test_observation_input(self) -> None:
        """Test preprocessor handles Observation dataclass and preserves fields."""
        preprocessor = Cosmos3Preprocessor(embodiment="droid_lerobot")
        obs = Observation(
            images={
                "wrist_image_left": torch.zeros(3, 224, 224),
                "exterior_image_1_left": torch.zeros(3, 224, 224),
                "exterior_image_2_left": torch.zeros(3, 224, 224),
            },
            action=torch.ones(32, 10),
            state=torch.zeros(10),
        )
        result = preprocessor(obs)

        assert result["view_point"] == "concat_view"
        assert result[IMAGES].shape == (3, 336, 224)
        assert ACTION in result
        assert STATE in result

    def test_lerobot_prefix_extraction(self) -> None:
        """Test preprocessor extracts cameras from observation.images.* format."""
        preprocessor = Cosmos3Preprocessor(embodiment="droid_lerobot")
        batch = {
            "observation.images.wrist_image_left": torch.zeros(3, 224, 224),
            "observation.images.exterior_image_1_left": torch.zeros(3, 224, 224),
            "observation.images.exterior_image_2_left": torch.zeros(3, 224, 224),
            "observation.images.wrist_image_left_is_pad": torch.zeros(3, dtype=torch.bool),
        }
        result = preprocessor(batch)

        assert result["view_point"] == "concat_view"
        assert result[IMAGES].shape == (3, 336, 224)

    def test_channels_last_conversion(self) -> None:
        """Test (H, W, C) input is converted to channels-first (C, H, W)."""
        preprocessor = Cosmos3Preprocessor(embodiment="pusht")
        batch = {IMAGES: torch.zeros(224, 224, 3)}
        result = preprocessor(batch)

        assert result[IMAGES].shape == (3, 224, 224)

    def test_idempotency(self) -> None:
        """Test preprocessor is idempotent when batch is already preprocessed."""
        preprocessor = Cosmos3Preprocessor(embodiment="droid_lerobot")
        batch = {
            IMAGES: torch.zeros(3, 336, 224),
            "view_point": "concat_view",
        }
        result = preprocessor(batch)
        assert result[IMAGES].shape == (3, 336, 224)
        assert result["view_point"] == "concat_view"

    def test_missing_image_raises_key_error(self) -> None:
        """Test missing image in batch raises KeyError."""
        preprocessor = Cosmos3Preprocessor(embodiment="pusht")
        with pytest.raises(KeyError, match="No image tensor found"):
            preprocessor({"action": torch.zeros(10)})


# ============================================================================ #
# Flow Matching Output Unpacking Tests                                         #
# ============================================================================ #


@dataclass
class _MockCosmosTransformerOutput(BaseOutput):
    """Mock Cosmos3OmniTransformerOutput inheriting from BaseOutput."""

    sample: list[torch.Tensor]
    sound: list[torch.Tensor] | None = None
    action: list[torch.Tensor] | None = None


class TestFlowMatchingOutputUnpacking:
    """Tests verifying robust unpacking of transformer forward outputs in flow matching."""

    def test_unpack_base_output_dataclass(self) -> None:
        """Test unpacking Cosmos3OmniTransformerOutput / BaseOutput where sound is None.

        When sound is omitted (as in policy mode), iterating over BaseOutput produces
        only 2 keys (sample and action), which previously caused:
        ValueError: not enough values to unpack (expected 3, got 2).
        """
        pred_v = [torch.zeros(1, 16, 1, 14, 14)]
        pred_a = [torch.zeros(5, 64)]
        out = _MockCosmosTransformerOutput(sample=pred_v, sound=None, action=pred_a)

        # Confirm that plain tuple unpacking raises ValueError
        with pytest.raises(ValueError, match="not enough values to unpack"):
            _, _, _ = out

        # Confirm _unpack_transformer_output successfully extracts vision and action
        vision_out, action_out = _unpack_transformer_output(out)
        assert vision_out == pred_v
        assert action_out == pred_a

    def test_unpack_tuple_formats(self) -> None:
        """Test unpacking 3-tuple (with sound) and 2-tuple (without sound)."""
        pred_v = [torch.zeros(1, 16, 1, 14, 14)]
        pred_a = [torch.zeros(5, 64)]

        # 3-tuple: (preds_vision, preds_sound, preds_action)
        out_3 = (pred_v, None, pred_a)
        v3, a3 = _unpack_transformer_output(out_3)
        assert v3 == pred_v
        assert a3 == pred_a

        # 2-tuple: (preds_vision, preds_action)
        out_2 = (pred_v, pred_a)
        v2, a2 = _unpack_transformer_output(out_2)
        assert v2 == pred_v
        assert a2 == pred_a

    def test_unpack_dict_format(self) -> None:
        """Test unpacking dictionary format."""
        pred_v = [torch.zeros(1, 16, 1, 14, 14)]
        pred_a = [torch.zeros(5, 64)]

        out_dict = {"sample": pred_v, "action": pred_a}
        v, a = _unpack_transformer_output(out_dict)
        assert v == pred_v
        assert a == pred_a

    def test_unpack_invalid_type_raises(self) -> None:
        """Test unpacking unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="Unexpected transformer output format"):
            _unpack_transformer_output(42)

    def test_flow_matching_step_with_mock_cosmos_output(self) -> None:
        """Test flow_matching_step handles BaseOutput with sound=None without unpack error."""
        pack = {
            "text": {
                "input_ids": torch.zeros((1, 10), dtype=torch.long),
                "text_indexes": torch.arange(10),
                "und_len": 10,
            },
            "vis": {
                "vision_token_shapes": [(1, 14, 14)],
                "vision_sequence_indexes": torch.arange(10, 20),
                "vision_mse_loss_indexes": torch.arange(10, 20),
                "num_noisy_vision_tokens": 10,
                "vision_noisy_frame_indexes": [torch.zeros(1, dtype=torch.long)],
            },
            "act": {
                "action_token_shapes": [(1, 4, 1)],
                "action_sequence_indexes": torch.arange(20, 24),
                "action_mse_loss_indexes": torch.arange(20, 24),
                "num_noisy_action_tokens": 4,
                "action_noisy_frame_indexes": [torch.zeros(1, dtype=torch.long)],
            },
            "position_ids": torch.zeros((3, 24), dtype=torch.long),
            "seq_len": 24,
            "vision_keep": torch.zeros((1, 1, 1)),
            "action_keep": torch.zeros((5, 1)),
            "train_video": True,
            "train_action": True,
        }
        x0_vision = torch.zeros(1, 16, 1, 14, 14)
        x0_action = torch.zeros(5, 64)
        domain_id = torch.tensor([0], dtype=torch.long)

        pred_v = [torch.zeros(1, 16, 1, 14, 14)]
        pred_a = [torch.zeros(5, 64)]
        mock_tf = MagicMock(return_value=_MockCosmosTransformerOutput(sample=pred_v, sound=None, action=pred_a))

        loss, loss_v, loss_a = flow_matching_step(
            mock_tf,
            pack,
            x0_vision,
            x0_action,
            domain_id,
            dtype=torch.float32,
            raw_dim=2,
            action_weight=10.0,
            device="cpu",
        )
        assert isinstance(loss, torch.Tensor)
        assert isinstance(loss_v, torch.Tensor)
        assert isinstance(loss_a, torch.Tensor)


# ============================================================================ #
# Normalization Parity Tests                                                   #
# ============================================================================ #


class TestNormalization:
    """Parity tests for the (offset, scale) affine seam vs cosmos-framework."""

    def test_resolve_affine_none_is_identity(self) -> None:
        """``none`` resolves to the identity affine."""
        from physicalai.policies.cosmos3.normalization import resolve_affine

        offset, scale = resolve_affine("none", {})
        assert offset.item() == 0.0
        assert scale.item() == 1.0

    def test_resolve_affine_minmax(self) -> None:
        """Minmax maps [min, max] -> [-1, 1] via offset=(hi+lo)/2, scale=(hi-lo)/2."""
        from physicalai.policies.cosmos3.normalization import resolve_affine

        stats = {"min": torch.tensor([0.0, -4.0]), "max": torch.tensor([10.0, 4.0])}
        offset, scale = resolve_affine("minmax", stats)
        torch.testing.assert_close(offset, torch.tensor([5.0, 0.0]))
        torch.testing.assert_close(scale, torch.tensor([5.0, 4.0]))

    def test_resolve_affine_quantile_matches_cosmos(self) -> None:
        """Quantile uses q01/q99, mirroring cosmos ``resolve_action_normalization``."""
        from physicalai.policies.cosmos3.normalization import resolve_affine

        q01 = torch.tensor([-1.0, 2.0])
        q99 = torch.tensor([3.0, 6.0])
        offset, scale = resolve_affine("quantile", {"q01": q01, "q99": q99})
        torch.testing.assert_close(offset, (q99 + q01) / 2.0)
        torch.testing.assert_close(scale, (q99 - q01) / 2.0)

    def test_resolve_affine_meanstd(self) -> None:
        """Meanstd resolves to offset=mean, scale=std."""
        from physicalai.policies.cosmos3.normalization import resolve_affine

        stats = {"mean": torch.tensor([1.0, 2.0]), "std": torch.tensor([0.5, 4.0])}
        offset, scale = resolve_affine("meanstd", stats)
        torch.testing.assert_close(offset, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(scale, torch.tensor([0.5, 4.0]))

    def test_resolve_affine_scale_clamped(self) -> None:
        """Zero-width stats produce a clamped, non-zero scale (no divide-by-zero)."""
        from physicalai.policies.cosmos3.normalization import resolve_affine

        offset, scale = resolve_affine("minmax", {"min": torch.tensor([5.0]), "max": torch.tensor([5.0])})
        assert offset.item() == 5.0
        assert scale.item() > 0.0

    def test_resolve_affine_unknown_method(self) -> None:
        """An unknown method raises ValueError."""
        from physicalai.policies.cosmos3.normalization import resolve_affine

        with pytest.raises(ValueError, match="Unknown normalization method"):
            resolve_affine("bogus", {})

    def test_resolve_affine_missing_keys(self) -> None:
        """A method with missing stats keys raises KeyError."""
        from physicalai.policies.cosmos3.normalization import resolve_affine

        with pytest.raises(KeyError):
            resolve_affine("quantile", {"q01": torch.tensor([0.0])})

    def test_normalize_denormalize_roundtrip(self) -> None:
        """Quantile normalize/denormalize is a round-trip identity on the model seam."""
        config = Cosmos3Config(embodiment="pusht", chunk_size=4)
        pipe = MagicMock()
        pipe.transformer = MagicMock()
        model = Cosmos3Model.__new__(Cosmos3Model)  # avoid heavy init
        # Minimal manual setup of the affine seam.
        torch.nn.Module.__init__(model)
        model.raw_dim = 2
        model.register_buffer("norm_offset", torch.tensor([1.0, -2.0]))
        model.register_buffer("norm_scale", torch.tensor([2.0, 4.0]))

        x = torch.tensor([[3.0, 6.0], [-1.0, -6.0]])
        y = model._normalize_action(x)
        torch.testing.assert_close(y, torch.tensor([[1.0, 2.0], [-1.0, -1.0]]))
        torch.testing.assert_close(model._denormalize_action(y), x)

    def test_load_stats_file_flat(self, tmp_path: Path) -> None:
        """A flat cosmos JSON with q01/q99 resolves through load_stats_file + resolve_affine."""
        import json

        from physicalai.policies.cosmos3.normalization import load_stats_file, resolve_affine

        path = tmp_path / "stats.json"
        path.write_text(json.dumps({"q01": [-1.0, 2.0], "q99": [3.0, 6.0]}))
        stats = load_stats_file(path, "quantile")
        offset, scale = resolve_affine("quantile", stats)
        torch.testing.assert_close(offset, torch.tensor([1.0, 4.0]))
        torch.testing.assert_close(scale, torch.tensor([2.0, 2.0]))

    def test_load_stats_file_nested_quantile_rot(self, tmp_path: Path) -> None:
        """quantile_rot reads the ``global_raw`` block; quantile reads ``global``."""
        import json

        from physicalai.policies.cosmos3.normalization import load_stats_file

        payload = {
            "global": {"q01": [0.0], "q99": [2.0]},
            "global_raw": {"q01": [-5.0], "q99": [5.0]},
        }
        path = tmp_path / "nested.json"
        path.write_text(json.dumps(payload))

        rot = load_stats_file(path, "quantile_rot")
        torch.testing.assert_close(rot["q01"], torch.tensor([-5.0]))
        plain = load_stats_file(path, "quantile")
        torch.testing.assert_close(plain["q01"], torch.tensor([0.0]))

    def test_load_stats_file_missing(self, tmp_path: Path) -> None:
        """A missing stats file raises FileNotFoundError."""
        from physicalai.policies.cosmos3.normalization import load_stats_file

        with pytest.raises(FileNotFoundError):
            load_stats_file(tmp_path / "nope.json", "quantile")

    def test_load_normalizer_stats_file_shape_assert(self, tmp_path: Path) -> None:
        """An explicit stats file whose width mismatches raw_dim raises ValueError."""
        import json

        config = Cosmos3Config(embodiment="droid_lerobot", chunk_size=4)
        model = Cosmos3Model.__new__(Cosmos3Model)
        torch.nn.Module.__init__(model)
        model.config = config
        model.raw_dim = 10
        model.norm_method = "none"
        model.register_buffer("norm_offset", torch.zeros(10))
        model.register_buffer("norm_scale", torch.ones(10))

        # File provides only 2 channels; raw_dim is 10 -> mismatch.
        path = tmp_path / "stats.json"
        path.write_text(json.dumps({"q01": [-1.0, -1.0], "q99": [1.0, 1.0]}))
        with pytest.raises(ValueError, match="does not match the raw action dim"):
            model._load_normalizer_stats_file(str(path))


class TestJointPosRepresentation:
    """DROID ``joint_pos`` (8D ``[joint(7), gripper(1)]``) parity with the released checkpoint."""

    def test_droid_maps_to_joint_pos(self) -> None:
        """droid_lerobot resolves to the joint_pos action space, not ee-pose."""
        from physicalai.policies.cosmos3.representation import embodiment_normalization, resolve_action_space

        assert resolve_action_space("droid_lerobot") == "joint_pos"
        assert resolve_action_space("pusht") == "identity"
        # An explicit override wins over the embodiment default.
        assert resolve_action_space("pusht", "joint_pos") == "joint_pos"
        # joint_pos actions are raw (no normalization), matching action_normalization=None.
        assert embodiment_normalization("droid_lerobot") == "none"

    def test_droid_raw_action_dim_is_8(self) -> None:
        """The DROID raw action dim is pinned to 8 (joint_pos), overriding the diffusers default."""
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import _EMBODIMENT_TO_RAW_ACTION_DIM

        import physicalai.policies.cosmos3.model  # noqa: F401  (import applies the override)

        assert _EMBODIMENT_TO_RAW_ACTION_DIM["droid_lerobot"] == 8

    def test_gripper_flip_last_channel(self) -> None:
        """DROID is gripper-flipped; the flip inverts only the final channel as 1 - g."""
        import numpy as np

        from physicalai.policies.cosmos3.representation import embodiment_gripper_flipped, flip_gripper_last_channel

        assert embodiment_gripper_flipped("droid_lerobot") is True
        assert embodiment_gripper_flipped("pusht") is False
        action = torch.tensor([[0.1, 0.2, 0.3, 0.7, 0.4, 0.5, 0.6, 0.0]])  # [1, 8]
        flipped = flip_gripper_last_channel(action)
        assert float(flipped[0, -1]) == pytest.approx(1.0)
        # Non-gripper channels are untouched.
        np.testing.assert_allclose(flipped[0, :-1].numpy(), action[0, :-1].numpy())

    def test_predict_action_chunk_droid_inverts_gripper(self) -> None:
        """predict_action_chunk inverts the predicted gripper back to dataset convention for DROID."""
        from unittest.mock import patch

        config = Cosmos3Config(embodiment="droid_lerobot", chunk_size=4)
        pipe = MagicMock()
        # Mock predicted action chunk with gripper=0.2 in model space
        mock_actions = torch.zeros(1, 5, 64)
        mock_actions[0, :, 7] = 0.2
        pipe.return_value = MagicMock(action=mock_actions)
        pipe.transformer = MagicMock()
        pipe.transformer.device = torch.device("cpu")
        pipe.transformer.dtype = torch.float32
        pipe.transformer.config.action_dim = 64
        pipe.transformer.parameters.return_value = []
        pipe.transformer.named_parameters.return_value = []
        pipe.vae = MagicMock()
        pipe.scheduler = None

        with patch("physicalai.policies.cosmos3.model.init_domain_action_head"):
            model = Cosmos3Model(config, pipeline=pipe)
        batch = {
            IMAGES: {
                "wrist_image_left": torch.zeros(3, 224, 224),
                "exterior_image_1_left": torch.zeros(3, 224, 224),
                "exterior_image_2_left": torch.zeros(3, 224, 224),
            },
        }
        pred = model.predict_action_chunk(batch)
        # Final channel (gripper) should be 1 - 0.2 = 0.8
        assert float(pred[0, 0, 7]) == pytest.approx(0.8)
