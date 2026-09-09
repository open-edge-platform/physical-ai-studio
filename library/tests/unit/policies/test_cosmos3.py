# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Cosmos3 policy.

Fast, self-contained tests with no external network calls or heavy model downloads.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from physicalai.config import Config
from physicalai.data.observation import ACTION, IMAGES, Observation, STATE
from physicalai.policies import Cosmos3, Cosmos3Config, Cosmos3Model, get_physicalai_policy_class, get_policy

# ============================================================================ #
# Configuration Tests                                                          #
# ============================================================================ #


class TestCosmos3Config:
    """Tests for Cosmos3Config dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = Cosmos3Config()
        assert config.pretrained_model_name_or_path == "nvidia/Cosmos3-Edge"
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
        assert config.domain == "pusht"
        assert config.dtype == "bfloat16"
        assert config.optimizer_lr == 1e-4

    def test_full_mode_defaults(self) -> None:
        """Test mode='full' default head_lr_mult is 10.0."""
        config = Cosmos3Config(mode="full")
        assert config.mode == "full"
        assert config.head_lr_mult == 10.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = Cosmos3Config(
            mode="full",
            paradigm="joint",
            head_lr_mult=5.0,
            chunk_size=16,
            n_action_steps=16,
            domain="droid_lerobot",
        )
        assert config.mode == "full"
        assert config.paradigm == "joint"
        assert config.head_lr_mult == 5.0
        assert config.chunk_size == 16
        assert config.n_action_steps == 16
        assert config.domain == "droid_lerobot"

    def test_n_action_steps_validation(self) -> None:
        """Test n_action_steps cannot exceed chunk_size."""
        with pytest.raises(ValueError, match="cannot exceed chunk_size"):
            Cosmos3Config(chunk_size=16, n_action_steps=32)

    def test_invalid_mode(self) -> None:
        """Test invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            Cosmos3Config(mode="invalid")  # type: ignore[arg-type]

    def test_invalid_paradigm(self) -> None:
        """Test invalid paradigm raises ValueError."""
        with pytest.raises(ValueError, match="Invalid paradigm"):
            Cosmos3Config(paradigm="invalid")  # type: ignore[arg-type]

    def test_invalid_resolution_tier(self) -> None:
        """Test invalid resolution_tier raises ValueError."""
        with pytest.raises(ValueError, match="Invalid resolution_tier"):
            Cosmos3Config(resolution_tier=128)  # type: ignore[arg-type]

    def test_invalid_dtype(self) -> None:
        """Test invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            Cosmos3Config(dtype="int8")  # type: ignore[arg-type]

    def test_serialization(self) -> None:
        """Test to_dict and from_dict round-trip."""
        config = Cosmos3Config(rank=64, chunk_size=16, n_action_steps=16)
        assert isinstance(config, Config)
        cfg_dict = config.to_dict()
        assert cfg_dict["rank"] == 64
        assert cfg_dict["chunk_size"] == 16

        restored = Cosmos3Config.from_dict(cfg_dict)
        assert restored.rank == 64
        assert restored.chunk_size == 16

    def test_frozen_dataclass(self) -> None:
        """Test config dataclass is immutable."""
        config = Cosmos3Config()
        with pytest.raises(AttributeError):
            config.chunk_size = 10  # type: ignore[misc]


# ============================================================================ #
# Policy Tests                                                                 #
# ============================================================================ #


class TestCosmos3Policy:
    """Tests for Cosmos3 Lightning policy wrapper."""

    def test_lazy_initialization(self) -> None:
        """Test policy initialization does not eagerly instantiate model."""
        policy = Cosmos3()
        assert policy.model is None

    def test_hyperparameters_saved(self) -> None:
        """Test hyperparameters and config dictionary are saved."""
        policy = Cosmos3(chunk_size=16, n_action_steps=16, mode="peft")
        assert policy.hparams.chunk_size == 16
        assert policy.hparams.mode == "peft"
        assert "config" in policy.hparams
        assert policy.hparams["config"]["chunk_size"] == 16

    def test_from_config(self) -> None:
        """Test instantiation via from_config classmethod."""
        config = Cosmos3Config(chunk_size=16, n_action_steps=16)
        policy = Cosmos3.from_config(config)
        assert policy.model is None
        assert policy.config.chunk_size == 16

    def test_methods_raise_without_model(self) -> None:
        """Test forward and predict_action_chunk raise before setup()."""
        policy = Cosmos3()
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

        policy = get_policy("cosmos3")
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
        config = Cosmos3Config(chunk_size=32)
        pipe = self._create_mock_pipeline()
        model = Cosmos3Model(config, pipeline=pipe)

        assert model.reward_delta_indices is None
        assert model.action_delta_indices == list(range(32))
        assert model.observation_delta_indices == list(range(33))

    def test_set_dataset_stats(self) -> None:
        """Test updating normalization bounds via dataset stats."""
        config = Cosmos3Config(chunk_size=16, domain="pusht")
        pipe = self._create_mock_pipeline()
        model = Cosmos3Model(config, pipeline=pipe)

        stats = {
            "action": {
                "min": [10.0, 20.0],
                "max": [50.0, 80.0],
            }
        }
        model.set_dataset_stats(stats)
        assert model.raw_dim == 2
        torch.testing.assert_close(model.a_min, torch.tensor([10.0, 20.0]))
        torch.testing.assert_close(model.a_max, torch.tensor([50.0, 80.0]))

    def test_predict_action_chunk_mocked(self) -> None:
        """Test predict_action_chunk flow with mock pipeline output."""
        config = Cosmos3Config(chunk_size=4, domain="pusht")
        pipe = self._create_mock_pipeline()

        # Mock result: action tensor of shape [1, chunk_size + 1, 64]
        mock_actions = torch.zeros(1, 5, 64)
        # Action tokens 1..4 in normalized [-1, 1] space
        mock_actions[0, 1:5, :2] = 0.0  # midpoint maps to (min + max)/2 = (0 + 10)/2 = 5.0
        pipe.return_value = MagicMock(action=mock_actions)

        model = Cosmos3Model(config, pipeline=pipe)
        model.a_min = torch.tensor([0.0, 0.0])
        model.a_max = torch.tensor([10.0, 10.0])

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

        config = Cosmos3Config(chunk_size=4, domain="pusht")
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
        """Test policy on_save_checkpoint removes frozen parameters."""
        config = Cosmos3Config(chunk_size=4)
        pipe = self._create_mock_pipeline()
        policy = Cosmos3(chunk_size=4, pipeline=pipe)

        # Mock state_dict with frozen and trainable keys
        checkpoint = {
            "state_dict": {
                "model.transformer.to_q.weight": torch.randn(2, 2),
                "model.transformer.frozen_backbone.weight": torch.randn(10, 10),
                "model.a_min": torch.tensor([-1.0, -1.0]),
                "model.a_max": torch.tensor([1.0, 1.0]),
                "model.domain_id": torch.tensor([4]),
            }
        }
        policy.on_save_checkpoint(checkpoint)

        saved_keys = checkpoint["state_dict"].keys()
        assert "model.a_min" in saved_keys
        assert "model.a_max" in saved_keys
        assert "model.domain_id" in saved_keys
        assert "model.transformer.frozen_backbone.weight" not in saved_keys
