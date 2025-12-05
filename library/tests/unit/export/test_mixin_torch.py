# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mixin_torch module."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from getiaction.config import Config
from getiaction.export import FromCheckpoint

from getiaction.export.mixin_torch import CONFIG_KEY


@dataclass
class DummyModelConfig(Config):
    """Dummy model configuration for testing."""

    hidden_dim: int = 128
    num_layers: int = 2


class DummyModel(torch.nn.Module):
    """Dummy model for testing."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.num_layers = num_layers

    @classmethod
    def from_config(cls, config: DummyModelConfig) -> "DummyModel":
        """Create model from config."""
        return cls(hidden_dim=config.hidden_dim, num_layers=config.num_layers)


class DummyPolicy(FromCheckpoint, torch.nn.Module):
    """Dummy policy for testing FromCheckpoint mixin."""

    model_type = DummyModel
    model_config_type = DummyModelConfig

    def __init__(self, model: DummyModel, extra_param: str = "default"):
        super().__init__()
        self.model = model
        self.extra_param = extra_param


class TestFromCheckpoint:
    """Test FromCheckpoint mixin."""

    @pytest.fixture
    def model_config(self) -> DummyModelConfig:
        """Create dummy model config."""
        return DummyModelConfig(hidden_dim=256, num_layers=3)

    @pytest.fixture
    def model(self, model_config: DummyModelConfig) -> DummyModel:
        """Create dummy model."""
        return DummyModel.from_config(model_config)

    @pytest.fixture
    def policy(self, model: DummyModel) -> DummyPolicy:
        """Create dummy policy."""
        return DummyPolicy(model=model, extra_param="test_value")

    @pytest.fixture
    def checkpoint_path(self, policy: DummyPolicy, model_config: DummyModelConfig, tmp_path: Path) -> Path:
        """Create checkpoint file."""
        checkpoint_file = tmp_path / "test_checkpoint.ckpt"

        checkpoint = {
            CONFIG_KEY: model_config.to_dict(),
            "state_dict": policy.state_dict(),
        }
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save(checkpoint, checkpoint_file)
        return checkpoint_file

    def test_load_from_checkpoint_basic(self, checkpoint_path: Path):
        """Test basic checkpoint loading."""
        loaded_policy = DummyPolicy.load_from_checkpoint(str(checkpoint_path))

        assert isinstance(loaded_policy, DummyPolicy)
        assert isinstance(loaded_policy.model, DummyModel)
        assert loaded_policy.model.linear.in_features == 256
        assert loaded_policy.model.num_layers == 3

    def test_load_from_checkpoint_with_kwargs(self, checkpoint_path: Path):
        """Test checkpoint loading with additional kwargs."""
        loaded_policy = DummyPolicy.load_from_checkpoint(
            str(checkpoint_path),
            extra_param="custom_value",
        )

        assert loaded_policy.extra_param == "custom_value"

    def test_load_from_checkpoint_map_location_cpu(self, checkpoint_path: Path):
        """Test checkpoint loading with CPU device."""
        loaded_policy = DummyPolicy.load_from_checkpoint(
            str(checkpoint_path),
            map_location="cpu",
        )

        assert isinstance(loaded_policy, DummyPolicy)
        for param in loaded_policy.parameters():
            assert param.device.type == "cpu"

    def test_load_from_checkpoint_map_location_device(self, checkpoint_path: Path):
        """Test checkpoint loading with torch.device."""
        device = torch.device("cpu")
        loaded_policy = DummyPolicy.load_from_checkpoint(
            str(checkpoint_path),
            map_location=device,
        )

        assert isinstance(loaded_policy, DummyPolicy)
        for param in loaded_policy.parameters():
            assert param.device == device

    def test_load_from_checkpoint_preserves_weights(self, checkpoint_path: Path, policy: DummyPolicy):
        """Test that checkpoint loading preserves model weights."""
        loaded_policy = DummyPolicy.load_from_checkpoint(str(checkpoint_path))

        # Compare state dicts
        original_state = policy.state_dict()
        loaded_state = loaded_policy.state_dict()

        assert set(original_state.keys()) == set(loaded_state.keys())
        for key in original_state.keys():
            assert torch.allclose(original_state[key], loaded_state[key])

    def test_load_from_checkpoint_missing_config_key(self, tmp_path: Path, model: DummyModel):
        """Test error when checkpoint missing config key."""
        checkpoint_file = tmp_path / "invalid_checkpoint.ckpt"

        # Save checkpoint without CONFIG_KEY
        checkpoint = {
            "state_dict": DummyPolicy(model=model).state_dict(),
        }
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save(checkpoint, checkpoint_file)

        with pytest.raises(KeyError, match=f"Checkpoint missing '{CONFIG_KEY}'"):
            DummyPolicy.load_from_checkpoint(str(checkpoint_file))

    def test_load_from_checkpoint_nonexistent_file(self):
        """Test error when checkpoint file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            DummyPolicy.load_from_checkpoint("nonexistent_checkpoint.ckpt")

    def test_load_from_checkpoint_corrupted_file(self, tmp_path: Path):
        """Test error when checkpoint file is corrupted."""
        checkpoint_file = tmp_path / "corrupted.ckpt"
        checkpoint_file.write_text("not a valid checkpoint")

        with pytest.raises((RuntimeError, pickle.UnpicklingError)):
            DummyPolicy.load_from_checkpoint(str(checkpoint_file))

    def test_load_from_checkpoint_returns_correct_type(self, checkpoint_path: Path):
        """Test that load_from_checkpoint returns correct type."""
        loaded_policy = DummyPolicy.load_from_checkpoint(str(checkpoint_path))

        assert type(loaded_policy) is DummyPolicy
        assert hasattr(loaded_policy, "model")
        assert hasattr(loaded_policy, "extra_param")
