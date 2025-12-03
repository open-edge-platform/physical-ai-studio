# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mixin_torch module."""

import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pytest
import torch
import yaml

from getiaction.config import Config
from getiaction.export import (
    FromCheckpoint,
    Export,
)

from getiaction.export.mixin_torch import CONFIG_KEY
from getiaction.export.mixin_export import ExportBackend


# Test enums
class ActivationType(StrEnum):
    """Test enum for activation types."""

    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"


# Test dataclasses
@dataclass
class SimpleConfig(Config):
    """Simple configuration for testing."""

    hidden_size: int = 128
    num_layers: int = 3
    activation: str = "relu"


@dataclass
class ComplexRoundTripConfig(Config):
    """Complex configuration for round-trip testing."""

    simple: SimpleConfig = field(default_factory=SimpleConfig)
    activation: ActivationType = ActivationType.RELU
    layers: tuple = (64, 128, 256)
    metadata: dict = field(default_factory=lambda: {"version": "1.0"})


# Test models
class SimpleModel(torch.nn.Module, FromCheckpoint):
    """Simple PyTorch model for testing."""

    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.config = config
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        return self.linear(x)

    @classmethod
    def from_dataclass(cls, config: SimpleConfig):
        """Create model from dataclass config."""
        return cls(config)


class ModelWithToTorch(Export, torch.nn.Module):
    """Model implementing ToTorch mixin."""

    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.config = config
        self.model = SimpleModel(config)


class ModelWithFromCheckpoint(FromCheckpoint, torch.nn.Module):
    """Model implementing FromCheckpoint mixin."""

    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.config = config
        self.model = SimpleModel(config)

    @classmethod
    def from_dataclass(cls, config: SimpleConfig):
        """Create model from dataclass config."""
        return cls(config)


class SimplePolicy(Export, torch.nn.Module):
    """Model implementing both ToTorch and FromCheckpoint mixins."""

    def __init__(self, model):
        super().__init__()
        self.model = model


class TestToTorch:
    """Tests for ToTorch mixin."""

    def test_to_torch_basic(self):
        """Test basic to_torch functionality."""
        config = SimpleConfig(hidden_size=64, num_layers=2)
        model = ModelWithToTorch(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            model.to_torch(checkpoint_path)

            assert checkpoint_path.exists()

            # Load and verify checkpoint
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)  # nosemgrep
            assert CONFIG_KEY in state_dict
            assert isinstance(state_dict[CONFIG_KEY], str)

    def test_to_torch_config_serialization(self):
        """Test that config is properly serialized in checkpoint."""
        config = SimpleConfig(hidden_size=256, num_layers=4, activation="gelu")
        model = ModelWithToTorch(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            model.to_torch(checkpoint_path)

            # Load checkpoint and parse config
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)  # nosemgrep
            config_yaml = state_dict[CONFIG_KEY]
            loaded_config = yaml.safe_load(config_yaml)

            assert loaded_config["class_path"] == f"{SimpleConfig.__module__}.{SimpleConfig.__qualname__}"
            assert loaded_config["init_args"]["hidden_size"] == 256
            assert loaded_config["init_args"]["num_layers"] == 4
            assert loaded_config["init_args"]["activation"] == "gelu"

    def test_to_torch_model_weights(self):
        """Test that model weights are saved correctly."""
        config = SimpleConfig(hidden_size=64)
        model = ModelWithToTorch(config)

        # Set specific weights
        with torch.no_grad():
            model.model.linear.weight.fill_(1.0)
            model.model.linear.bias.fill_(0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            model.to_torch(checkpoint_path)

            # Load and verify weights
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)  # nosemgrep
            state_dict.pop(CONFIG_KEY)  # Remove config key

            assert "linear.weight" in state_dict
            assert "linear.bias" in state_dict
            assert torch.all(state_dict["linear.weight"] == 1.0)
            assert torch.all(state_dict["linear.bias"] == 0.5)

    def test_to_torch_without_config(self):
        """Test to_torch with model that has no config attribute."""

        class ModelWithoutConfig(Export):
            def __init__(self):
                self.model = torch.nn.Linear(10, 10)

        model = ModelWithoutConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            model.export(checkpoint_path, ExportBackend.TORCH)

            # Should save successfully with empty config
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)  # nosemgrep
            assert CONFIG_KEY in state_dict
            config_yaml = state_dict[CONFIG_KEY]
            loaded_config = yaml.safe_load(config_yaml)
            assert loaded_config == {}

    def test_to_torch_with_string_path(self):
        """Test to_torch with string path instead of PathLike."""
        config = SimpleConfig()
        model = ModelWithToTorch(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = str(Path(tmpdir) / "model.pt")
            model.to_torch(checkpoint_path)

            assert Path(checkpoint_path).exists()


class TestFromCheckpoint:
    """Tests for FromCheckpoint mixin."""

    def test_from_snapshot_with_path(self):
        """Test loading from checkpoint file path."""
        config = SimpleConfig(hidden_size=128, num_layers=3)
        policy = SimplePolicy(SimpleModel(config))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            policy.to_torch(checkpoint_path)

            # Load from path
            loaded_model = SimpleModel.load_from_checkpoint(checkpoint_path)
            loaded_policy = SimplePolicy(config)
            loaded_policy.model = loaded_model

            assert loaded_policy.model.config.hidden_size == 128
            assert loaded_policy.model.config.num_layers == 3
            assert loaded_policy.model.config.activation == "relu"

    def test_from_snapshot_with_string_path(self):
        """Test loading from checkpoint with string path."""
        config = SimpleConfig(hidden_size=256)
        policy = SimplePolicy(SimpleModel(config))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = str(Path(tmpdir) / "model.pt")
            policy.to_torch(checkpoint_path)

            # Load from string path
            loaded_model = SimpleModel.load_from_checkpoint(checkpoint_path)

            assert loaded_model.config.hidden_size == 256

    def test_from_snapshot_with_dict(self):
        """Test loading from snapshot dictionary."""
        config = SimpleConfig(hidden_size=64)
        policy = SimplePolicy(SimpleModel(config))

        # Create a fake state dict
        state_dict = policy.model.state_dict()
        config_dict = config.to_jsonargparse()
        state_dict[CONFIG_KEY] = yaml.dump(config_dict)

        # Load from dict
        loaded_model = SimpleModel.load_from_checkpoint(state_dict)

        assert loaded_model.config.hidden_size == 64
        assert loaded_model.config.num_layers == 3

    def test_from_snapshot_preserves_config(self):
        """Test that all config parameters are preserved."""
        config = SimpleConfig(hidden_size=512, num_layers=6, activation="tanh")
        policy = SimplePolicy(SimpleModel(config))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            policy.to_torch(checkpoint_path)

            loaded_model = SimpleModel.load_from_checkpoint(checkpoint_path)

            assert loaded_model.config.hidden_size == config.hidden_size
            assert loaded_model.config.num_layers == config.num_layers
            assert loaded_model.config.activation == config.activation

    def test_from_snapshot_without_from_dataclass(self):
        """Test that NotImplementedError is raised when from_dataclass is missing."""

        class ModelWithoutFromDataclass(FromCheckpoint):
            def __init__(self, config: SimpleConfig):
                self.config = config
                self.model = SimpleModel(config)

        config = SimpleConfig()
        config_dict = config.to_jsonargparse()
        state_dict = {CONFIG_KEY: yaml.dump(config_dict)}

        with pytest.raises(NotImplementedError, match="from_dataclass"):
            ModelWithoutFromDataclass.load_from_checkpoint(state_dict)

    def test_from_snapshot_removes_config_key(self):
        """Test that config key is removed from copied state_dict."""
        config = SimpleConfig(hidden_size=128)

        # Create state dict with config key
        original_state_dict = {
            "linear.weight": torch.randn(128, 128),
            "linear.bias": torch.randn(128),
            CONFIG_KEY: yaml.dump(config.to_jsonargparse()),
        }

        # Verify config key is present
        assert CONFIG_KEY in original_state_dict

        # Load model (uses copy internally, so original should not be modified)
        SimpleModel.load_from_checkpoint(original_state_dict)

        # Original dict should still have the config key (since from_snapshot uses copy)
        assert CONFIG_KEY in original_state_dict


class TestRoundTrip:
    """Test round-trip save and load operations."""

    def test_simple_round_trip(self):
        """Test saving and loading preserves model state."""
        config = SimpleConfig(hidden_size=64, num_layers=2)
        original_policy = SimplePolicy(SimpleModel(config))

        # Set specific weights
        with torch.no_grad():
            original_policy.model.linear.weight.fill_(2.0)
            original_policy.model.linear.bias.fill_(0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            original_policy.to_torch(checkpoint_path)

            # Load model
            loaded_model = SimpleModel.load_from_checkpoint(checkpoint_path)

            # Load weights manually to verify
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)  # nosemgrep
            state_dict.pop(CONFIG_KEY)
            loaded_model.load_state_dict(state_dict)

            # Verify config
            assert loaded_model.config.hidden_size == original_policy.model.config.hidden_size
            assert loaded_model.config.num_layers == original_policy.model.config.num_layers

            # Verify weights
            assert torch.all(loaded_model.linear.weight == 2.0)
            assert torch.all(loaded_model.linear.bias == 0.1)

    def test_complex_config_round_trip(self):
        """Test round-trip with complex configuration."""

        class ComplexTestModel(torch.nn.Module, FromCheckpoint):
            def __init__(self, config: ComplexRoundTripConfig):
                super().__init__()
                self.config = config
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

            @classmethod
            def from_dataclass(cls, config: ComplexRoundTripConfig):
                """Create model from dataclass config."""
                return cls(config)


        class ComplexModel(Export):
            def __init__(self, config: ComplexRoundTripConfig):
                self.config = config
                self.model = ComplexTestModel(config)

            @classmethod
            def from_dataclass(cls, config: ComplexRoundTripConfig):
                return cls(config)

        config = SimpleConfig(hidden_size=256)
        config = ComplexRoundTripConfig(
            simple=config,
            activation=ActivationType.GELU,
            layers=(64, 128, 256),
            metadata={"version": "1.0"},
        )

        original_model = ComplexModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            original_model.to_torch(checkpoint_path)

            loaded_model = ComplexTestModel.load_from_checkpoint(checkpoint_path)

            # Note: nested dataclasses are loaded as dicts since to_jsonargparse()
            # only adds class_path at the top level. The init_args are plain dicts.
            assert loaded_model.config.simple["hidden_size"] == 256
            assert loaded_model.config.activation == "gelu"  # StrEnum converted to str
            assert loaded_model.config.layers == [64, 128, 256]  # tuple converted to list
            assert loaded_model.config.metadata == {"version": "1.0"}

    def test_multiple_round_trips(self):
        """Test multiple save/load cycles preserve model state."""
        config = SimpleConfig(hidden_size=32)
        policy = SimplePolicy(SimpleModel(config))

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                checkpoint_path = Path(tmpdir) / f"model_{i}.pt"
                policy.to_torch(checkpoint_path)
                model = SimpleModel.load_from_checkpoint(checkpoint_path)

                # Config should remain consistent
                assert model.config.hidden_size == 32
                assert model.config.num_layers == 3
                assert model.config.activation == "relu"
