# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mixin_torch module."""

import dataclasses
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from getiaction.export import (
    FromCheckpoint,
    Export,
)

from getiaction.export.mixin_torch import CONFIG_KEY
from getiaction.export.mixin_export import ExportBackend, _serialize_model_config


# Test enums
class ActivationType(StrEnum):
    """Test enum for activation types."""

    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"


# Test dataclasses
@dataclass
class SimpleConfig:
    """Simple configuration for testing."""

    hidden_size: int = 128
    num_layers: int = 3
    activation: str = "relu"


@dataclass
class NestedConfig:
    """Nested configuration for testing."""

    model_config: SimpleConfig
    learning_rate: float = 0.001
    batch_size: int = 32


@dataclass
class ComplexConfig:
    """Complex configuration with various data types."""

    name: str = "test_model"
    hidden_size: int = 256
    activation: ActivationType = ActivationType.RELU
    layers: tuple = (64, 128, 256)
    weights: np.ndarray = dataclasses.field(default_factory=lambda: np.array([1.0, 2.0, 3.0]))
    nested: SimpleConfig = dataclasses.field(default_factory=SimpleConfig)
    metadata: dict = dataclasses.field(default_factory=lambda: {"version": "1.0"})


@dataclass
class DictWithDataclassConfig:
    """Configuration with dict containing dataclasses."""

    name: str = "dict_test"
    models: dict = dataclasses.field(default_factory=lambda: {"encoder": SimpleConfig(), "decoder": SimpleConfig()})


@dataclass
class ComplexRoundTripConfig:
    """Complex configuration for round-trip testing."""

    simple: SimpleConfig
    activation: ActivationType
    layers: tuple
    metadata: dict


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


class TestSerializeModelConfig:
    """Tests for _serialize_model_config function."""

    def test_simple_config(self):
        """Test serialization of simple configuration."""
        config = SimpleConfig(hidden_size=256, num_layers=5, activation="gelu")
        result = _serialize_model_config(config)

        assert "class_path" in result
        assert "init_args" in result
        assert result["class_path"] == f"{SimpleConfig.__module__}.{SimpleConfig.__qualname__}"
        assert result["init_args"]["hidden_size"] == 256
        assert result["init_args"]["num_layers"] == 5
        assert result["init_args"]["activation"] == "gelu"

    def test_nested_config(self):
        """Test serialization of nested configuration."""
        simple = SimpleConfig(hidden_size=128)
        config = NestedConfig(model_config=simple, learning_rate=0.0001)
        result = _serialize_model_config(config)

        assert result["class_path"] == f"{NestedConfig.__module__}.{NestedConfig.__qualname__}"
        assert "model_config" in result["init_args"]
        assert isinstance(result["init_args"]["model_config"], dict)
        assert "class_path" in result["init_args"]["model_config"]
        assert result["init_args"]["model_config"]["init_args"]["hidden_size"] == 128
        assert result["init_args"]["learning_rate"] == 0.0001

    def test_str_enum_conversion(self):
        """Test that StrEnum values are converted to strings."""
        config = ComplexConfig(activation=ActivationType.GELU)
        result = _serialize_model_config(config)

        assert result["init_args"]["activation"] == "gelu"
        assert isinstance(result["init_args"]["activation"], str)

    def test_numpy_array_conversion(self):
        """Test that numpy arrays are converted to lists."""
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = ComplexConfig(weights=weights)
        result = _serialize_model_config(config)

        assert isinstance(result["init_args"]["weights"], list)
        assert result["init_args"]["weights"] == [[1.0, 2.0], [3.0, 4.0]]

    def test_tuple_conversion(self):
        """Test that tuples are converted to lists."""
        config = ComplexConfig(layers=(32, 64, 128))
        result = _serialize_model_config(config)

        assert isinstance(result["init_args"]["layers"], list)
        assert result["init_args"]["layers"] == [32, 64, 128]

    def test_dict_with_dataclass(self):
        """Test serialization of dict containing dataclasses."""
        encoder = SimpleConfig(hidden_size=256)
        decoder = SimpleConfig(hidden_size=512)
        config = DictWithDataclassConfig(models={"encoder": encoder, "decoder": decoder})
        result = _serialize_model_config(config)

        assert "models" in result["init_args"]
        assert "encoder" in result["init_args"]["models"]
        assert "decoder" in result["init_args"]["models"]
        assert "class_path" in result["init_args"]["models"]["encoder"]
        assert result["init_args"]["models"]["encoder"]["init_args"]["hidden_size"] == 256
        assert result["init_args"]["models"]["decoder"]["init_args"]["hidden_size"] == 512

    def test_dict_with_str_enum_keys(self):
        """Test that StrEnum keys in dicts are converted to strings."""
        config = ComplexConfig(metadata={ActivationType.RELU: "relu_config"})
        result = _serialize_model_config(config)

        assert "relu" in result["init_args"]["metadata"]
        assert result["init_args"]["metadata"]["relu"] == "relu_config"

    def test_complex_config_all_types(self):
        """Test serialization of complex config with multiple data types."""
        simple = SimpleConfig(hidden_size=128)
        config = ComplexConfig(
            name="complex_model",
            hidden_size=512,
            activation=ActivationType.TANH,
            layers=(64, 128, 256, 512),
            weights=np.array([0.1, 0.2, 0.3]),
            nested=simple,
            metadata={"version": "2.0", "author": "test"},
        )
        result = _serialize_model_config(config)

        assert result["init_args"]["name"] == "complex_model"
        assert result["init_args"]["hidden_size"] == 512
        assert result["init_args"]["activation"] == "tanh"
        assert result["init_args"]["layers"] == [64, 128, 256, 512]
        assert result["init_args"]["weights"] == [0.1, 0.2, 0.3]
        assert "class_path" in result["init_args"]["nested"]
        assert result["init_args"]["metadata"] == {"version": "2.0", "author": "test"}


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
            model.export(ExportBackend.TORCH, checkpoint_path)

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
        config_dict = _serialize_model_config(config)
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
        config_dict = _serialize_model_config(config)
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
            CONFIG_KEY: yaml.dump(_serialize_model_config(config)),
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
            def from_dataclass(cls, config: SimpleConfig):
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

            assert loaded_model.config.simple.hidden_size == 256
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


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_config(self):
        """Test serialization with minimal config."""

        @dataclass
        class EmptyConfig:
            pass

        config = EmptyConfig()
        result = _serialize_model_config(config)

        assert "class_path" in result
        assert "init_args" in result
        assert result["init_args"] == {}

    def test_config_with_none_values(self):
        """Test serialization with None values."""

        @dataclass
        class ConfigWithNone:
            value: int | None = None

        config = ConfigWithNone(value=None)
        result = _serialize_model_config(config)

        assert result["init_args"]["value"] is None

    def test_deeply_nested_config(self):
        """Test serialization of deeply nested configurations."""

        @dataclass
        class Level3Config:
            value: int = 3

        @dataclass
        class Level2Config:
            level3: Level3Config = dataclasses.field(default_factory=Level3Config)
            value: int = 2

        @dataclass
        class Level1Config:
            level2: Level2Config = dataclasses.field(default_factory=Level2Config)
            value: int = 1

        config = Level1Config()
        result = _serialize_model_config(config)

        assert result["init_args"]["value"] == 1
        assert result["init_args"]["level2"]["init_args"]["value"] == 2
        assert result["init_args"]["level2"]["init_args"]["level3"]["init_args"]["value"] == 3

    def test_numpy_scalar(self):
        """Test serialization of numpy scalar values."""

        @dataclass
        class ConfigWithNumpyScalar:
            value: float = 1.0

        config = ConfigWithNumpyScalar(value=np.float32(3.14))
        result = _serialize_model_config(config)

        # Numpy scalars should be preserved as-is (not converted)
        assert isinstance(result["init_args"]["value"], (float, np.floating))

    def test_multidimensional_numpy_array(self):
        """Test serialization of multi-dimensional numpy arrays."""

        @dataclass
        class ConfigWithMultiDimArray:
            tensor: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((2, 3, 4)))

        config = ConfigWithMultiDimArray(tensor=np.ones((2, 3, 4)))
        result = _serialize_model_config(config)

        assert isinstance(result["init_args"]["tensor"], list)
        assert len(result["init_args"]["tensor"]) == 2
        assert len(result["init_args"]["tensor"][0]) == 3
        assert len(result["init_args"]["tensor"][0][0]) == 4
