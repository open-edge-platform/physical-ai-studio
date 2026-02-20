# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for config module."""

import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pytest
from pydantic import BaseModel

from getiaction.config import Config
from getiaction.config.instantiate import _import_class, instantiate_obj
from getiaction.config.mixin import FromConfig


# =============================================================================
# Test Fixtures
# =============================================================================


class SampleModel(FromConfig):
    """Sample model implementing FromConfig."""

    def __init__(self, hidden_size: int, num_layers: int = 3, **kwargs):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kwargs = kwargs


class SampleModelConfig(BaseModel):
    """Pydantic config for SampleModel."""

    hidden_size: int = 128
    num_layers: int = 3


@dataclasses.dataclass
class SampleModelDataclass:
    """Dataclass config for SampleModel."""

    hidden_size: int = 128
    num_layers: int = 3


class ActivationType(StrEnum):
    """Test enum."""

    RELU = "relu"
    GELU = "gelu"


@dataclass
class SimpleConfig(Config):
    """Simple serializable config."""

    hidden_size: int = 128
    num_layers: int = 3


@dataclass
class NestedConfig(Config):
    """Nested serializable config."""

    model: SimpleConfig = field(default_factory=SimpleConfig)
    learning_rate: float = 0.001


@dataclass
class ComplexConfig(Config):
    """Config with various types."""

    activation: ActivationType = ActivationType.RELU
    layers: tuple = (64, 128)
    weights: np.ndarray = field(default_factory=lambda: np.array([1.0, 2.0]))


# =============================================================================
# instantiate_obj Tests
# =============================================================================


class TestInstantiateObj:
    """Tests for instantiate_obj function."""

    def test_from_dict(self):
        """Test instantiation from dict with class_path."""
        result = instantiate_obj({"class_path": "builtins.dict", "init_args": {"key": "value"}})
        assert result == {"key": "value"}

    def test_from_dict_with_key(self):
        """Test instantiation with key extraction."""
        config = {"model": {"class_path": "builtins.dict", "init_args": {"size": 128}}}
        assert instantiate_obj(config, key="model") == {"size": 128}

    def test_nested_instantiation(self):
        """Test nested class_path instantiation."""
        config = {
            "class_path": "builtins.dict",
            "init_args": {"nested": {"class_path": "builtins.dict", "init_args": {"k": "v"}}},
        }
        assert instantiate_obj(config)["nested"] == {"k": "v"}

    def test_from_file(self, tmp_path):
        """Test instantiation from YAML file."""
        (tmp_path / "config.yaml").write_text("class_path: builtins.dict\ninit_args:\n  key: value")
        assert instantiate_obj(tmp_path / "config.yaml") == {"key": "value"}

    def test_missing_class_path_raises(self):
        """Test missing class_path raises ValueError."""
        with pytest.raises(ValueError, match="class_path"):
            instantiate_obj({"init_args": {}})

    def test_invalid_import_raises(self):
        """Test invalid class_path raises ImportError."""
        with pytest.raises(ImportError):
            _import_class("nonexistent.module.Class")


# =============================================================================
# FromConfig Mixin Tests
# =============================================================================


class TestFromConfigMixin:
    """Tests for FromConfig mixin."""

    def test_from_dict(self):
        """Test from_dict instantiation."""
        model = SampleModel.from_dict({"hidden_size": 256, "num_layers": 4})
        assert model.hidden_size == 256
        assert model.num_layers == 4

    def test_from_dict_with_key(self):
        """Test from_dict with key extraction."""
        model = SampleModel.from_dict({"model": {"hidden_size": 512, "num_layers": 6}}, key="model")
        assert model.hidden_size == 512

    def test_from_pydantic(self):
        """Test from_pydantic instantiation."""
        model = SampleModel.from_pydantic(SampleModelConfig(hidden_size=256))
        assert model.hidden_size == 256

    def test_from_dataclass(self):
        """Test from_dataclass instantiation."""
        model = SampleModel.from_dataclass(SampleModelDataclass(hidden_size=512))
        assert model.hidden_size == 512

    def test_from_yaml(self, tmp_path):
        """Test from_yaml instantiation."""
        (tmp_path / "config.yaml").write_text("hidden_size: 1024\nnum_layers: 8")
        model = SampleModel.from_yaml(tmp_path / "config.yaml")
        assert model.hidden_size == 1024

    def test_from_config_unified(self):
        """Test from_config routes correctly for different types."""
        assert SampleModel.from_config({"hidden_size": 128, "num_layers": 3}).hidden_size == 128
        assert SampleModel.from_config(SampleModelConfig()).hidden_size == 128
        assert SampleModel.from_config(SampleModelDataclass()).hidden_size == 128

    def test_recursive_parameter(self):
        """Test recursive parameter for nested structures."""

        @dataclass
        class Nested:
            size: int = 64

        @dataclass
        class Parent:
            hidden_size: int = 128
            nested: Nested = field(default_factory=Nested)

        class Model(FromConfig):
            def __init__(self, hidden_size: int, nested=None):
                self.hidden_size = hidden_size
                self.nested = nested

        parent = Parent()
        assert isinstance(Model.from_dataclass(parent, recursive=False).nested, Nested)
        assert isinstance(Model.from_dataclass(parent, recursive=True).nested, dict)


# =============================================================================
# Config Base Class Tests
# =============================================================================


class TestConfigSerialization:
    """Tests for Config base class."""

    def test_to_jsonargparse(self):
        """Test to_jsonargparse includes class_path."""
        result = SimpleConfig(hidden_size=256).to_jsonargparse()
        assert "class_path" in result
        assert result["init_args"]["hidden_size"] == 256

    def test_to_dict(self):
        """Test to_dict returns plain dict."""
        result = SimpleConfig(hidden_size=256).to_dict()
        assert "class_path" not in result
        assert result["hidden_size"] == 256

    def test_from_dict(self):
        """Test from_dict reconstructs config."""
        config = SimpleConfig.from_dict({"hidden_size": 512, "num_layers": 8})
        assert config.hidden_size == 512

    def test_from_dict_nested(self):
        """Test from_dict reconstructs nested dataclass."""
        config = NestedConfig.from_dict({"model": {"hidden_size": 256, "num_layers": 4}, "learning_rate": 0.01})
        assert isinstance(config.model, SimpleConfig)
        assert config.model.hidden_size == 256

    def test_round_trip(self):
        """Test to_dict/from_dict round-trip."""
        original = NestedConfig(model=SimpleConfig(hidden_size=512), learning_rate=0.005)
        restored = NestedConfig.from_dict(original.to_dict())
        assert restored.model.hidden_size == 512
        assert restored.learning_rate == 0.005

    def test_type_conversions(self):
        """Test enum, tuple, numpy conversions."""
        result = ComplexConfig(
            activation=ActivationType.GELU,
            layers=(32, 64),
            weights=np.array([[1.0, 2.0]]),
        ).to_jsonargparse()
        assert result["init_args"]["activation"] == "gelu"
        assert result["init_args"]["layers"] == [32, 64]
        assert result["init_args"]["weights"] == [[1.0, 2.0]]


class TestConfigSaveLoad:
    """Tests for Config save/load."""

    def test_save_load_jsonargparse(self, tmp_path):
        """Test save/load with jsonargparse format."""
        path = tmp_path / "config.yaml"
        SimpleConfig(hidden_size=256).save(path)
        assert SimpleConfig.load(path).hidden_size == 256

    def test_save_load_dict_format(self, tmp_path):
        """Test save/load with dict format."""
        path = tmp_path / "config.yaml"
        SimpleConfig(hidden_size=512).save(path, format="dict")
        assert SimpleConfig.load(path).hidden_size == 512

    def test_invalid_extension_raises(self, tmp_path):
        """Test invalid extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            SimpleConfig().save(tmp_path / "config.json")

    def test_not_dataclass_raises(self):
        """Test non-dataclass raises TypeError."""

        class NotDataclass(Config):
            pass

        with pytest.raises(TypeError, match="must be a dataclass"):
            NotDataclass().to_dict()
