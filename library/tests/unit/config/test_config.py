# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for config module functionality."""

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from getiaction.config.instantiate import (
    _import_class,
    instantiate_obj,
    instantiate_obj_from_dataclass,
    instantiate_obj_from_dict,
    instantiate_obj_from_file,
    instantiate_obj_from_pydantic,
)
from getiaction.config.mixin import FromConfig


# Test classes
class TestModel(FromConfig):
    """Test model for configuration testing."""

    def __init__(self, hidden_size: int, num_layers: int, activation: str = "relu", **kwargs):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.kwargs = kwargs

    def __eq__(self, other):
        if not isinstance(other, TestModel):
            return False
        return (
            self.hidden_size == other.hidden_size
            and self.num_layers == other.num_layers
            and self.activation == other.activation
            and self.kwargs == other.kwargs
        )


class TestNestedModel(FromConfig):
    """Test model with nested configuration."""

    def __init__(self, name: str, model: TestModel, metadata: dict[str, Any] | None = None):
        self.name = name
        self.model = model
        self.metadata = metadata or {}

    def __eq__(self, other):
        if not isinstance(other, TestNestedModel):
            return False
        return (
            self.name == other.name
            and self.model == other.model
            and self.metadata == other.metadata
        )


# Pydantic models
class TestModelConfig(BaseModel):
    """Pydantic model for testing."""
    hidden_size: int = 128
    num_layers: int = 3
    activation: str = "relu"


class TestNestedModelConfig(BaseModel):
    """Pydantic model with nested Pydantic model attribute."""
    name: str = "test_model"
    model_params: TestModelConfig = TestModelConfig(hidden_size=256, num_layers=4, activation="swish")
    metadata: dict[str, Any] = {"version": "1.0"}

    class Config:
        arbitrary_types_allowed = True


# Dataclasses
@dataclasses.dataclass
class TestModelDataclass:
    """Dataclass for testing."""
    hidden_size: int = 128
    num_layers: int = 3
    activation: str = "relu"


@dataclasses.dataclass
class TestNestedModelDataclass:
    """Dataclass with nested dataclass attribute."""
    name: str = "test_model"
    model_params: TestModelDataclass = dataclasses.field(default_factory=lambda: TestModelDataclass(
        hidden_size=256, num_layers=4, activation="swish"
    ))
    metadata: dict[str, Any] = dataclasses.field(default_factory=lambda: {"version": "1.0"})


class TestImportClass:
    """Test the _import_class helper function."""

    def test_import_builtin_class(self):
        """Test importing built-in classes."""
        cls = _import_class("builtins.dict")
        assert cls is dict

    def test_import_standard_library_class(self):
        """Test importing standard library classes."""
        cls = _import_class("pathlib.Path")
        assert cls is Path

    def test_import_invalid_module(self):
        """Test importing from non-existent module."""
        with pytest.raises(ImportError, match="Cannot import 'nonexistent.module.Class'"):
            _import_class("nonexistent.module.Class")

    def test_import_invalid_class(self):
        """Test importing non-existent class from valid module."""
        with pytest.raises(ImportError, match="Cannot import 'pathlib.NonExistentClass'"):
            _import_class("pathlib.NonExistentClass")


class TestInstantiateObjFromDict:
    """Test the instantiate_obj_from_dict function."""

    def test_instantiate_simple_class(self):
        """Test instantiating a simple class with init_args."""
        config = {
            "class_path": "builtins.dict",
            "init_args": {"key1": "value1", "key2": "value2"}
        }
        result = instantiate_obj_from_dict(config)
        assert isinstance(result, dict)
        assert result == {"key1": "value1", "key2": "value2"}

    def test_instantiate_without_init_args(self):
        """Test instantiating a class without init_args."""
        config = {"class_path": "builtins.dict"}
        result = instantiate_obj_from_dict(config)
        assert isinstance(result, dict)
        assert result == {}

    def test_instantiate_with_key(self):
        """Test instantiating with a specific key from config."""
        config = {
            "model": {
                "class_path": "builtins.dict",
                "init_args": {"size": 128}
            }
        }
        result = instantiate_obj_from_dict(config, key="model")
        assert isinstance(result, dict)
        assert result == {"size": 128}

    def test_instantiate_missing_class_path(self):
        """Test error when class_path is missing."""
        config = {"init_args": {"size": 128}}
        with pytest.raises(ValueError, match="Configuration must contain 'class_path' key"):
            instantiate_obj_from_dict(config)

    def test_instantiate_missing_key(self):
        """Test error when specified key is missing."""
        config = {"class_path": "builtins.dict"}
        with pytest.raises(ValueError, match="Configuration must contain 'missing' key"):
            instantiate_obj_from_dict(config, key="missing")

    def test_instantiate_nested_configs(self):
        """Test instantiating nested configurations."""
        config = {
            "class_path": "builtins.dict",
            "init_args": {
                "nested": {
                    "class_path": "builtins.dict",
                    "init_args": {"key": "value"}
                }
            }
        }
        result = instantiate_obj_from_dict(config)
        assert isinstance(result, dict)
        assert isinstance(result["nested"], dict)
        assert result["nested"] == {"key": "value"}

    def test_instantiate_nested_list_configs(self):
        """Test instantiating configurations in lists."""
        config = {
            "class_path": "builtins.dict",
            "init_args": {
                "items": [
                    {"class_path": "builtins.dict", "init_args": {"key1": "value1"}},
                    {"class_path": "builtins.dict", "init_args": {"key2": "value2"}},
                    "regular_string"
                ]
            }
        }
        result = instantiate_obj_from_dict(config)
        assert isinstance(result, dict)
        assert len(result["items"]) == 3
        assert isinstance(result["items"][0], dict)
        assert result["items"][0] == {"key1": "value1"}
        assert isinstance(result["items"][1], dict)
        assert result["items"][1] == {"key2": "value2"}
        assert result["items"][2] == "regular_string"


class TestInstantiateObjFromPydantic:
    """Test the instantiate_obj_from_pydantic function."""

    def test_instantiate_from_pydantic_model(self):
        """Test instantiating from a Pydantic model."""
        # This test shows that instantiate_obj_from_pydantic expects jsonargparse pattern
        # For realistic nested Pydantic models, we'd use from_pydantic on the model class
        class TestConfig(BaseModel):
            class_path: str = "tests.unit.config.test_config.TestModel"
            init_args: dict[str, Any] = {
                "hidden_size": 256,
                "num_layers": 4,
                "activation": "swish"
            }

        config = TestConfig()
        result = instantiate_obj_from_pydantic(config)
        assert isinstance(result, TestModel)
        assert result.hidden_size == 256
        assert result.num_layers == 4
        assert result.activation == "swish"

    def test_instantiate_from_pydantic_with_key(self):
        """Test instantiating from a Pydantic model with key extraction."""
        class TestConfig(BaseModel):
            model: dict[str, Any] = {
                "class_path": "builtins.dict",
                "init_args": {"size": 128}
            }

        config = TestConfig()
        result = instantiate_obj_from_pydantic(config, key="model")
        assert isinstance(result, dict)
        assert result == {"size": 128}


class TestInstantiateObjFromDataclass:
    """Test the instantiate_obj_from_dataclass function."""

    def test_instantiate_from_dataclass(self):
        """Test instantiating from a dataclass."""
        # This test shows that instantiate_obj_from_dataclass expects jsonargparse pattern
        # For realistic nested dataclasses, we'd use from_dataclass on the model class
        @dataclasses.dataclass
        class TestConfig:
            class_path: str = "tests.unit.config.test_config.TestModel"
            init_args: dict[str, Any] = dataclasses.field(default_factory=lambda: {
                "hidden_size": 256,
                "num_layers": 4,
                "activation": "swish"
            })

        config = TestConfig()
        result = instantiate_obj_from_dataclass(config)
        assert isinstance(result, TestModel)
        assert result.hidden_size == 256
        assert result.num_layers == 4
        assert result.activation == "swish"

    def test_instantiate_from_dataclass_with_key(self):
        """Test instantiating from a dataclass with key extraction."""
        @dataclasses.dataclass
        class TestConfig:
            model: dict[str, Any] = dataclasses.field(default_factory=lambda: {
                "class_path": "builtins.dict",
                "init_args": {"size": 128}
            })

        config = TestConfig()
        result = instantiate_obj_from_dataclass(config, key="model")
        assert isinstance(result, dict)
        assert result == {"size": 128}


class TestInstantiateObjFromFile:
    """Test the instantiate_obj_from_file function."""

    def test_instantiate_from_yaml_file(self, tmp_path):
        """Test instantiating from a YAML file."""
        yaml_content = """
class_path: builtins.dict
init_args:
  key1: value1
  key2: value2
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        result = instantiate_obj_from_file(yaml_file)
        assert isinstance(result, dict)
        assert result == {"key1": "value1", "key2": "value2"}

    def test_instantiate_from_yaml_file_with_key(self, tmp_path):
        """Test instantiating from a YAML file with key extraction."""
        yaml_content = """
model:
  class_path: builtins.dict
  init_args:
    size: 128
    layers: 3
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        result = instantiate_obj_from_file(yaml_file, key="model")
        assert isinstance(result, dict)
        assert result == {"size": 128, "layers": 3}

    def test_instantiate_from_nonexistent_file(self, tmp_path):
        """Test error when file doesn't exist."""
        nonexistent_file = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            instantiate_obj_from_file(nonexistent_file)


class TestInstantiateObj:
    """Test the main instantiate_obj function."""

    def test_instantiate_from_dict(self):
        """Test instantiating from a dictionary."""
        config = {
            "class_path": "builtins.dict",
            "init_args": {"key": "value"}
        }
        result = instantiate_obj(config)
        assert isinstance(result, dict)
        assert result == {"key": "value"}

    def test_instantiate_from_pydantic(self):
        """Test instantiating from a Pydantic model."""
        # This test shows that instantiate_obj expects jsonargparse pattern
        class TestConfig(BaseModel):
            class_path: str = "tests.unit.config.test_config.TestModel"
            init_args: dict[str, Any] = {
                "hidden_size": 256,
                "num_layers": 4,
                "activation": "swish"
            }

        config = TestConfig()
        result = instantiate_obj(config)
        assert isinstance(result, TestModel)
        assert result.hidden_size == 256

    def test_instantiate_from_dataclass(self):
        """Test instantiating from a dataclass."""
        # This test shows that instantiate_obj expects jsonargparse pattern
        @dataclasses.dataclass
        class TestConfig:
            class_path: str = "tests.unit.config.test_config.TestModel"
            init_args: dict[str, Any] = dataclasses.field(default_factory=lambda: {
                "hidden_size": 256,
                "num_layers": 4,
                "activation": "swish"
            })

        config = TestConfig()
        result = instantiate_obj(config)
        assert isinstance(result, TestModel)
        assert result.hidden_size == 256

    def test_instantiate_from_file_path(self, tmp_path):
        """Test instantiating from a file path."""
        yaml_content = """
class_path: builtins.dict
init_args:
  key: value
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        result = instantiate_obj(str(yaml_file))
        assert isinstance(result, dict)
        assert result == {"key": "value"}

        # Test with Path object
        result = instantiate_obj(yaml_file)
        assert isinstance(result, dict)
        assert result == {"key": "value"}

    def test_instantiate_unsupported_type(self):
        """Test error with unsupported configuration type."""
        with pytest.raises(TypeError, match="Unsupported configuration type"):
            instantiate_obj(123)  # type: ignore[arg-type]

    def test_instantiate_with_key(self):
        """Test instantiating with key extraction."""
        config = {
            "model": {
                "class_path": "builtins.dict",
                "init_args": {"size": 128}
            }
        }
        result = instantiate_obj(config, key="model")
        assert isinstance(result, dict)
        assert result == {"size": 128}


class TestFromConfigMixin:
    """Test the FromConfig mixin functionality."""

    def test_from_dict_direct_instantiation(self):
        """Test from_dict with direct instantiation pattern."""
        config = {
            "hidden_size": 128,
            "num_layers": 3,
            "activation": "gelu"
        }
        model = TestModel.from_dict(config)
        assert isinstance(model, TestModel)
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.activation == "gelu"

    def test_from_dict_jsonargparse_pattern(self):
        """Test from_dict with jsonargparse pattern."""
        config = {
            "class_path": "tests.unit.config.test_config.TestModel",
            "init_args": {
                "hidden_size": 256,
                "num_layers": 4,
                "activation": "swish"
            }
        }
        model = TestModel.from_dict(config)
        assert isinstance(model, TestModel)
        assert model.hidden_size == 256
        assert model.num_layers == 4
        assert model.activation == "swish"

    def test_from_dict_with_key_direct(self):
        """Test from_dict with key extraction for direct instantiation."""
        config = {
            "model": {
                "hidden_size": 512,
                "num_layers": 6
            }
        }
        model = TestModel.from_dict(config, key="model")
        assert isinstance(model, TestModel)
        assert model.hidden_size == 512
        assert model.num_layers == 6
        assert model.activation == "relu"  # default value

    def test_from_dict_with_key_jsonargparse(self):
        """Test from_dict with key extraction for jsonargparse pattern."""
        config = {
            "model": {
                "class_path": "tests.unit.config.test_config.TestModel",
                "init_args": {
                    "hidden_size": 1024,
                    "num_layers": 8,
                    "activation": "mish"
                }
            }
        }
        model = TestModel.from_dict(config, key="model")
        assert isinstance(model, TestModel)
        assert model.hidden_size == 1024
        assert model.num_layers == 8
        assert model.activation == "mish"

    def test_from_dict_missing_key(self):
        """Test error when specified key is missing."""
        config = {"hidden_size": 128, "num_layers": 3}
        with pytest.raises(ValueError, match="Configuration must contain 'missing' key"):
            TestModel.from_dict(config, key="missing")

    def test_from_yaml_direct_instantiation(self, tmp_path):
        """Test from_yaml with direct instantiation pattern."""
        yaml_content = """
hidden_size: 128
num_layers: 3
activation: gelu
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        model = TestModel.from_yaml(yaml_file)
        assert isinstance(model, TestModel)
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.activation == "gelu"

    def test_from_yaml_jsonargparse_pattern(self, tmp_path):
        """Test from_yaml with jsonargparse pattern."""
        yaml_content = """
class_path: tests.unit.config.test_config.TestModel
init_args:
  hidden_size: 256
  num_layers: 4
  activation: swish
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        model = TestModel.from_yaml(yaml_file)
        assert isinstance(model, TestModel)
        assert model.hidden_size == 256
        assert model.num_layers == 4
        assert model.activation == "swish"

    def test_from_yaml_with_key(self, tmp_path):
        """Test from_yaml with key extraction."""
        yaml_content = """
model:
  hidden_size: 512
  num_layers: 6
  activation: gelu
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        model = TestModel.from_yaml(yaml_file, key="model")
        assert isinstance(model, TestModel)
        assert model.hidden_size == 512
        assert model.num_layers == 6
        assert model.activation == "gelu"

    def test_from_pydantic_direct_instantiation(self):
        """Test from_pydantic with direct instantiation pattern."""
        config = TestModelConfig()
        model = TestModel.from_pydantic(config)
        assert isinstance(model, TestModel)
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.activation == "relu"

    def test_from_pydantic_jsonargparse_pattern(self):
        """Test from_pydantic with jsonargparse pattern."""
        class ModelConfig(BaseModel):
            class_path: str = "tests.unit.config.test_config.TestModel"
            init_args: dict[str, Any] = {
                "hidden_size": 256,
                "num_layers": 4,
                "activation": "swish"
            }

        config = ModelConfig()
        model = TestModel.from_pydantic(config)
        assert isinstance(model, TestModel)
        assert model.hidden_size == 256
        assert model.num_layers == 4
        assert model.activation == "swish"

    def test_from_pydantic_with_key(self):
        """Test from_pydantic with key extraction."""
        class Config(BaseModel):
            model: dict[str, Any] = {
                "hidden_size": 512,
                "num_layers": 6,
                "activation": "mish"
            }

        config = Config()
        model = TestModel.from_pydantic(config, key="model")
        assert isinstance(model, TestModel)
        assert model.hidden_size == 512
        assert model.num_layers == 6
        assert model.activation == "mish"

    def test_from_dataclass_direct_instantiation(self):
        """Test from_dataclass with direct instantiation pattern."""
        config = TestModelDataclass()
        model = TestModel.from_dataclass(config)
        assert isinstance(model, TestModel)
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.activation == "relu"

    def test_from_dataclass_jsonargparse_pattern(self):
        """Test from_dataclass with jsonargparse pattern."""
        @dataclasses.dataclass
        class ModelConfig:
            class_path: str = "tests.unit.config.test_config.TestModel"
            init_args: dict[str, Any] = dataclasses.field(default_factory=lambda: {
                "hidden_size": 256,
                "num_layers": 4,
                "activation": "swish"
            })

        config = ModelConfig()
        model = TestModel.from_dataclass(config)
        assert isinstance(model, TestModel)
        assert model.hidden_size == 256
        assert model.num_layers == 4
        assert model.activation == "swish"

    def test_from_dataclass_with_key(self):
        """Test from_dataclass with key extraction."""
        @dataclasses.dataclass
        class Config:
            model: dict[str, Any] = dataclasses.field(default_factory=lambda: {
                "hidden_size": 512,
                "num_layers": 6,
                "activation": "mish"
            })

        config = Config()
        model = TestModel.from_dataclass(config, key="model")
        assert isinstance(model, TestModel)
        assert model.hidden_size == 512
        assert model.num_layers == 6
        assert model.activation == "mish"

    def test_from_dataclass_invalid_type(self):
        """Test error when config is not a dataclass."""
        with pytest.raises(TypeError, match="Expected dataclass instance"):
            TestModel.from_dataclass("not_a_dataclass")  # type: ignore[arg-type]

    def test_from_config_dict(self):
        """Test from_config with dictionary input."""
        config = {
            "hidden_size": 128,
            "num_layers": 3,
            "activation": "gelu"
        }
        model = TestModel.from_config(config)
        assert isinstance(model, TestModel)
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.activation == "gelu"

    def test_from_config_pydantic(self):
        """Test from_config with Pydantic model input."""
        config = TestModelConfig()
        model = TestModel.from_config(config)
        assert isinstance(model, TestModel)
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.activation == "relu"

    def test_from_config_dataclass(self):
        """Test from_config with dataclass input."""
        config = TestModelDataclass()
        model = TestModel.from_config(config)
        assert isinstance(model, TestModel)
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.activation == "relu"

    def test_from_config_file_path(self, tmp_path):
        """Test from_config with file path input."""
        yaml_content = """
hidden_size: 1024
num_layers: 8
activation: gelu
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        # Test with string path
        model = TestModel.from_config(str(yaml_file))
        assert isinstance(model, TestModel)
        assert model.hidden_size == 1024
        assert model.num_layers == 8
        assert model.activation == "gelu"

        # Test with Path object
        model = TestModel.from_config(yaml_file)
        assert isinstance(model, TestModel)
        assert model.hidden_size == 1024
        assert model.num_layers == 8
        assert model.activation == "gelu"

    def test_from_config_with_key(self):
        """Test from_config with key extraction."""
        config = {
            "model": {
                "hidden_size": 128,
                "num_layers": 3,
                "activation": "gelu"
            }
        }
        model = TestModel.from_config(config, key="model")
        assert isinstance(model, TestModel)
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.activation == "gelu"

    def test_realistic_nested_pydantic_usage(self):
        """Test realistic usage with nested Pydantic models."""
        # This shows how you'd actually use nested Pydantic models in practice
        config = TestNestedModelConfig()

        # Extract the nested model config and create the model
        model_config = config.model_params
        model = TestModel.from_pydantic(model_config)

        assert isinstance(model, TestModel)
        assert model.hidden_size == 256
        assert model.num_layers == 4
        assert model.activation == "swish"

        # Test the full nested structure
        assert config.name == "test_model"
        assert config.metadata == {"version": "1.0"}

    def test_complex_nested_configuration_structure(self):
        """Test a more complex nested configuration structure."""
        # This shows a realistic complex configuration with multiple nested models
        class OptimizerConfig(BaseModel):
            lr: float = 0.001
            weight_decay: float = 0.01
            optimizer_type: str = "adam"

        class TrainingConfig(BaseModel):
            epochs: int = 100
            batch_size: int = 32
            learning_rate: float = 0.001

        class ComplexModelConfig(BaseModel):
            name: str = "complex_model"
            model_params: TestModelConfig = TestModelConfig(hidden_size=512, num_layers=6, activation="gelu")
            optimizer_config: OptimizerConfig = OptimizerConfig()
            training_config: TrainingConfig = TrainingConfig()
            metadata: dict[str, Any] = {"version": "2.0", "description": "Complex model"}

            class Config:
                arbitrary_types_allowed = True

        config = ComplexModelConfig()

        # Create the main model
        model = TestModel.from_pydantic(config.model_params)
        assert isinstance(model, TestModel)
        assert model.hidden_size == 512
        assert model.num_layers == 6
        assert model.activation == "gelu"

        # Test the nested configurations
        assert config.optimizer_config.lr == 0.001
        assert config.optimizer_config.weight_decay == 0.01
        assert config.optimizer_config.optimizer_type == "adam"

        assert config.training_config.epochs == 100
        assert config.training_config.batch_size == 32
        assert config.training_config.learning_rate == 0.001

        assert config.metadata == {"version": "2.0", "description": "Complex model"}

    def test_realistic_nested_dataclass_usage(self):
        """Test realistic usage with nested dataclasses."""
        # This shows how you'd actually use nested dataclasses in practice
        config = TestNestedModelDataclass()

        # Extract the nested model config and create the model
        model_config = config.model_params
        model = TestModel.from_dataclass(model_config)

        assert isinstance(model, TestModel)
        assert model.hidden_size == 256
        assert model.num_layers == 4
        assert model.activation == "swish"

        # Test the full nested structure
        assert config.name == "test_model"
        assert config.metadata == {"version": "1.0"}


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_instantiate_with_empty_dict(self):
        """Test instantiation with empty configuration dictionary."""
        config = {}
        with pytest.raises(ValueError, match="Configuration must contain 'class_path' key"):
            instantiate_obj(config)

    def test_instantiate_with_none_values(self):
        """Test instantiation with None values in configuration."""
        config = {
            "class_path": "builtins.dict",
            "init_args": {
                "key1": None,
                "key2": "value",
                "key3": None
            }
        }
        result = instantiate_obj(config)
        assert isinstance(result, dict)
        assert result == {"key1": None, "key2": "value", "key3": None}

    def test_instantiate_with_empty_init_args(self):
        """Test instantiation with empty init_args."""
        config = {
            "class_path": "builtins.dict",
            "init_args": {}
        }
        result = instantiate_obj(config)
        assert isinstance(result, dict)
        assert result == {}

    def test_instantiate_with_missing_init_args_key(self):
        """Test instantiation when init_args key is missing."""
        config = {
            "class_path": "builtins.dict"
        }
        result = instantiate_obj(config)
        assert isinstance(result, dict)
        assert result == {}

    def test_instantiate_with_deeply_nested_configs(self):
        """Test instantiation with deeply nested configurations."""
        config = {
            "class_path": "builtins.dict",
            "init_args": {
                "level1": {
                    "class_path": "builtins.dict",
                    "init_args": {
                        "level2": {
                            "class_path": "builtins.dict",
                            "init_args": {
                                "level3": {
                                    "class_path": "builtins.dict",
                                    "init_args": {"key": "value"}
                                }
                            }
                        }
                    }
                }
            }
        }
        result = instantiate_obj(config)
        assert isinstance(result, dict)
        assert isinstance(result["level1"], dict)
        assert isinstance(result["level1"]["level2"], dict)
        assert isinstance(result["level1"]["level2"]["level3"], dict)
        assert result["level1"]["level2"]["level3"] == {"key": "value"}

    def test_instantiate_with_mixed_list_and_dict(self):
        """Test instantiation with mixed list and dictionary configurations."""
        config = {
            "class_path": "builtins.dict",
            "init_args": {
                "items": [
                    {"class_path": "builtins.dict", "init_args": {"key1": "value1"}},
                    "regular_string",
                    {"class_path": "builtins.dict", "init_args": {"key2": "value2"}},
                    42,
                    {"class_path": "builtins.dict", "init_args": {"key3": "value3"}}
                ]
            }
        }
        result = instantiate_obj(config)
        assert isinstance(result, dict)
        assert len(result["items"]) == 5
        assert isinstance(result["items"][0], dict)
        assert result["items"][0] == {"key1": "value1"}
        assert result["items"][1] == "regular_string"
        assert isinstance(result["items"][2], dict)
        assert result["items"][2] == {"key2": "value2"}
        assert result["items"][3] == 42
        assert isinstance(result["items"][4], dict)
        assert result["items"][4] == {"key3": "value3"}

    def test_from_dict_with_extra_kwargs(self):
        """Test from_dict with extra keyword arguments."""
        config = {
            "hidden_size": 128,
            "num_layers": 3,
            "extra_param": "extra_value",
            "another_param": 42
        }
        model = TestModel.from_dict(config)
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.kwargs == {"extra_param": "extra_value", "another_param": 42}

    def test_from_dict_with_empty_config(self):
        """Test from_dict with empty configuration."""
        config = {}
        with pytest.raises(TypeError):  # Missing required arguments
            TestModel.from_dict(config)

    def test_from_dict_with_none_values(self):
        """Test from_dict with None values."""
        config = {
            "hidden_size": None,
            "num_layers": 3,
            "param1": None,
            "param2": "not_none"
        }
        model = TestModel.from_dict(config)
        assert model.hidden_size is None
        assert model.num_layers == 3
        assert model.kwargs == {"param1": None, "param2": "not_none"}

    def test_from_yaml_with_comments(self, tmp_path):
        """Test from_yaml with YAML comments."""
        yaml_content = """
# This is a comment
hidden_size: 128
# Another comment
num_layers: 3
activation: gelu
# Final comment
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        model = TestModel.from_yaml(yaml_file)
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.activation == "gelu"

    def test_from_yaml_with_multiline_strings(self, tmp_path):
        """Test from_yaml with multiline strings."""
        yaml_content = """
hidden_size: 128
num_layers: 3
activation: |
  This is a multiline
  string with multiple
  lines
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        model = TestModel.from_yaml(yaml_file)
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert "multiline" in model.activation

    def test_from_pydantic_with_validation_errors(self):
        """Test from_pydantic with Pydantic validation."""
        class TestConfig(BaseModel):
            hidden_size: int
            num_layers: int
            activation: str

        # This should work with valid data
        config = TestConfig(hidden_size=128, num_layers=3, activation="gelu")
        model = TestModel.from_pydantic(config)
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.activation == "gelu"

    def test_from_dataclass_with_default_factory(self):
        """Test from_dataclass with default_factory."""
        @dataclasses.dataclass
        class TestConfig:
            hidden_size: int = 128
            num_layers: int = 3
            activation: str = "relu"
            param1: list[str] = dataclasses.field(default_factory=list)
            param2: dict[str, Any] = dataclasses.field(default_factory=dict)

        config = TestConfig()
        model = TestModel.from_dataclass(config)
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.activation == "relu"
        assert model.kwargs == {"param1": [], "param2": {}}

    def test_from_config_with_unsupported_type_fallback(self):
        """Test from_config fallback to instantiate_obj for unsupported types."""
        # This tests the fallback mechanism in from_config
        class CustomConfig:
            def __init__(self):
                self.value = "custom"

        custom_config = CustomConfig()
        # This should fall back to instantiate_obj and potentially fail
        # depending on the custom config structure
        with pytest.raises((TypeError, AttributeError)):
            TestModel.from_config(custom_config)


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple features."""

    def test_yaml_to_pydantic_to_instantiation(self, tmp_path):
        """Test complete workflow: YAML -> Pydantic -> Instantiation."""
        yaml_content = """
hidden_size: 128
num_layers: 3
activation: gelu
extra_param: extra_value
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        # Load YAML into Pydantic model
        class ConfigModel(BaseModel):
            hidden_size: int
            num_layers: int
            activation: str = "relu"
            extra_param: str = "default"

        with yaml_file.open("r") as f:
            import yaml
            data = yaml.safe_load(f)
            pydantic_config = ConfigModel(**data)

        # Instantiate from Pydantic model
        model = TestModel.from_pydantic(pydantic_config)
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.activation == "gelu"
        assert model.kwargs == {"extra_param": "extra_value"}

    def test_nested_configuration_workflow(self, tmp_path):
        """Test complex nested configuration workflow."""
        yaml_content = """
main_model:
  class_path: builtins.dict
  init_args:
    required_param: "nested_test"
    optional_param: 200
    nested_config:
      class_path: builtins.dict
      init_args:
        key1: "value1"
        key2: "value2"

sub_model:
  hidden_size: 128
  num_layers: 3
  activation: gelu
"""
        yaml_file = tmp_path / "nested_config.yaml"
        yaml_file.write_text(yaml_content)

        # Test jsonargparse pattern
        main_model = instantiate_obj(yaml_file, key="main_model")
        assert isinstance(main_model, dict)
        assert main_model["required_param"] == "nested_test"
        assert main_model["optional_param"] == 200
        assert isinstance(main_model["nested_config"], dict)
        assert main_model["nested_config"] == {"key1": "value1", "key2": "value2"}

        # Test direct instantiation pattern
        sub_model = TestModel.from_yaml(yaml_file, key="sub_model")
        assert sub_model.hidden_size == 128
        assert sub_model.num_layers == 3
        assert sub_model.activation == "gelu"

    def test_mixed_configuration_formats(self):
        """Test mixing different configuration formats."""
        # Start with a dictionary
        base_config = {
            "hidden_size": 128,
            "num_layers": 3,
            "activation": "gelu"
        }

        # Convert to Pydantic
        class ConfigModel(BaseModel):
            hidden_size: int
            num_layers: int
            activation: str = "relu"

        pydantic_config = ConfigModel(**base_config)

        # Convert to dataclass
        @dataclasses.dataclass
        class ConfigDataclass:
            hidden_size: int
            num_layers: int
            activation: str = "relu"

        dataclass_config = ConfigDataclass(**base_config)

        # Test all formats produce the same result
        model1 = TestModel.from_dict(base_config)
        model2 = TestModel.from_pydantic(pydantic_config)
        model3 = TestModel.from_dataclass(dataclass_config)

        assert model1.hidden_size == model2.hidden_size == model3.hidden_size
        assert model1.num_layers == model2.num_layers == model3.num_layers
        assert model1.activation == model2.activation == model3.activation

    def test_error_handling_chain(self):
        """Test error handling across different configuration formats."""
        # Test missing required parameter
        with pytest.raises(TypeError):
            TestModel.from_dict({})

        # Test with valid configuration
        model = TestModel.from_dict({"hidden_size": 128, "num_layers": 3})
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.activation == "relu"  # default value
