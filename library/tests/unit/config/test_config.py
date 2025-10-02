# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for config module functionality."""

import dataclasses
import os
import tempfile
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
class ConfigTestModel(FromConfig):
    """Test model for configuration testing."""

    def __init__(self, hidden_size: int, num_layers: int, activation: str = "relu", **kwargs):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.kwargs = kwargs

    def __eq__(self, other):
        if not isinstance(other, ConfigTestModel):
            return False
        return (
            self.hidden_size == other.hidden_size
            and self.num_layers == other.num_layers
            and self.activation == other.activation
            and self.kwargs == other.kwargs
        )


# Pydantic models
class ConfigTestModelConfig(BaseModel):
    """Pydantic model for testing."""
    hidden_size: int = 128
    num_layers: int = 3
    activation: str = "relu"


# Dataclasses
@dataclasses.dataclass
class ConfigTestModelDataclass:
    """Dataclass for testing."""
    hidden_size: int = 128
    num_layers: int = 3
    activation: str = "relu"


class TestImportClass:
    """Test the _import_class helper function."""

    @pytest.mark.parametrize("class_path,expected_class", [
        ("builtins.dict", dict),
        ("pathlib.Path", Path),
    ])
    def test_import_valid_classes(self, class_path, expected_class):
        """Test importing valid classes."""
        assert _import_class(class_path) is expected_class

    @pytest.mark.parametrize("class_path,expected_error", [
        ("nonexistent.module.Class", "Cannot import 'nonexistent.module.Class'"),
        ("pathlib.NonExistentClass", "Cannot import 'pathlib.NonExistentClass'"),
    ])
    def test_import_invalid_classes(self, class_path, expected_error):
        """Test importing invalid classes."""
        with pytest.raises(ImportError, match=expected_error):
            _import_class(class_path)


class TestInstantiateObjFromDict:
    """Test the instantiate_obj_from_dict function."""

    @pytest.mark.parametrize("config,expected", [
        ({"class_path": "builtins.dict", "init_args": {"key1": "value1"}}, {"key1": "value1"}),
        ({"class_path": "builtins.dict"}, {}),
        ({"model": {"class_path": "builtins.dict", "init_args": {"size": 128}}}, {"size": 128}),
    ])
    def test_basic_instantiation(self, config, expected):
        """Test basic instantiation scenarios."""
        key = "model" if "model" in config else None
        result = instantiate_obj_from_dict(config, key=key)
        assert result == expected

    def test_nested_instantiation(self):
        """Test nested configurations."""
        config = {
            "class_path": "builtins.dict",
            "init_args": {
                "nested": {"class_path": "builtins.dict", "init_args": {"key": "value"}},
                "items": [
                    {"class_path": "builtins.dict", "init_args": {"key1": "value1"}},
                    "regular_string"
                ]
            }
        }
        result = instantiate_obj_from_dict(config)
        assert isinstance(result, dict)
        assert result["nested"] == {"key": "value"}
        assert result["items"][0] == {"key1": "value1"}
        assert result["items"][1] == "regular_string"

    @pytest.mark.parametrize("config,key,expected_error", [
        ({"init_args": {"size": 128}}, None, "Configuration must contain 'class_path' key"),
        ({"class_path": "builtins.dict"}, "missing", "Configuration must contain 'missing' key"),
    ])
    def test_error_conditions(self, config, key, expected_error):
        """Test error conditions."""
        with pytest.raises(ValueError, match=expected_error):
            instantiate_obj_from_dict(config, key=key)


class TestInstantiateObjFromPydantic:
    """Test the instantiate_obj_from_pydantic function."""

    def test_instantiate_from_pydantic(self):
        """Test instantiating from Pydantic models."""
        # Direct instantiation pattern
        class TestConfig(BaseModel):
            class_path: str = "tests.unit.config.test_config.ConfigTestModel"
            init_args: dict[str, Any] = {"hidden_size": 256, "num_layers": 4, "activation": "swish"}

        config = TestConfig()
        result = instantiate_obj_from_pydantic(config)
        assert isinstance(result, ConfigTestModel)
        assert result.hidden_size == 256

        # With key extraction
        class TestConfigWithKey(BaseModel):
            model: dict[str, Any] = {"class_path": "builtins.dict", "init_args": {"size": 128}}

        config = TestConfigWithKey()
        result = instantiate_obj_from_pydantic(config, key="model")
        assert result == {"size": 128}


class TestInstantiateObjFromDataclass:
    """Test the instantiate_obj_from_dataclass function."""

    def test_instantiate_from_dataclass(self):
        """Test instantiating from dataclasses."""
        # Direct instantiation pattern
        @dataclasses.dataclass
        class TestConfig:
            class_path: str = "tests.unit.config.test_config.ConfigTestModel"
            init_args: dict[str, Any] = dataclasses.field(default_factory=lambda: {
                "hidden_size": 256, "num_layers": 4, "activation": "swish"
            })

        config = TestConfig()
        result = instantiate_obj_from_dataclass(config)
        assert isinstance(result, ConfigTestModel)
        assert result.hidden_size == 256

        # With key extraction
        @dataclasses.dataclass
        class TestConfigWithKey:
            model: dict[str, Any] = dataclasses.field(default_factory=lambda: {
                "class_path": "builtins.dict", "init_args": {"size": 128}
            })

        config = TestConfigWithKey()
        result = instantiate_obj_from_dataclass(config, key="model")
        assert result == {"size": 128}


class TestInstantiateObjFromFile:
    """Test the instantiate_obj_from_file function."""

    @pytest.mark.parametrize("yaml_content,key,expected", [
        ("class_path: builtins.dict\ninit_args:\n  key1: value1\n  key2: value2", None, {"key1": "value1", "key2": "value2"}),
        ("model:\n  class_path: builtins.dict\n  init_args:\n    size: 128", "model", {"size": 128}),
    ])
    def test_instantiate_from_yaml_file(self, tmp_path, yaml_content, key, expected):
        """Test instantiating from YAML files."""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)
        result = instantiate_obj_from_file(yaml_file, key=key)
        assert result == expected

    def test_file_not_found(self, tmp_path):
        """Test file not found error."""
        nonexistent_file = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            instantiate_obj_from_file(nonexistent_file)


class TestInstantiateObj:
    """Test the main instantiate_obj function."""

    def test_instantiate_from_different_sources(self):
        """Test instantiating from different configuration sources."""
        # From dict
        config = {"class_path": "builtins.dict", "init_args": {"key": "value"}}
        result = instantiate_obj(config)
        assert isinstance(result, dict)
        assert result == {"key": "value"}

        # From Pydantic with class_path
        class TestConfig(BaseModel):
            class_path: str = "tests.unit.config.test_config.ConfigTestModel"
            init_args: dict[str, Any] = {"hidden_size": 256, "num_layers": 4}

        config = TestConfig()
        result = instantiate_obj(config)
        assert isinstance(result, ConfigTestModel)
        assert result.hidden_size == 256

    def test_instantiate_from_file_path(self, tmp_path):
        """Test instantiating from file paths."""
        yaml_content = "class_path: builtins.dict\ninit_args:\n  key: value"
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        # String path
        result = instantiate_obj(str(yaml_file))
        assert result == {"key": "value"}

        # Path object
        result = instantiate_obj(yaml_file)
        assert result == {"key": "value"}

    def test_instantiate_error_conditions(self):
        """Test error conditions."""
        # Unsupported type
        with pytest.raises(TypeError, match="Unsupported configuration type"):
            instantiate_obj(123)  # type: ignore[arg-type]

        # With key extraction
        config = {"model": {"class_path": "builtins.dict", "init_args": {"size": 128}}}
        result = instantiate_obj(config, key="model")
        assert result == {"size": 128}


class TestFromConfigMixin:
    """Test the FromConfig mixin functionality."""

    @pytest.mark.parametrize("config,key,expected_size,expected_activation", [
        ({"hidden_size": 128, "num_layers": 3, "activation": "gelu"}, None, 128, "gelu"),
        ({"class_path": "tests.unit.config.test_config.ConfigTestModel", "init_args": {"hidden_size": 256, "num_layers": 4, "activation": "swish"}}, None, 256, "swish"),
        ({"model": {"hidden_size": 512, "num_layers": 6}}, "model", 512, "relu"),
    ])
    def test_from_dict_patterns(self, config, key, expected_size, expected_activation):
        """Test from_dict with different patterns."""
        model = ConfigTestModel.from_dict(config, key=key)
        assert model.hidden_size == expected_size
        assert model.activation == expected_activation

    def test_from_dict_missing_key_error(self):
        """Test missing key error."""
        config = {"model": {"hidden_size": 512, "num_layers": 6}}
        with pytest.raises(ValueError, match="Configuration must contain 'missing' key"):
            ConfigTestModel.from_dict(config, key="missing")

    @pytest.mark.parametrize("yaml_content,key,expected_size,expected_activation", [
        ("hidden_size: 128\nnum_layers: 3\nactivation: gelu", None, 128, "gelu"),
        ("class_path: tests.unit.config.test_config.ConfigTestModel\ninit_args:\n  hidden_size: 256\n  num_layers: 4", None, 256, "relu"),
        ("model:\n  hidden_size: 512\n  num_layers: 6", "model", 512, "relu"),
    ])
    def test_from_yaml_patterns(self, tmp_path, yaml_content, key, expected_size, expected_activation):
        """Test from_yaml with different patterns."""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)
        model = ConfigTestModel.from_yaml(yaml_file, key=key)
        assert model.hidden_size == expected_size
        assert model.activation == expected_activation

    @pytest.mark.parametrize("config_class,expected_size", [
        (ConfigTestModelConfig, 128),
    ])
    def test_from_pydantic_patterns(self, config_class, expected_size):
        """Test from_pydantic with different patterns."""
        config = config_class()
        model = ConfigTestModel.from_pydantic(config)
        assert model.hidden_size == expected_size
        assert model.activation == "relu"

    @pytest.mark.parametrize("config_class,expected_size", [
        (ConfigTestModelDataclass, 128),
    ])
    def test_from_dataclass_patterns(self, config_class, expected_size):
        """Test from_dataclass with different patterns."""
        config = config_class()
        model = ConfigTestModel.from_dataclass(config)
        assert model.hidden_size == expected_size
        assert model.activation == "relu"

    def test_from_dataclass_invalid_type_error(self):
        """Test invalid type error for dataclass."""
        with pytest.raises(TypeError, match="Expected dataclass instance"):
            ConfigTestModel.from_dataclass("not_a_dataclass")  # type: ignore[arg-type]

    @pytest.mark.parametrize("config,expected_size", [
        ({"hidden_size": 128, "num_layers": 3, "activation": "gelu"}, 128),
        (ConfigTestModelConfig(), 128),
        (ConfigTestModelDataclass(), 128),
    ])
    def test_from_config_unified_interface(self, config, expected_size):
        """Test from_config unified interface."""
        if isinstance(config, dict):
            model = ConfigTestModel.from_config(config)
        else:
            model = ConfigTestModel.from_config(config)
        assert model.hidden_size == expected_size

    def test_from_config_file_path(self):
        """Test from_config with file path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("hidden_size: 1024\nnum_layers: 8\nactivation: gelu")
            temp_file = f.name

        try:
            model = ConfigTestModel.from_config(temp_file)
            assert model.hidden_size == 1024
        finally:
            os.unlink(temp_file)

    def test_from_config_with_key_extraction(self):
        """Test from_config with key extraction."""
        config = {"model": {"hidden_size": 128, "num_layers": 3}}
        model = ConfigTestModel.from_config(config, key="model")
        assert model.hidden_size == 128


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.parametrize("config,expected", [
        ({}, ValueError),  # Empty dict
        ({"class_path": "builtins.dict", "init_args": {"key1": None, "key2": "value"}}, {"key1": None, "key2": "value"}),
        ({"class_path": "builtins.dict", "init_args": {}}, {}),
        ({"class_path": "builtins.dict"}, {}),
    ])
    def test_instantiation_edge_cases(self, config, expected):
        """Test various edge cases for instantiation."""
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                instantiate_obj(config)
        else:
            result = instantiate_obj(config)
            assert result == expected

    def test_deeply_nested_instantiation(self):
        """Test deeply nested configurations."""
        config = {
            "class_path": "builtins.dict",
            "init_args": {
                "level1": {
                    "class_path": "builtins.dict",
                    "init_args": {
                        "level2": {
                            "class_path": "builtins.dict",
                            "init_args": {"key": "value"}
                        }
                    }
                }
            }
        }
        result = instantiate_obj(config)
        assert isinstance(result, dict)
        assert result["level1"]["level2"] == {"key": "value"}

    def test_mixed_list_and_dict_instantiation(self):
        """Test mixed list and dict instantiation."""
        config = {
            "class_path": "builtins.dict",
            "init_args": {
                "items": [
                    {"class_path": "builtins.dict", "init_args": {"key1": "value1"}},
                    "regular_string",
                    42
                ]
            }
        }
        result = instantiate_obj(config)
        assert isinstance(result, dict)
        assert result["items"][0] == {"key1": "value1"}
        assert result["items"][1] == "regular_string"
        assert result["items"][2] == 42

    def test_from_dict_edge_cases(self):
        """Test edge cases for from_dict."""
        # Extra kwargs
        config = {"hidden_size": 128, "num_layers": 3, "extra_param": "extra_value"}
        model = ConfigTestModel.from_dict(config)
        assert model.kwargs == {"extra_param": "extra_value"}

        # Empty config
        with pytest.raises(TypeError):  # Missing required arguments
            ConfigTestModel.from_dict({})

        # None values
        config = {"hidden_size": None, "num_layers": 3, "param1": None}
        model = ConfigTestModel.from_dict(config)
        assert model.hidden_size is None
        assert model.kwargs == {"param1": None}

    @pytest.mark.parametrize("yaml_content,expected_activation", [
        ("# Comment\nhidden_size: 128\nnum_layers: 3\nactivation: gelu", "gelu"),
        ("hidden_size: 128\nnum_layers: 3\nactivation: |\n  multiline\n  string", "multiline\nstring"),
    ])
    def test_yaml_edge_cases(self, tmp_path, yaml_content, expected_activation):
        """Test YAML edge cases."""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)
        model = ConfigTestModel.from_yaml(yaml_file)
        assert model.hidden_size == 128
        assert expected_activation in model.activation

    def test_dataclass_edge_cases(self):
        """Test dataclass edge cases."""
        @dataclasses.dataclass
        class TestConfig:
            hidden_size: int = 128
            num_layers: int = 3
            param1: list[str] = dataclasses.field(default_factory=list)
            param2: dict[str, Any] = dataclasses.field(default_factory=dict)

        config = TestConfig()
        model = ConfigTestModel.from_dataclass(config)
        assert model.kwargs == {"param1": [], "param2": {}}

    def test_error_handling(self):
        """Test error handling scenarios."""
        # Unsupported type fallback
        class CustomConfig:
            def __init__(self):
                self.value = "custom"

        with pytest.raises((TypeError, AttributeError)):
            ConfigTestModel.from_config(CustomConfig())


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple features."""

    def test_yaml_to_pydantic_workflow(self, tmp_path):
        """Test YAML -> Pydantic -> Instantiation workflow."""
        yaml_content = "hidden_size: 128\nnum_layers: 3\nactivation: gelu\nextra_param: extra_value"
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        class ConfigModel(BaseModel):
            hidden_size: int
            num_layers: int
            activation: str = "relu"
            extra_param: str = "default"

        with yaml_file.open("r") as f:
            import yaml  # type: ignore[import-untyped]
            data = yaml.safe_load(f)
            pydantic_config = ConfigModel(**data)

        model = ConfigTestModel.from_pydantic(pydantic_config)
        assert model.hidden_size == 128
        assert model.kwargs == {"extra_param": "extra_value"}

    def test_nested_configuration_workflow(self, tmp_path):
        """Test complex nested configuration workflow."""
        yaml_content = """
main_model:
  class_path: builtins.dict
  init_args:
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

        # JSONargparse pattern
        main_model = instantiate_obj(yaml_file, key="main_model")
        assert isinstance(main_model, dict)
        assert main_model["nested_config"] == {"key1": "value1", "key2": "value2"}

        # Direct instantiation pattern
        sub_model = ConfigTestModel.from_yaml(yaml_file, key="sub_model")
        assert sub_model.hidden_size == 128

    def test_mixed_configuration_formats(self):
        """Test mixing different configuration formats."""
        base_config = {"hidden_size": 128, "num_layers": 3, "activation": "gelu"}

        # Convert to different formats
        class ConfigModel(BaseModel):
            hidden_size: int
            num_layers: int
            activation: str = "relu"

        @dataclasses.dataclass
        class ConfigDataclass:
            hidden_size: int
            num_layers: int
            activation: str = "relu"

        pydantic_config = ConfigModel(**base_config)
        dataclass_config = ConfigDataclass(**base_config)

        # Test all formats produce the same result
        model1 = ConfigTestModel.from_dict(base_config)
        model2 = ConfigTestModel.from_pydantic(pydantic_config)
        model3 = ConfigTestModel.from_dataclass(dataclass_config)

        assert model1.hidden_size == model2.hidden_size == model3.hidden_size
        assert model1.activation == model2.activation == model3.activation
