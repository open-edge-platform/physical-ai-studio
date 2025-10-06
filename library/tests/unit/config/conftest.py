# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test fixtures and utilities for config module tests."""

import pytest


@pytest.fixture
def yaml_config_content():
    """Fixture providing YAML configuration content."""
    return """
# Test configuration
model:
  class_path: builtins.dict
  init_args:
    hidden_size: 256
    num_layers: 4
    activation: swish

optimizer:
  class_path: builtins.dict
  init_args:
    lr: 0.001
    weight_decay: 0.01

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
"""


@pytest.fixture
def yaml_config_file(tmp_path, yaml_config_content):
    """Fixture providing a YAML configuration file."""
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_config_content)
    return yaml_file
