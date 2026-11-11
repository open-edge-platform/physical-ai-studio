# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, S404, S603
# Tests run only a fixed script in the current Python interpreter.

"""Keep archive helpers independent of the training dependency stack."""

import subprocess
import sys
from importlib import import_module

import pytest


def test_archive_helpers_do_not_import_training_dependencies() -> None:
    """Importing archive safety in a fresh interpreter must not load training libraries."""
    script = """
import sys
from physicalai.data.archive_safety import SafeZipArchive
assert SafeZipArchive is not None
for module in ('torch', 'lightning', 'lerobot', 'transformers'):
    assert module not in sys.modules, module
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, check=False, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("DataModule", "datamodules"),
        ("Dataset", "dataset"),
        ("LeRobotDataModule", "lerobot"),
        ("Feature", "observation"),
        ("FeatureType", "observation"),
        ("NormalizationParameters", "observation"),
        ("NormalizationValue", "observation"),
        ("Observation", "observation"),
    ],
)
def test_lazy_data_exports_keep_their_public_identity(name: str, module: str) -> None:
    """Public re-exports must resolve to and cache the original objects."""
    data = import_module("physicalai.data")
    assert getattr(data, name) is getattr(import_module(f"physicalai.data.{module}"), name)
    assert data.__dict__[name] is getattr(data, name)


def test_unknown_data_export_raises_attribute_error() -> None:
    """Unknown attributes must retain normal module lookup semantics."""
    data = import_module("physicalai.data")
    with pytest.raises(AttributeError, match="no attribute 'unknown_export'"):
        _ = data.unknown_export
