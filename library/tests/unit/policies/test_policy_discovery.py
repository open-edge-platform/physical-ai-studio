# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, S404, S603

"""Policy names resolve without a central import registry."""

import importlib
import subprocess
import sys
from pathlib import Path

from physicalai import policies
from physicalai.policies import get_physicalai_policy_class


def test_short_names_resolve_to_direct_class_paths() -> None:
    """Every registered short name and training class path loads the same class."""
    for short_name, (directory, class_name) in policies._POLICY_PATHS.items():  # noqa: SLF001
        cls = getattr(importlib.import_module(f"physicalai.policies.{directory}.policy"), class_name)
        assert get_physicalai_policy_class(short_name.upper()) is cls
        assert getattr(policies, class_name) is cls

    root = Path(__file__).resolve().parents[3]
    for config in (root / "configs/physicalai").rglob("*.yaml"):
        for line in config.read_text(encoding="utf-8").splitlines():
            if "class_path: physicalai.policies." not in line:
                continue
            path = line.split("class_path: ", 1)[1].strip()
            if ".lerobot." in path:
                continue
            module, name = path.rsplit(".", 1)
            cls = getattr(importlib.import_module(module), name)
            assert get_physicalai_policy_class(name) is cls


def test_loading_one_policy_does_not_import_the_others() -> None:
    """Short-name discovery leaves unrelated optional policies unloaded."""
    script = """
import sys
import physicalai.policies as policies
assert 'physicalai.policies.cosmos3' not in sys.modules
assert policies.get_physicalai_policy_class('act') is policies.ACT
assert 'physicalai.policies.cosmos3' not in sys.modules
assert 'Cosmos3' in dir(policies)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, check=False, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
