# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, S404, S603

"""Policy names resolve without importing unused policies."""

import importlib
import subprocess
import sys
from pathlib import Path

from physicalai import policies
from physicalai.policies import get_physicalai_policy_class


def test_short_names_resolve_to_direct_class_paths() -> None:
    """Every registered short name and training class path loads the same class."""
    for short_name, class_name in policies._POLICIES.items():  # noqa: SLF001
        cls = getattr(importlib.import_module(f"physicalai.policies.{short_name}.policy"), class_name)
        assert get_physicalai_policy_class(short_name.upper()) is cls

    root = Path(__file__).resolve().parents[3]
    policy_dirs = {path.parent.name for path in (root / "src/physicalai/policies").glob("*/policy.py")}
    assert policy_dirs - {"base", "lerobot"} == set(policies._POLICIES)  # noqa: SLF001
    for config in (root / "configs/physicalai").rglob("*.yaml"):
        for line in config.read_text(encoding="utf-8").splitlines():
            if "class_path: physicalai.policies." not in line:
                continue
            path = line.split("class_path: ", 1)[1].strip()
            if ".lerobot." in path:
                continue
            module, name = path.rsplit(".", 1)
            cls = getattr(importlib.import_module(module), name)
            assert get_physicalai_policy_class(cls.__module__.split(".")[2]) is cls


def test_loading_one_policy_does_not_import_the_others() -> None:
    """Root exports and short names load only the requested policy."""
    script = """
import sys
import physicalai.policies as policies
assert 'physicalai.policies.cosmos3' not in sys.modules
from physicalai.policies import ACT, ACTConfig, ACTModel
from physicalai.policies.act.policy import ACT as DirectACT
assert ACT is DirectACT
assert policies.get_physicalai_policy_class('ACT') is ACT
assert ACTConfig is policies.ACTConfig and ACTModel is policies.ACTModel
assert {'ACT', 'ACTConfig', 'ACTModel'} <= set(policies.__all__)
assert 'physicalai.policies.cosmos3' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, check=False, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
