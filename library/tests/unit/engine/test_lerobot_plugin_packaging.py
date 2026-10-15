# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Installed entry-point, worker lock, and Runtime adapter contracts."""

from __future__ import annotations

from importlib import resources
from importlib.metadata import distribution, entry_points
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement

from physicalai.engine._environment import locked_version, supported_versions
from physicalai_lerobot_policy_plugin.engine import LeRobotEngine

_PLUGIN_DISTRIBUTION = "physicalai-lerobot-policy-plugin"


def test_lerobot_plugin_owns_engine_and_runtime_adapter_entrypoints() -> None:
    engine_entries = [
        entry
        for entry in entry_points(group="physicalai.engines")
        if entry.name == "lerobot" and entry.dist is not None and entry.dist.name == _PLUGIN_DISTRIBUTION
    ]
    adapter_entries = [
        entry
        for entry in entry_points(group="physicalai.inference.adapters")
        if entry.name == "lerobot" and entry.dist is not None and entry.dist.name == _PLUGIN_DISTRIBUTION
    ]

    assert len(engine_entries) == 1
    assert engine_entries[0].value == "physicalai_lerobot_policy_plugin.engine:LeRobotEngine"
    assert len(adapter_entries) == 1
    assert adapter_entries[0].value == "physicalai_lerobot_policy_plugin.inference:register"
    package = resources.files("physicalai_lerobot_policy_plugin")
    assert (package / "env" / "pyproject.toml").is_file()
    assert (package / "env" / "uv.lock").is_file()


def test_worker_lock_is_inside_the_range_declared_by_plugin_metadata() -> None:
    assert locked_version(LeRobotEngine.env) in supported_versions(LeRobotEngine.env)


def test_worker_project_has_only_locked_framework_dependencies() -> None:
    project = resources.files("physicalai_lerobot_policy_plugin") / "env"
    pyproject = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    lock_text = (project / "uv.lock").read_text(encoding="utf-8")
    lock = tomllib.loads(lock_text)
    project_dependencies = {item.split("[", maxsplit=1)[0].split(">", maxsplit=1)[0] for item in pyproject["project"]["dependencies"]}
    lock_packages = {package["name"] for package in lock["package"]}

    assert {"cpu", "cu128", "xpu"} <= set(pyproject["project"]["optional-dependencies"])
    assert "physicalai" not in project_dependencies
    assert "physicalai-train" not in project_dependencies
    assert "physicalai" not in lock_packages
    assert "physicalai-train" not in lock_packages
    assert "https://download.pytorch.org/whl/cpu" in lock_text
    assert "https://download.pytorch.org/whl/cu128" in lock_text
    assert "https://download.pytorch.org/whl/xpu" in lock_text


def test_runtime_registration_import_does_not_load_the_training_engine() -> None:
    code = (
        "import sys; import physicalai_lerobot_policy_plugin.inference; "
        "assert not any(name == 'physicalai.engine' or name.startswith('physicalai.engine.') for name in sys.modules); "
        "assert 'lerobot' not in sys.modules and 'torch' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_library_lerobot_extra_installs_plugin_train_extra() -> None:
    train = distribution("physicalai-train")
    requirements = [Requirement(requirement) for requirement in train.requires or []]
    plugin_requirement = next(
        (
            requirement
            for requirement in requirements
            if requirement.name == _PLUGIN_DISTRIBUTION
            and requirement.marker is not None
            and requirement.marker.evaluate({"extra": "lerobot"})
        ),
        None,
    )

    assert plugin_requirement is not None
    assert "train" in plugin_requirement.extras
    assert plugin_requirement.marker is not None
    assert not plugin_requirement.marker.evaluate({"extra": "cpu"})

    plugin = distribution(_PLUGIN_DISTRIBUTION)
    plugin_requirements = [Requirement(requirement) for requirement in plugin.requires or []]
    engine_dependency = next(
        requirement
        for requirement in plugin_requirements
        if requirement.name == "physicalai-train"
    )
    assert engine_dependency.marker is not None
    assert engine_dependency.marker.evaluate({"extra": "train"})

    lerobot_training = next(
        requirement
        for requirement in plugin_requirements
        if requirement.name == "lerobot" and "training" in requirement.extras
    )
    assert lerobot_training.marker is not None
    assert lerobot_training.marker.evaluate({"extra": "train"})
