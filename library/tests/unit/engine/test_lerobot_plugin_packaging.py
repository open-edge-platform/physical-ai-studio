# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Installed entry-point, worker lock, and Runtime adapter contracts."""

from __future__ import annotations

from importlib import resources
from importlib.metadata import distribution, entry_points
import subprocess
import sys
import tomllib

import pytest
from packaging.requirements import Requirement

pytest.importorskip("physicalai_lerobot_policy_plugin", reason="needs the lerobot extra: uv sync --extra lerobot")

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
    assert (package / "envs" / "default" / "pyproject.toml").is_file()
    assert (package / "envs" / "default" / "uv.lock").is_file()
    assert (package / "envs" / "export" / "pyproject.toml").is_file()
    assert (package / "envs" / "export" / "uv.lock").is_file()


def test_worker_lock_is_inside_the_range_declared_by_plugin_metadata() -> None:
    assert locked_version(LeRobotEngine.env) in supported_versions(LeRobotEngine.env)
    export_env = LeRobotEngine.environment("openvino")
    assert locked_version(export_env) in supported_versions(export_env)


def test_worker_project_has_only_locked_framework_dependencies() -> None:
    project = resources.files("physicalai_lerobot_policy_plugin") / "envs" / "default"
    pyproject = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    lock_text = (project / "uv.lock").read_text(encoding="utf-8")
    lock = tomllib.loads(lock_text)
    project_dependencies = {item.split("[", maxsplit=1)[0].split(">", maxsplit=1)[0] for item in pyproject["project"]["dependencies"]}
    lock_packages = {package["name"] for package in lock["package"]}

    assert {"cpu", "cu132", "xpu"} <= set(pyproject["project"]["optional-dependencies"])
    assert "physicalai" not in project_dependencies
    assert "physicalai-train" not in project_dependencies
    assert "physicalai" not in lock_packages
    assert "physicalai-train" not in lock_packages
    assert "https://download.pytorch.org/whl/cpu" in lock_text
    assert "https://download.pytorch.org/whl/cu132" in lock_text
    assert "https://download.pytorch.org/whl/xpu" in lock_text


@pytest.mark.parametrize(
    ("family", "marker_package"),
    [("smolvla", "num2words"), ("pi", "scipy"), ("molmoact2", "peft")],
)
def test_each_policy_family_installs_its_own_extra_of_the_shared_worker_project(family: str, marker_package: str) -> None:
    """Each family gets its own environment from the one locked project, through its extra."""
    env = LeRobotEngine.environment_for_policy({"smolvla": "smolvla", "pi": "pi05", "molmoact2": "molmoact2"}[family])
    project = resources.files("physicalai_lerobot_policy_plugin") / "envs" / "default"
    pyproject = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    lock_packages = {package["name"] for package in tomllib.loads((project / "uv.lock").read_text(encoding="utf-8"))["package"]}
    families = {"smolvla", "pi", "molmoact2"}

    assert env.name == family
    assert env.extras_for("cuda") == ("cu132", family)
    assert env.extras_for("cpu") == ("cpu", family)
    assert LeRobotEngine.env.extras_for("cuda") == ("cu132",)
    assert env.key() != LeRobotEngine.env.key()
    assert locked_version(env) in supported_versions(env)
    assert pyproject["project"]["optional-dependencies"][family] == [f"lerobot[{family}]>=0.7,<0.8"]
    assert [{"extra": name} for name in sorted(families)] in [
        sorted(group, key=lambda item: item["extra"]) for group in pyproject["tool"]["uv"]["conflicts"]
    ]
    assert marker_package in lock_packages


def test_bundled_worker_projects_declare_default_and_export_locks() -> None:
    plugin = resources.files("physicalai_lerobot_policy_plugin")
    assert (plugin / "envs" / "default" / "uv.lock").is_file()
    assert (plugin / "envs" / "export" / "uv.lock").is_file()
    assert {env.name for env in LeRobotEngine.environments()} == {"default", "smolvla", "pi", "molmoact2", "export"}


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
