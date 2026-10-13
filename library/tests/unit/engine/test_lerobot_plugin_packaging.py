# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Installed distribution contract for the standalone LeRobot policy plugin."""

from __future__ import annotations

from importlib.metadata import distribution, entry_points
import subprocess
import sys

from packaging.requirements import Requirement

_PLUGIN_DISTRIBUTION = "physicalai-lerobot-policy-plugin"


def test_lerobot_plugin_owns_framework_and_runtime_entrypoints() -> None:
    plugin = distribution(_PLUGIN_DISTRIBUTION)
    framework_entries = [
        entry
        for entry in entry_points(group="physicalai.engine.frameworks")
        if entry.name == "lerobot" and entry.dist is not None and entry.dist.name == _PLUGIN_DISTRIBUTION
    ]
    adapter_entries = [
        entry
        for entry in entry_points(group="physicalai.inference.adapters")
        if entry.name == "lerobot" and entry.dist is not None and entry.dist.name == _PLUGIN_DISTRIBUTION
    ]

    assert len(framework_entries) == 1
    assert framework_entries[0].value == "physicalai_lerobot_policy_plugin.framework:register"
    assert len(adapter_entries) == 1
    assert adapter_entries[0].value == "physicalai_lerobot_policy_plugin.inference:register"

    requirements = [requirement.lower() for requirement in plugin.requires or []]
    assert not any("physicalai-train" in requirement for requirement in requirements)


def test_runtime_registration_import_does_not_load_the_training_engine() -> None:
    code = (
        "import sys; import physicalai_lerobot_policy_plugin.inference; "
        "assert not any(name == 'physicalai.engine' or name.startswith('physicalai.engine.') for name in sys.modules); "
        "assert 'lerobot' not in sys.modules and 'torch' not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_physicalai_train_lerobot_extra_installs_the_standalone_plugin() -> None:
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
    lerobot_training = next(
        requirement
        for requirement in plugin_requirements
        if requirement.name == "lerobot" and "training" in requirement.extras
    )
    assert lerobot_training.marker is not None
    assert lerobot_training.marker.evaluate({"extra": "train"})

    diffusion_training = next(
        requirement
        for requirement in plugin_requirements
        if requirement.name == "diffusers"
    )
    assert diffusion_training.marker is not None
    assert diffusion_training.marker.evaluate({"extra": "train"})

    lerobot = distribution("lerobot")
    lerobot_requirements = [Requirement(requirement) for requirement in lerobot.requires or []]
    assert any(
        requirement.name == "wandb"
        and requirement.marker is not None
        and requirement.marker.evaluate({"extra": "training"})
        for requirement in lerobot_requirements
    )
    assert any(
        requirement.name == "accelerate"
        and requirement.marker is not None
        and requirement.marker.evaluate({"extra": "accelerate-dep"})
        for requirement in lerobot_requirements
    )
