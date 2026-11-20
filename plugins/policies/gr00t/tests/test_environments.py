# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for isolate-only GR00T environment declarations and resolution."""

from __future__ import annotations

import re
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Runs a fixed local import-isolation probe.
import sys
from pathlib import Path

import pytest
from physicalai.isolate import Environment, GitProject, SyncProfile

from physicalai_gr00t_policy_plugin import environments
from physicalai_gr00t_policy_plugin.engine import (
    NATIVE as NATIVE_WORKER_ENV,
)
from physicalai_gr00t_policy_plugin.engine import (
    OPENVINO as OPENVINO_WORKER_ENV,
)
from physicalai_gr00t_policy_plugin.engine import (
    OPENVINO_ENABLEMENT,
    Gr00tEngine,
)

_SHA1 = re.compile(r"[0-9a-f]{40}")


def test_runtime_environment_declarations_use_isolate_sources_and_profiles() -> None:
    series = environments.ROBOTICS_AI_SUITE_OPENVINO
    assert series.revision == "2960959ae6cc6d45150316ed6612f686c2d915b7"
    assert series.tree == "d657bcb89b8937495b8c849d7cfb8c9f3d1e5601"
    assert [patch.name for patch in series.files] == sorted(patch.name for patch in series.files)
    assert len(series.files) == 8
    assert series.url(series.files[0]) == (
        "https://raw.githubusercontent.com/open-edge-platform/robotics-ai-suite/"
        "2960959ae6cc6d45150316ed6612f686c2d915b7/src/pipelines/gr00t-n1d7-ov/patches/"
        "0001-Add-OpenVINO-model-convert-inference-code.patch"
    )

    native, openvino = environments.NATIVE, environments.OPENVINO
    assert isinstance(native, environments.WorkerEnvironmentSpec)
    assert isinstance(native.environment, Environment)
    assert isinstance(native.source, GitProject)
    assert isinstance(openvino.source, GitProject)
    assert native.source.revision == openvino.source.revision == "3df8b3825d67f755e69141446f4315f281b9b7e6"
    assert native.source.layers == ()
    assert openvino.source.layers == (series,)
    assert (native.name, native.profile, native.accelerators, native.key() == openvino.key()) == (
        "default",
        SyncProfile(python="3.10"),
        ("cu128",),
        False,
    )
    assert (openvino.name, openvino.profile, openvino.accelerators) == (
        "robotics-ai-suite-openvino",
        SyncProfile(python="3.12"),
        ("cpu",),
    )
    assert {"native": native, "openvino": openvino} == environments.PACKAGE_ENVIRONMENTS
    assert _SHA1.fullmatch(native.source.revision)


def test_engine_adapts_runtime_declarations_to_worker_env_and_enablement() -> None:
    assert Gr00tEngine.environments() == (NATIVE_WORKER_ENV, OPENVINO_WORKER_ENV)
    assert Gr00tEngine.environment("robotics-ai-suite-openvino") is OPENVINO_WORKER_ENV
    assert OPENVINO_ENABLEMENT.env is OPENVINO_WORKER_ENV
    assert environments.NATIVE.key() == NATIVE_WORKER_ENV.key()
    assert environments.OPENVINO.key() == OPENVINO_WORKER_ENV.key()
    assert NATIVE_WORKER_ENV.project is environments.NATIVE.source
    assert OPENVINO_WORKER_ENV.project is environments.OPENVINO.source


def test_provenance_records_isolate_environment_key_and_source_pins() -> None:
    record = environments.provenance(environments.OPENVINO)
    assert record["name"] == "robotics-ai-suite-openvino"
    assert record["key"] == environments.OPENVINO.key()
    assert record["source"]["patches"][0]["kind"] == "published"
    assert record["source"]["patches"][0]["tree"] == "d657bcb89b8937495b8c849d7cfb8c9f3d1e5601"


def test_resolve_prepares_with_isolate_and_accepts_user_checkouts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = type("Prepared", (), {"python": tmp_path / "python", "project": tmp_path / "project"})()
    calls: list[Environment] = []

    def fake_prepare(environment: Environment) -> object:
        calls.append(environment)
        return prepared

    monkeypatch.setattr(environments, "isolate_prepare", fake_prepare)
    assert environments.resolve(environments.NATIVE) == (tmp_path / "python", tmp_path / "project", True)
    assert calls == [environments.NATIVE.environment]

    monkeypatch.setenv("PHYSICALAI_WORKER_PROJECT", "/opt/gr00t")
    assert environments.resolve(environments.NATIVE, interpreter="/opt/py") == (
        Path("/opt/py"),
        Path("/opt/gr00t"),
        False,
    )
    monkeypatch.delenv("PHYSICALAI_WORKER_PROJECT")
    with pytest.raises(ValueError, match="Isaac-GR00T checkout"):
        environments.resolve(environments.NATIVE, interpreter="/opt/py")


def test_runtime_environment_modules_import_without_physicalai_engine() -> None:
    package_source = Path(__file__).parents[1] / "src"
    code = f"""
import builtins
import sys

sys.path.insert(0, {str(package_source)!r})
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "physicalai.engine" or name.startswith("physicalai.engine."):
        raise AssertionError("Runtime plugin imported physicalai.engine")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import physicalai_gr00t_policy_plugin.environments
import physicalai_gr00t_policy_plugin.runtime
assert not any(name == "physicalai.engine" or name.startswith("physicalai.engine.") for name in sys.modules)
"""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # Runs a fixed local import probe.
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        shell=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
