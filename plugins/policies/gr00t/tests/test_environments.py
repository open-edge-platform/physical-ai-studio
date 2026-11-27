# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for isolate-only GR00T environment declarations and resolution."""

from __future__ import annotations

import re
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Runs a fixed local import-isolation probe.
import sys
from pathlib import Path
from types import SimpleNamespace

import physicalai.engine
import pytest
from physicalai.engine import to_isolate_environment
from physicalai.isolate import Environment, ExternalInterpreter, GitProject, SyncProfile, UvProvider

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


@pytest.mark.parametrize("spec", [environments.NATIVE, environments.OPENVINO], ids=lambda spec: spec.name)
def test_engine_declarations_prepare_the_same_isolate_environments(spec: environments.WorkerEnvironmentSpec) -> None:
    """The engine and a Runtime-only install must share one cached environment, not build two."""
    worker_env = spec.worker_env()
    assert worker_env in Gr00tEngine.environments()
    assert to_isolate_environment(worker_env, spec.accelerators[0]) == spec.environment


@pytest.mark.parametrize(
    ("spec", "accelerator"),
    [(environments.NATIVE, "cu128"), (environments.OPENVINO, "cpu")],
    ids=["default", "openvino"],
)
def test_prepare_goes_through_the_engine_when_it_is_installed(
    monkeypatch: pytest.MonkeyPatch,
    spec: environments.WorkerEnvironmentSpec,
    accelerator: str,
) -> None:
    prepared = SimpleNamespace(isolate=object())
    calls: list[tuple[object, str]] = []

    def fake_prepare_environment(env: object, selected: str) -> object:
        calls.append((env, selected))
        return prepared

    monkeypatch.setattr(physicalai.engine, "prepare_environment", fake_prepare_environment)
    monkeypatch.setattr(environments, "isolate_prepare", pytest.fail)

    assert environments.prepare(spec) is prepared.isolate
    assert calls == [(spec.worker_env(), accelerator)]


def test_runtime_only_install_prepares_through_isolate_with_the_installed_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Environment, object]] = []

    def fake_prepare(environment: Environment, *, provider: object) -> object:
        calls.append((environment, provider))
        return "prepared"

    monkeypatch.setitem(sys.modules, "physicalai.engine", None)  # Importing it now fails, as without the train extra.
    monkeypatch.setattr(environments, "isolate_prepare", fake_prepare)
    monkeypatch.setattr(environments, "_installed_uv", lambda: "/opt/uv")

    assert environments.prepare(environments.NATIVE) == "prepared"
    ((environment, provider),) = calls
    assert environment == environments.NATIVE.environment
    assert isinstance(provider, UvProvider)
    assert provider.uv == "/opt/uv"


def test_resolve_prepares_managed_environments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prepared = SimpleNamespace(python=tmp_path / "python", project=tmp_path / "project")
    monkeypatch.setattr(environments, "prepare", lambda spec: prepared if spec is environments.NATIVE else None)

    assert environments.resolve(environments.NATIVE) == (prepared, tmp_path / "project", True)


def test_resolve_launches_user_interpreters_and_requires_their_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(environments, "prepare", pytest.fail)
    monkeypatch.setenv("PHYSICALAI_WORKER_PROJECT", "/opt/gr00t")
    assert environments.resolve(environments.NATIVE, interpreter="/opt/py") == (
        ExternalInterpreter("/opt/py"),
        Path("/opt/gr00t"),
        False,
    )

    # A bare name is looked up on PATH, as the engine's runner does, not taken relative to the working directory.
    python = tmp_path / "gr00t-python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    target, _, _ = environments.resolve(environments.NATIVE, interpreter="gr00t-python")
    assert target == ExternalInterpreter(python)
    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        environments.resolve(environments.NATIVE, interpreter="missing-python")

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
