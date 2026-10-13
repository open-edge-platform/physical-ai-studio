# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Public API tests for engine framework discovery and dispatch."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from physicalai.engine import Engine, FrameworkRegistry, Run


class _EntryPoint:
    def __init__(self, name: str, register: Any) -> None:
        self.name = name
        self._register = register

    def load(self) -> Any:
        return self._register


class _BrokenEntryPoint:
    def __init__(self, name: str) -> None:
        self.name = name

    def load(self) -> Any:
        raise ImportError("optional plugin import failed")


class _Framework:
    def __init__(self, name: str, accepted_config: object) -> None:
        self.name = name
        self.accepted_config = accepted_config
        self.received_config: object | None = None
        self.received_hooks: object | None = None

    def accepts(self, config: object) -> bool:
        return config is self.accepted_config

    def train(self, config: object, hooks: object) -> Run:
        self.received_config = config
        self.received_hooks = hooks
        return Run(framework=self.name, output_dir=Path("run"), checkpoint=Path("run/checkpoint"))

    def export(self, source: object, output_dir: Path, format: str | None = None) -> Path:  # noqa: ARG002
        return output_dir


def test_discovered_frameworks_are_listed_and_engine_dispatches_publicly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builtin_config = object()
    external_config = object()
    builtin = _Framework("native", builtin_config)
    external = _Framework("external", external_config)

    def register_builtin(registry: FrameworkRegistry) -> None:
        registry.register(builtin)

    def register_external(registry: FrameworkRegistry) -> None:
        registry.register(external)

    entries = [
        _EntryPoint("native", register_builtin),
        _EntryPoint("external", register_external),
    ]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda *, group: entries)

    registry = FrameworkRegistry.discover()
    engine = Engine(registry=registry)
    hooks = object()

    run = engine.train(external_config, hooks=hooks)

    assert engine.frameworks() == ["external", "native"]
    assert run.framework == "external"
    assert external.received_config is external_config
    assert external.received_hooks is hooks


def test_broken_framework_import_does_not_hide_healthy_plugins() -> None:
    def register_healthy(registry: FrameworkRegistry) -> None:
        registry.register(_Framework("healthy", object()))

    registry = FrameworkRegistry.discover(
        [_BrokenEntryPoint("a-broken"), _EntryPoint("z-healthy", register_healthy)]
    )

    assert registry.names() == ["healthy"]


def test_partial_framework_registration_is_rolled_back_before_loading_healthy_plugins() -> None:
    def register_broken(registry: FrameworkRegistry) -> None:
        registry.register(_Framework("partial", object()))
        raise RuntimeError("optional plugin registration failed")

    def register_healthy(registry: FrameworkRegistry) -> None:
        registry.register(_Framework("healthy", object()))

    registry = FrameworkRegistry.discover(
        [_EntryPoint("a-broken", register_broken), _EntryPoint("z-healthy", register_healthy)]
    )

    assert registry.names() == ["healthy"]


def test_duplicate_framework_names_are_rejected_deterministically() -> None:
    def register_first(registry: FrameworkRegistry) -> None:
        registry.register(_Framework("duplicate", object()))

    def register_second(registry: FrameworkRegistry) -> None:
        registry.register(_Framework("duplicate", object()))

    first_order = [_EntryPoint("z-provider", register_second), _EntryPoint("a-provider", register_first)]
    reverse_order = list(reversed(first_order))
    errors = []
    for entries in (first_order, reverse_order):
        with pytest.raises(ValueError, match="Duplicate framework") as exc_info:
            FrameworkRegistry.discover(entries)
        errors.append(str(exc_info.value))

    assert errors[0] == errors[1]


def test_engine_import_and_builtin_discovery_keep_model_dependencies_lazy() -> None:
    code = (
        "import sys; from physicalai.engine import Engine; "
        "assert 'lerobot' in Engine().frameworks(); "
        "heavy = ('torch', 'lerobot', 'lightning', 'physicalai.inference'); "
        "assert not [name for name in sys.modules if any(name == item or name.startswith(item + '.') for item in heavy)]"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_registry_rejects_duplicate_names() -> None:
    registry = FrameworkRegistry()
    config = object()
    registry.register(_Framework("lerobot", config))

    with pytest.raises(ValueError, match="Duplicate framework.*lerobot"):
        registry.register(_Framework("lerobot", object()))


def test_engine_requires_unambiguous_config_or_known_framework() -> None:
    config = object()
    registry = FrameworkRegistry()
    registry.register(_Framework("first", config))
    registry.register(_Framework("second", config))
    engine = Engine(registry=registry)

    with pytest.raises(ValueError, match="ambiguous.*framework", check=lambda error: "first" in str(error)):
        engine.train(config)

    with pytest.raises(ValueError, match="Unknown framework.*missing"):
        engine.train(config, framework="missing")

    with pytest.raises(ValueError, match="No registered framework accepts"):
        engine.train(object())
