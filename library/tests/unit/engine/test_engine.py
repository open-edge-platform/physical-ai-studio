# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Public Engine API and entry-point discovery tests."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest

from physicalai.engine import Enablement, Engine, Run, TrainHooks


class _EntryPoint:
    def __init__(self, name: str, engine_class: type[Engine]) -> None:
        self.name = name
        self._engine_class = engine_class
        self.loaded = False

    def load(self) -> type[Engine]:
        self.loaded = True
        return self._engine_class


class _FakeEngine(Engine):
    framework: ClassVar[str] = "fake"
    env = None
    training_accelerators = ("cpu",)
    stock_export_formats = ("native",)
    enablements: ClassVar[tuple[Enablement, ...]] = (
        Enablement(
            name="upstream-xpu",
            provides={"tiny": {"train": ("xpu",), "export": ("onnx",)}},
            upstream="pytorch-xpu@2.11.0",
        ),
    )
    calls: ClassVar[list[tuple[str, object]]] = []

    def __init__(self, *, value: str = "default") -> None:
        self.value = value

    @classmethod
    def cli(
        cls,
        argv: Sequence[str],
        *,
        interpreter: str | Path | None = None,
        accelerator: str = "auto",
    ) -> int:
        cls.calls.append(("cli", (list(argv), interpreter, accelerator)))
        return 7

    @classmethod
    def run(
        cls,
        settings: Mapping[str, Any],
        *,
        dataset: Path,
        output_dir: Path,
        device: str,
        hooks: TrainHooks | None = None,
        interpreter: str | Path | None = None,
    ) -> Run:
        cls.calls.append(("run", (settings, dataset, output_dir, device, hooks, interpreter)))
        return Run(framework=cls.framework, output_dir=output_dir, checkpoint=None)

    @classmethod
    def export(
        cls,
        source: Run | str | Path,
        output_path: str | Path,
        backend: str | None = None,
    ) -> Path:
        cls.calls.append(("export", (source, output_path, backend)))
        return Path(output_path)

    @classmethod
    def policies(cls) -> list[str]:
        return ["tiny"]

    @classmethod
    def config_schema(cls, policy: str) -> dict[str, Any]:
        return {"title": policy, "type": "object"}

def test_engine_discovery_lists_entry_points_without_importing_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physicalai.engine import _registry

    entry = _EntryPoint("fake", _FakeEngine)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda *, group: [entry])
    monkeypatch.setattr(_registry, "_entry_points", lambda: [entry])

    assert Engine.names() == ["fake"]
    assert not entry.loaded


def test_engine_get_and_create_return_the_framework_specific_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physicalai.engine import _registry

    entry = _EntryPoint("fake", _FakeEngine)
    monkeypatch.setattr(_registry, "_entry_points", lambda: [entry])

    assert Engine.get("fake") is _FakeEngine
    engine = Engine.create("fake", value="configured")
    assert isinstance(engine, _FakeEngine)
    assert engine.value == "configured"


def test_engine_rejects_duplicate_names_and_entry_point_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physicalai.engine import _registry

    duplicate_entries = [_EntryPoint("fake", _FakeEngine), _EntryPoint("fake", _FakeEngine)]
    monkeypatch.setattr(_registry, "_entry_points", lambda: duplicate_entries)
    with pytest.raises(ValueError, match="Duplicate engine"):
        Engine.names()

    mismatched = _EntryPoint("not-fake", _FakeEngine)
    monkeypatch.setattr(_registry, "_entry_points", lambda: [mismatched])
    with pytest.raises(ValueError, match="framework.*does not match"):
        Engine.get("not-fake")


def test_engine_import_and_listing_do_not_load_framework_dependencies() -> None:
    code = (
        "import sys; from physicalai.engine import Engine; Engine.names(); "
        "heavy = ('torch', 'lerobot', 'lightning', 'physicalai.inference'); "
        "assert not [name for name in sys.modules if any(name == item or name.startswith(item + '.') for item in heavy)]"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_unknown_engine_error_lists_available_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    from physicalai.engine import _registry

    monkeypatch.setattr(_registry, "_entry_points", lambda: [_EntryPoint("fake", _FakeEngine)])

    with pytest.raises(ValueError, match="Unknown engine 'missing'.*fake"):
        Engine.get("missing")


def test_engine_capabilities_combine_stock_and_validated_enablements(monkeypatch: pytest.MonkeyPatch) -> None:
    from physicalai.engine import _registry

    monkeypatch.setattr(_registry, "_entry_points", lambda: [_EntryPoint("fake", _FakeEngine)])

    capabilities = Engine.get("fake").capabilities("tiny")

    assert capabilities.training_accelerators == ("cpu", "xpu")
    assert capabilities.export_formats == ("native", "onnx")
    assert capabilities.enablements == ("upstream-xpu",)


def test_run_and_cli_are_framework_class_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    from physicalai.engine import _registry

    monkeypatch.setattr(_registry, "_entry_points", lambda: [_EntryPoint("fake", _FakeEngine)])
    _FakeEngine.calls.clear()
    run = Engine.get("fake").run(
        {"policy": {"type": "tiny"}},
        dataset=Path("dataset"),
        output_dir=Path("output"),
        device="cpu",
    )

    assert run.framework == "fake"
    assert Engine.get("fake").cli(["--native", "flag"], accelerator="xpu") == 7
    assert _FakeEngine.calls == [
        ("run", ({"policy": {"type": "tiny"}}, Path("dataset"), Path("output"), "cpu", None, None)),
        ("cli", (["--native", "flag"], None, "xpu")),
    ]
