# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hardware-aware integration target selection tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from physicalai.engine import _integration


def test_detect_gpu_hardware_uses_supported_pci_devices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def add_device(name: str, *, vendor: str, class_code: str) -> None:
        device = tmp_path / name
        device.mkdir()
        (device / "vendor").write_text(vendor, encoding="ascii")
        (device / "class").write_text(class_code, encoding="ascii")

    add_device("nvidia", vendor="0x10de", class_code="0x030000")
    add_device("intel-display", vendor="0x8086", class_code="0x038000")
    add_device("intel-audio", vendor="0x8086", class_code="0x040000")
    add_device("amd-display", vendor="0x1002", class_code="0x030000")
    monkeypatch.setattr(_integration, "_command_reports_gpu", Mock(return_value=False))

    assert _integration.detect_gpu_hardware(tmp_path) == frozenset({"cuda", "xpu"})


def test_probe_torch_devices_checks_the_selected_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock(return_value=Mock(returncode=0, stdout='{"cuda":true,"xpu":false}', stderr=""))
    monkeypatch.setattr(_integration.subprocess, "run", run)

    availability = _integration.probe_torch_devices("/venv/bin/python")

    assert availability == _integration.TorchDeviceAvailability(cuda=True, xpu=False)
    run.assert_called_once()
    assert run.call_args.args[0][0] == "/venv/bin/python"
    assert run.call_args.kwargs["shell"] is False


def test_auto_selects_usable_xpu_before_cuda() -> None:
    target = _integration.select_integration_target(
        "auto",
        detected_hardware={"xpu", "cuda"},
        torch_devices={"xpu": True, "cuda": True},
    )

    assert target.device == "xpu"
    assert target.uv_extra == "xpu"


def test_auto_selects_cuda_when_xpu_is_not_usable() -> None:
    target = _integration.select_integration_target(
        "auto",
        detected_hardware={"xpu", "cuda"},
        torch_devices={"xpu": False, "cuda": True},
    )

    assert target.device == "cuda"
    assert target.uv_extra == "cu128"


def test_auto_selects_cpu_only_when_no_gpu_is_detected_or_usable() -> None:
    target = _integration.select_integration_target(
        "auto",
        detected_hardware=set(),
        torch_devices={"xpu": False, "cuda": False},
    )

    assert target.device == "cpu"
    assert target.uv_extra == "cpu"


def test_explicit_cpu_override_is_honored_when_gpu_hardware_exists() -> None:
    target = _integration.select_integration_target(
        "cpu",
        detected_hardware={"cuda"},
        torch_devices={"xpu": False, "cuda": False},
    )

    _integration.require_torch_device(
        target,
        torch_devices=_integration.TorchDeviceAvailability(cuda=False, xpu=False),
        detected_hardware={"cuda"},
        interpreter=Path("/venv/bin/python"),
    )
    assert target.device == "cpu"
    assert target.uv_extra == "cpu"


def test_detected_gpu_with_unusable_host_interpreter_fails_with_setup_guidance() -> None:
    target = _integration.select_integration_target(
        "auto",
        detected_hardware={"cuda"},
        torch_devices={"xpu": False, "cuda": False},
    )

    with pytest.raises(RuntimeError, match="CUDA hardware was detected.*cu128.*--device cpu"):
        _integration.require_torch_device(
            target,
            torch_devices=_integration.TorchDeviceAvailability(cuda=False, xpu=False),
            detected_hardware={"cuda"},
            interpreter=Path("/venv/bin/python"),
        )


def test_explicit_gpu_override_maps_to_uv_extra_and_requires_a_usable_runtime() -> None:
    target = _integration.select_integration_target(
        "cuda",
        detected_hardware=set(),
        torch_devices={"xpu": False, "cuda": True},
    )
    assert target.device == "cuda"
    assert target.uv_extra == "cu128"

    with pytest.raises(RuntimeError, match="CUDA was explicitly selected"):
        _integration.require_torch_device(
            target,
            torch_devices=_integration.TorchDeviceAvailability(cuda=False, xpu=False),
            detected_hardware=set(),
            interpreter=Path("/venv/bin/python"),
        )


def test_unknown_device_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        _integration.select_integration_target(
            "mps",
            detected_hardware=set(),
            torch_devices={"xpu": False, "cuda": False},
        )
