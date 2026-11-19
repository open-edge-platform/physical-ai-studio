# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot integration-test device selection checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.integration import _device


def _torch(*, cuda: bool, xpu: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        xpu=SimpleNamespace(is_available=lambda: xpu),
    )


def test_auto_uses_core_selected_xpu_when_host_torch_can_use_it(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[str] = []

    def resolve(requested: str) -> str:
        requests.append(requested)
        return "xpu"

    monkeypatch.setattr(_device, "resolve_accelerator", resolve)

    assert _device.select_lerobot_test_device("auto", torch_module=_torch(cuda=True, xpu=True)) == "xpu"
    assert requests == ["auto"]


def test_auto_maps_cuda_extra_to_native_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_device, "resolve_accelerator", lambda requested: "cu128")

    assert _device.select_lerobot_test_device("auto", torch_module=_torch(cuda=True, xpu=False)) == "cuda"


def test_gpu_hardware_selection_fails_when_host_torch_cannot_use_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_device, "resolve_accelerator", lambda requested: "cu128")

    with pytest.raises(RuntimeError, match="selected cuda.*PHYSICALAI_LEROBOT_DEVICE=cpu"):
        _device.select_lerobot_test_device("auto", torch_module=_torch(cuda=False, xpu=False))


def test_explicit_cpu_override_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[str] = []

    def resolve(requested: str) -> str:
        requests.append(requested)
        return "cpu"

    monkeypatch.setattr(_device, "resolve_accelerator", resolve)

    assert _device.select_lerobot_test_device("cpu", torch_module=_torch(cuda=False, xpu=False)) == "cpu"
    assert requests == ["cpu"]


def test_explicit_cuda_requires_a_usable_host_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_device, "resolve_accelerator", lambda requested: "cu128")

    with pytest.raises(RuntimeError, match="explicitly requested cuda"):
        _device.select_lerobot_test_device("cuda", torch_module=_torch(cuda=False, xpu=False))
