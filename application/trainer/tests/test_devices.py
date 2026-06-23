# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for trainer hardware discovery and the /devices endpoint."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from trainer import devices as devices_module


def _torch_stub(*, cuda: list[tuple[str, int]] | None = None, xpu: list[tuple[str, int]] | None = None) -> MagicMock:
    cuda = cuda or []
    xpu = xpu or []
    torch = MagicMock()
    torch.xpu.is_available.return_value = bool(xpu)
    torch.xpu.device_count.return_value = len(xpu)
    torch.xpu.get_device_properties.side_effect = lambda i: SimpleNamespace(name=xpu[i][0], total_memory=xpu[i][1])
    torch.cuda.is_available.return_value = bool(cuda)
    torch.cuda.device_count.return_value = len(cuda)
    torch.cuda.get_device_properties.side_effect = lambda i: SimpleNamespace(name=cuda[i][0], total_memory=cuda[i][1])
    return torch


def test_get_training_devices_reports_cpu_only_without_accelerators(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _torch_stub())

    result = devices_module.get_training_devices()

    assert [d.type for d in result] == ["cpu"]
    assert result[0].name == "CPU"
    assert result[0].memory is None


def test_get_training_devices_reports_cuda(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _torch_stub(cuda=[("NVIDIA A100", 42949672960)]))

    result = devices_module.get_training_devices()

    assert [d.type for d in result] == ["cpu", "cuda"]
    gpu = result[1]
    assert gpu.name == "NVIDIA A100"
    assert gpu.memory == 42949672960
    assert gpu.index == 0


def test_devices_endpoint_returns_device_list(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from trainer import main

    monkeypatch.setattr(
        main,
        "get_training_devices",
        lambda: [main.DeviceInfo(type="cuda", name="NVIDIA A100", memory=42949672960, index=0)],
    )

    # No context manager: the /devices route needs no app lifespan/queue manager.
    client = TestClient(main.app)
    response = client.get("/devices")

    assert response.status_code == 200
    body = response.json()
    assert body == [{"type": "cuda", "name": "NVIDIA A100", "memory": 42949672960, "index": 0}]
