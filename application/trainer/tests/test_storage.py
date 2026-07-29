# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for trainer storage discovery and the /storage endpoint."""

from __future__ import annotations

from types import SimpleNamespace

from trainer import storage as storage_module


def test_get_storage_info_reports_disk_usage(monkeypatch, tmp_path) -> None:
    settings = storage_module.get_settings()
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    monkeypatch.setattr(
        storage_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=1000, used=400, free=600),
    )

    result = storage_module.get_storage_info()

    assert result.total_bytes == 1000
    assert result.free_bytes == 600
    assert (tmp_path / "storage").is_dir()


def test_storage_endpoint_returns_storage_info(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from trainer import main

    monkeypatch.setattr(
        main,
        "get_storage_info",
        lambda: main.StorageInfo(total_bytes=1000, free_bytes=600),
    )

    # No context manager: the /storage route needs no app lifespan/queue manager.
    client = TestClient(main.app)
    response = client.get("/storage")

    assert response.status_code == 200
    assert response.json() == {"total_bytes": 1000, "free_bytes": 600}
