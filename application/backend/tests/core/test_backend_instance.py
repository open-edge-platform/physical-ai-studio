# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the stable per-installation backend instance id."""

from __future__ import annotations

from core.backend_instance import get_backend_instance_id, reset_backend_instance_id_cache
from settings import Settings


def _use_storage_dir(monkeypatch, tmp_path) -> None:
    settings = Settings(STORAGE_DIR=str(tmp_path))
    monkeypatch.setattr("core.backend_instance.get_settings", lambda: settings)
    reset_backend_instance_id_cache()


def test_get_backend_instance_id_is_created_once_and_persisted(monkeypatch, tmp_path) -> None:
    _use_storage_dir(monkeypatch, tmp_path)

    first = get_backend_instance_id()
    reset_backend_instance_id_cache()
    second = get_backend_instance_id()

    assert first == second
    assert (tmp_path / "data" / "backend_instance_id").is_file()


def test_get_backend_instance_id_is_cached_without_rereading_disk(monkeypatch, tmp_path) -> None:
    _use_storage_dir(monkeypatch, tmp_path)

    first = get_backend_instance_id()
    (tmp_path / "data" / "backend_instance_id").write_text("not-a-uuid", encoding="utf-8")
    second = get_backend_instance_id()

    assert first == second


def test_corrupt_file_is_replaced_rather_than_adopted(monkeypatch, tmp_path) -> None:
    _use_storage_dir(monkeypatch, tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "backend_instance_id").write_text("not-a-uuid", encoding="utf-8")

    generated = get_backend_instance_id()

    assert generated != "not-a-uuid"
