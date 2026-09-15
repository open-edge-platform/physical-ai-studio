# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for trainer service settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trainer.settings import TrainerSettings

if TYPE_CHECKING:
    from pathlib import Path


def test_default_storage_dir_is_expanded() -> None:
    settings = TrainerSettings()

    assert "~" not in str(settings.storage_dir)
    assert settings.storage_dir.is_absolute()


def test_storage_dir_override_expands_user(monkeypatch, tmp_path: Path) -> None:
    """
    Regression test: TRAINER_STORAGE_DIR=~/... previously reflected as a
    literal "~" path segment (Field(default=...expanduser()) only expands
    the default, not a value actually supplied via the env var/.env file),
    so downstream directories derived from it (datasets_dir, models_dir,
    etc.) never resolved to the real home directory.
    """
    monkeypatch.setenv("HOME", str(tmp_path))

    settings = TrainerSettings(TRAINER_STORAGE_DIR="~/custom-trainer-storage")

    assert settings.storage_dir == tmp_path / "custom-trainer-storage"
    assert "~" not in str(settings.storage_dir)


def test_storage_dir_override_absolute_path_unaffected(tmp_path: Path) -> None:
    custom = tmp_path / "already-absolute"

    settings = TrainerSettings(TRAINER_STORAGE_DIR=str(custom))

    assert settings.storage_dir == custom


def test_datasets_dir_derives_from_expanded_storage_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    settings = TrainerSettings(TRAINER_STORAGE_DIR="~/custom-trainer-storage")

    assert settings.datasets_dir == tmp_path / "custom-trainer-storage" / "datasets"
