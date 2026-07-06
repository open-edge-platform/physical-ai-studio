# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trainer's ZIP safety wrapper."""

from __future__ import annotations

import zipfile
from typing import TYPE_CHECKING

import pytest

from trainer.archive_safety import (
    InvalidArchiveError,
    SafeZipArchive,
    ZipBombDetectedError,
    flatten_single_root_directory,
)

if TYPE_CHECKING:
    from pathlib import Path

_LARGE_LIMIT = 100 * 1024 * 1024


def _make_zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, mode="w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def test_validate_and_extract_roundtrip(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path / "snap.zip", {"meta/info.json": b"{}", "data/chunk.parquet": b"x"})
    dest = tmp_path / "out"

    safe = SafeZipArchive(archive, max_uncompressed_bytes=_LARGE_LIMIT)
    safe.validate()
    extracted = safe.extract_to(dest)

    assert extracted == 2
    assert (dest / "meta" / "info.json").read_bytes() == b"{}"


def test_path_traversal_entry_rejected(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path / "evil.zip", {"../escape.txt": b"x"})

    safe = SafeZipArchive(archive, max_uncompressed_bytes=_LARGE_LIMIT)
    with pytest.raises(ZipBombDetectedError):
        safe.validate()


def test_nested_zip_rejected(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path / "nested.zip", {"inner.zip": b"x"})

    safe = SafeZipArchive(archive, max_uncompressed_bytes=_LARGE_LIMIT)
    with pytest.raises(ZipBombDetectedError):
        safe.validate()


def test_uncompressed_limit_enforced(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path / "big.zip", {"data.bin": b"x" * 1024})

    safe = SafeZipArchive(archive, max_uncompressed_bytes=512)
    with pytest.raises(ZipBombDetectedError):
        safe.validate()


def test_invalid_zip_raises(tmp_path: Path) -> None:
    not_a_zip = tmp_path / "broken.zip"
    not_a_zip.write_bytes(b"not a zip file")

    safe = SafeZipArchive(not_a_zip, max_uncompressed_bytes=_LARGE_LIMIT)
    with pytest.raises(InvalidArchiveError):
        safe.validate()


def test_flatten_single_root_directory(tmp_path: Path) -> None:
    root = tmp_path / "out"
    nested = root / "dataset"
    nested.mkdir(parents=True)
    (nested / "info.json").write_text("{}")

    flatten_single_root_directory(root)

    assert (root / "info.json").exists()
    assert not nested.exists()
