# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103, PLR2004, S101

"""Unit tests for archive safety utilities."""

from __future__ import annotations

import stat
import zipfile
from typing import TYPE_CHECKING, NamedTuple

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from physicalai.data import archive_safety
from physicalai.data.archive_safety import (
    ArchiveErrorTypes,
    InsufficientDiskSpaceError,
    InvalidArchiveError,
    SafeZipArchive,
    ZipBombDetectedError,
    check_disk_headroom,
    flatten_single_root_directory,
)

_LARGE_LIMIT = 100 * 1024 * 1024


class _DiskUsage(NamedTuple):
    total: int
    used: int
    free: int


def _make_zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, mode="w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def test_validate_and_extract_roundtrip(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path / "snap.zip", {"meta/info.json": b"{}", "data/chunk.parquet": b"x"})
    destination = tmp_path / "out"

    safe = SafeZipArchive(archive, max_uncompressed_bytes=_LARGE_LIMIT)
    safe.validate()

    assert safe.extract_to(destination) == 2
    assert (destination / "meta" / "info.json").read_bytes() == b"{}"


def test_path_traversal_entry_rejected(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path / "evil.zip", {"../escape.txt": b"x"})

    with pytest.raises(ZipBombDetectedError):
        SafeZipArchive(archive, max_uncompressed_bytes=_LARGE_LIMIT).validate()


def test_symlink_entry_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "symlink.zip"
    member = zipfile.ZipInfo("link")
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr(member, "target")

    with pytest.raises(ZipBombDetectedError, match="symlink"):
        SafeZipArchive(archive_path, max_uncompressed_bytes=_LARGE_LIMIT).validate()


def test_nested_zip_rejected(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path / "nested.zip", {"inner.zip": b"x"})

    with pytest.raises(ZipBombDetectedError, match="nested zip"):
        SafeZipArchive(archive, max_uncompressed_bytes=_LARGE_LIMIT).validate()


def test_uncompressed_limit_enforced(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path / "big.zip", {"data.bin": b"x" * 1024})

    with pytest.raises(ZipBombDetectedError, match="uncompressed size"):
        SafeZipArchive(archive, max_uncompressed_bytes=512).validate()


def test_invalid_zip_raises(tmp_path: Path) -> None:
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"not a zip file")

    with pytest.raises(InvalidArchiveError):
        SafeZipArchive(archive, max_uncompressed_bytes=_LARGE_LIMIT).validate()


def test_read_helpers_support_a_wrapped_dataset_root(tmp_path: Path) -> None:
    archive = _make_zip(
        tmp_path / "dataset.zip",
        {
            "dataset/meta/info.json": b'{"robot_type": "test"}',
            "dataset/meta/episodes.jsonl": b'{"episode": 0}\n{"episode": 1}\n',
        },
    )
    safe = SafeZipArchive(archive, max_uncompressed_bytes=_LARGE_LIMIT)

    assert safe.resolve_member_name("meta/info.json") == "dataset/meta/info.json"
    assert safe.read_json("meta/info.json") == {"robot_type": "test"}
    assert list(safe.read_jsonl("meta/episodes.jsonl")) == [{"episode": 0}, {"episode": 1}]
    assert safe.read_bytes("missing") is None


def test_error_types_allow_application_specific_exceptions(tmp_path: Path) -> None:
    class ApplicationArchiveError(Exception):
        pass

    archive = _make_zip(tmp_path / "evil.zip", {"../escape.txt": b"x"})
    errors = ArchiveErrorTypes(
        invalid_archive=ApplicationArchiveError,
        zip_bomb_detected=ApplicationArchiveError,
        insufficient_disk_space=ApplicationArchiveError,
    )

    with pytest.raises(ApplicationArchiveError):
        SafeZipArchive(archive, max_uncompressed_bytes=_LARGE_LIMIT, error_types=errors).validate()


def test_check_disk_headroom_raises_configured_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(archive_safety.shutil, "disk_usage", lambda _directory: _DiskUsage(100, 95, 5))

    with pytest.raises(InsufficientDiskSpaceError, match="need 11"):
        check_disk_headroom(tmp_path, required_bytes=10, min_free_bytes=1)


def test_flatten_single_root_directory_ignores_macos_junk(tmp_path: Path) -> None:
    root = tmp_path / "out"
    nested = root / "dataset"
    (nested / "meta").mkdir(parents=True)
    (nested / "meta" / "info.json").write_text("{}")
    (root / "__MACOSX" / "dataset").mkdir(parents=True)
    (root / "__MACOSX" / "dataset" / "._info.json").write_text("junk")
    (root / ".DS_Store").write_text("junk")
    (root / "._dataset").write_text("junk")

    flatten_single_root_directory(root)

    assert (root / "meta" / "info.json").exists()
    assert not nested.exists()
    assert not (root / "__MACOSX").exists()
    assert not (root / ".DS_Store").exists()
    assert not (root / "._dataset").exists()
