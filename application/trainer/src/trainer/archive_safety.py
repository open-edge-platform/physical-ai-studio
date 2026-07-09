# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Security-aware ZIP handling for uploaded dataset archives.

Mirrors the studio backend's ``archive_safety`` so the trainer can validate and
extract uploads without depending on the backend package. Guards against zip
bombs, path traversal, symlinks, and nested archives, and checks disk headroom
before extraction.
"""

from __future__ import annotations

import shutil
import stat
from pathlib import Path
from zipfile import BadZipFile, ZipFile, ZipInfo

DEFAULT_MAX_FILE_COUNT = 200_000

_NOT_A_ZIP_MSG = "Uploaded file is not a valid ZIP archive"


class InvalidArchiveError(Exception):
    """Raised when an uploaded archive is not a readable ZIP."""


class ZipBombDetectedError(Exception):
    """Raised when an archive violates a safety limit or contains an unsafe entry."""


class InsufficientDiskSpaceError(Exception):
    """Raised when the target filesystem lacks headroom for the extraction."""


def _normalize_zip_member_name(name: str) -> str:
    return name.replace("\\", "/").strip("/").removeprefix("./")


def _is_symlink(member_external_attr: int) -> bool:
    """Return True when the member's mode bits mark it a symlink."""
    mode = member_external_attr >> 16
    return (mode & stat.S_IFLNK) == stat.S_IFLNK


def validate_zip_entries(
    members: list[ZipInfo],
    *,
    max_file_count: int | None,
    max_uncompressed_bytes: int,
) -> int:
    """Validate ZIP entries against safety limits; return total uncompressed bytes."""
    if max_file_count is None:
        max_file_count = DEFAULT_MAX_FILE_COUNT

    if len(members) > max_file_count:
        msg = f"Archive contains too many entries ({len(members)} > {max_file_count})"
        raise ZipBombDetectedError(msg)

    total_uncompressed = sum(member.file_size for member in members)
    if total_uncompressed > max_uncompressed_bytes:
        msg = f"Archive uncompressed size exceeds allowed limit ({total_uncompressed} > {max_uncompressed_bytes} bytes)"
        raise ZipBombDetectedError(msg)

    for member in members:
        name = _normalize_zip_member_name(member.filename)
        if _is_symlink(member.external_attr):
            msg = f"Archive contains symlink entry '{name}', which is not allowed"
            raise ZipBombDetectedError(msg)

        normalized_path = Path(name)
        if normalized_path.is_absolute() or ".." in normalized_path.parts:
            msg = f"Archive contains unsafe entry path '{member.filename}'"
            raise ZipBombDetectedError(msg)

        if normalized_path.suffix.lower() == ".zip":
            msg = f"Archive contains nested zip entry '{member.filename}', which is not allowed"
            raise ZipBombDetectedError(msg)

    return total_uncompressed


def check_disk_headroom(directory: Path, required_bytes: int, min_free_bytes: int) -> None:
    """Ensure *directory*'s filesystem keeps *min_free_bytes* free after writing *required_bytes*."""
    directory.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(directory)
    needed = required_bytes + min_free_bytes
    if usage.free < needed:
        msg = f"Insufficient disk space on '{directory}': {usage.free} bytes free, need {needed} bytes"
        raise InsufficientDiskSpaceError(msg)


class SafeZipArchive:
    """Validate a ZIP once, then extract it with per-entry path containment."""

    def __init__(
        self,
        archive_path: str | Path,
        *,
        max_uncompressed_bytes: int,
        max_file_count: int | None = None,
    ) -> None:
        """Store the archive path and the safety limits to enforce."""
        self.path = Path(archive_path)
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.max_file_count = max_file_count
        self._validated_members: list[ZipInfo] | None = None

    def validate(self) -> None:
        """Run the one-time safety pass over archive entries."""
        self._get_validated_members()

    def estimated_uncompressed_size(self) -> int:
        """Return total uncompressed bytes across validated members."""
        return sum(member.file_size for member in self._get_validated_members())

    def extract_to(self, destination_dir: str | Path, *, min_free_bytes: int = 0) -> int:
        """Extract validated members into ``destination_dir``; return file count.

        Each target path is resolved and verified to stay within the destination
        root before extraction, preventing path-escape writes.
        """
        destination_root = Path(destination_dir)
        if min_free_bytes > 0:
            check_disk_headroom(destination_root, self.estimated_uncompressed_size(), min_free_bytes)

        members = self._get_validated_members()
        extracted_count = 0
        resolved_destination = destination_root.resolve()

        try:
            with ZipFile(self.path) as archive:
                for member in members:
                    member_name = _normalize_zip_member_name(member.filename)
                    target_path = (resolved_destination / member_name).resolve()
                    if resolved_destination not in target_path.parents and target_path != resolved_destination:
                        msg = f"Archive contains unsafe entry path '{member.filename}'"
                        raise ZipBombDetectedError(msg)

                    archive.extract(member, resolved_destination)
                    if not member.is_dir():
                        extracted_count += 1
        except BadZipFile as error:
            raise InvalidArchiveError(_NOT_A_ZIP_MSG) from error

        return extracted_count

    def _get_validated_members(self) -> list[ZipInfo]:
        if self._validated_members is not None:
            return self._validated_members

        try:
            with ZipFile(self.path) as archive:
                members = archive.infolist()
        except BadZipFile as error:
            raise InvalidArchiveError(_NOT_A_ZIP_MSG) from error

        validate_zip_entries(
            members,
            max_file_count=self.max_file_count,
            max_uncompressed_bytes=self.max_uncompressed_bytes,
        )
        self._validated_members = members
        return members


def _is_ignorable_extraction_entry(entry: Path) -> bool:
    """Return True for OS-generated junk that should not count as dataset content."""
    name = entry.name
    return name in {"__MACOSX", ".DS_Store"} or name.startswith("._")


def flatten_single_root_directory(destination_dir: str | Path) -> None:
    """Flatten an extraction that nests everything under one top-level directory.

    OS-generated junk entries (e.g. macOS ``__MACOSX`` and ``.DS_Store``) are
    removed and ignored when determining whether a single root directory is
    present.
    """
    root = Path(destination_dir)

    meaningful_entries: list[Path] = []
    for entry in root.iterdir():
        if _is_ignorable_extraction_entry(entry):
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
            continue
        meaningful_entries.append(entry)

    if len(meaningful_entries) != 1 or not meaningful_entries[0].is_dir():
        return

    nested_root = meaningful_entries[0]
    for child in list(nested_root.iterdir()):
        shutil.move(str(child), str(root / child.name))
    nested_root.rmdir()
