# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Security-aware ZIP archive validation and extraction utilities.

The module rejects unsafe ZIP members before reading or extracting them. It is
framework-independent so applications can supply their own exception types.
"""

from __future__ import annotations

import json
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import BadZipFile, ZipFile, ZipInfo

if TYPE_CHECKING:
    from collections.abc import Iterator

DEFAULT_MAX_FILE_COUNT = 200_000
_NOT_A_ZIP_MSG = "Uploaded file is not a valid ZIP archive"


class InvalidArchiveError(Exception):
    """Raised when an archive cannot be read as a ZIP file."""


class ZipBombDetectedError(Exception):
    """Raised when an archive exceeds limits or contains an unsafe member."""


class InsufficientDiskSpaceError(Exception):
    """Raised when a target filesystem lacks extraction headroom."""


@dataclass(frozen=True)
class ArchiveErrorTypes:
    """Exception types raised by archive operations.

    Applications with transport-specific exception types can provide a custom
    bundle without coupling this module to their framework.
    """

    invalid_archive: type[Exception] = InvalidArchiveError
    zip_bomb_detected: type[Exception] = ZipBombDetectedError
    insufficient_disk_space: type[Exception] = InsufficientDiskSpaceError


DEFAULT_ARCHIVE_ERROR_TYPES = ArchiveErrorTypes()


def _normalize_zip_member_name(name: str) -> str:
    return name.replace("\\", "/").strip("/").removeprefix("./")


def _is_symlink(member_external_attr: int) -> bool:
    """Return whether ZIP mode bits mark an entry as a symbolic link."""
    mode = member_external_attr >> 16
    return (mode & stat.S_IFLNK) == stat.S_IFLNK


def _collect_total_uncompressed(members: list[ZipInfo]) -> int:
    return sum(member.file_size for member in members)


def validate_zip_entries(
    members: list[ZipInfo],
    *,
    max_file_count: int | None,
    max_uncompressed_bytes: int,
    error_types: ArchiveErrorTypes = DEFAULT_ARCHIVE_ERROR_TYPES,
) -> int:
    """Validate ZIP members and return their total uncompressed size.

    Args:
        members: ZIP entries to inspect.
        max_file_count: Maximum number of entries. Uses the default when None.
        max_uncompressed_bytes: Maximum aggregate uncompressed size.
        error_types: Exception types to raise for invalid input.

    Returns:
        Aggregate uncompressed size in bytes.
    """
    if max_file_count is None:
        max_file_count = DEFAULT_MAX_FILE_COUNT

    if len(members) > max_file_count:
        msg = f"Archive contains too many entries ({len(members)} > {max_file_count})"
        raise error_types.zip_bomb_detected(msg)

    total_uncompressed = _collect_total_uncompressed(members)
    if total_uncompressed > max_uncompressed_bytes:
        msg = f"Archive uncompressed size exceeds allowed limit ({total_uncompressed} > {max_uncompressed_bytes} bytes)"
        raise error_types.zip_bomb_detected(msg)

    for member in members:
        name = _normalize_zip_member_name(member.filename)
        if _is_symlink(member.external_attr):
            msg = f"Archive contains symlink entry '{name}', which is not allowed"
            raise error_types.zip_bomb_detected(msg)

        normalized_path = Path(name)
        if normalized_path.is_absolute() or ".." in normalized_path.parts:
            msg = f"Archive contains unsafe entry path '{member.filename}'"
            raise error_types.zip_bomb_detected(msg)

        if normalized_path.suffix.lower() == ".zip":
            msg = f"Archive contains nested zip entry '{member.filename}', which is not allowed"
            raise error_types.zip_bomb_detected(msg)

    return total_uncompressed


def check_disk_headroom(
    directory: Path,
    required_bytes: int,
    min_free_bytes: int,
    *,
    error_types: ArchiveErrorTypes = DEFAULT_ARCHIVE_ERROR_TYPES,
) -> None:
    """Ensure a filesystem retains free space after a pending write.

    Args:
        directory: Target directory or path on the target filesystem.
        required_bytes: Bytes expected to be written.
        min_free_bytes: Minimum free bytes that must remain after the write.
        error_types: Exception types to raise when the check fails.

    """
    directory.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(directory)
    needed = required_bytes + min_free_bytes
    if usage.free < needed:
        msg = (
            f"Insufficient disk space on '{directory}': {usage.free} bytes free, need {needed} bytes "
            f"({required_bytes} for data + {min_free_bytes} headroom)"
        )
        raise error_types.insufficient_disk_space(msg)


class SafeZipArchive:
    """Validate a ZIP archive once, then provide safe reads and extraction."""

    def __init__(
        self,
        archive_path: str | Path,
        *,
        max_uncompressed_bytes: int,
        max_file_count: int | None = None,
        error_types: ArchiveErrorTypes = DEFAULT_ARCHIVE_ERROR_TYPES,
    ) -> None:
        """Store archive safety limits and application-specific exception types."""
        self.path = Path(archive_path)
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.max_file_count = max_file_count
        self.error_types = error_types
        self._validated_members: list[ZipInfo] | None = None

    def validate(self) -> None:
        """Run the one-time safety pass over archive entries."""
        self._get_validated_members()

    def estimated_uncompressed_size(self) -> int:
        """Return the aggregate uncompressed size of validated archive members."""
        return _collect_total_uncompressed(self._get_validated_members())

    def iter_normalized_names(self) -> Iterator[str]:
        """Iterate validated member names normalized to forward-slash paths.

        Yields:
            A normalized archive member name.
        """
        for member in self._get_validated_members():
            yield _normalize_zip_member_name(member.filename)

    def resolve_member_name(self, target_name: str) -> str | None:
        """Resolve a logical member path, including a single enclosing directory.

        Returns:
            The original ZIP member name, or None when no member matches.
        """
        normalized_target = _normalize_zip_member_name(target_name)
        for member in self._get_validated_members():
            normalized = _normalize_zip_member_name(member.filename)
            if normalized == normalized_target or normalized.endswith(f"/{normalized_target}"):
                return member.filename
        return None

    def read_json(self, target_name: str) -> dict | None:
        """Read a JSON object from a validated member, or None when absent.

        Returns:
            The decoded JSON value or None when the member is absent.
        """
        raw_bytes = self._read_member(target_name)
        if raw_bytes is None:
            return None

        try:
            return json.loads(raw_bytes.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            msg = f"Unable to read JSON '{target_name}' from archive"
            raise self.error_types.invalid_archive(msg) from error

    def read_jsonl(self, target_name: str) -> Iterator[dict]:
        """Iterate JSON objects from a validated JSONL member.

        Yields:
            Each decoded JSONL value.
        """
        raw_bytes = self._read_member(target_name)
        if raw_bytes is None:
            return

        try:
            for line in raw_bytes.decode().splitlines():
                if line := line.strip():
                    yield json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            msg = f"Unable to read JSONL '{target_name}' from archive"
            raise self.error_types.invalid_archive(msg) from error

    def read_bytes(self, target_name: str) -> bytes | None:
        """Read raw bytes from a validated member, or None when absent.

        Returns:
            Member bytes or None when the member is absent.
        """
        return self._read_member(target_name)

    def extract_to(self, destination_dir: str | Path, *, min_free_bytes: int = 0) -> int:
        """Extract validated entries into a containment-checked destination.

        Args:
            destination_dir: Directory to receive archive contents.
            min_free_bytes: Free space that must remain after extraction.

        Returns:
            Number of non-directory entries extracted.

        """
        destination_root = Path(destination_dir)
        if min_free_bytes > 0:
            check_disk_headroom(
                destination_root,
                self.estimated_uncompressed_size(),
                min_free_bytes,
                error_types=self.error_types,
            )

        members = self._get_validated_members()
        resolved_destination = destination_root.resolve()

        try:
            with ZipFile(self.path) as archive:
                extracted_count = self._extract_members(archive, members, resolved_destination)
        except BadZipFile as error:
            raise self.error_types.invalid_archive(_NOT_A_ZIP_MSG) from error

        return extracted_count

    def _extract_members(self, archive: ZipFile, members: list[ZipInfo], destination: Path) -> int:
        """Extract members after checking each resolved target remains contained.

        Returns:
            Number of non-directory members extracted.
        """
        extracted_count = 0
        for member in members:
            member_name = _normalize_zip_member_name(member.filename)
            target_path = (destination / member_name).resolve()
            if not target_path.is_relative_to(destination):
                msg = f"Archive contains unsafe entry path '{member.filename}'"
                raise self.error_types.zip_bomb_detected(msg)

            archive.extract(member, destination)
            if not member.is_dir():
                extracted_count += 1
        return extracted_count

    def _get_validated_members(self) -> list[ZipInfo]:
        if self._validated_members is not None:
            return self._validated_members

        try:
            with ZipFile(self.path) as archive:
                members = archive.infolist()
        except BadZipFile as error:
            raise self.error_types.invalid_archive(_NOT_A_ZIP_MSG) from error

        validate_zip_entries(
            members,
            max_file_count=self.max_file_count,
            max_uncompressed_bytes=self.max_uncompressed_bytes,
            error_types=self.error_types,
        )
        self._validated_members = members
        return members

    def _read_member(self, target_name: str) -> bytes | None:
        member_name = self.resolve_member_name(target_name)
        if member_name is None:
            return None

        try:
            with ZipFile(self.path) as archive, archive.open(member_name) as file_obj:
                return file_obj.read()
        except (BadZipFile, OSError) as error:
            msg = f"Unable to read '{target_name}' from archive"
            raise self.error_types.invalid_archive(msg) from error


def _is_ignorable_extraction_entry(entry: Path) -> bool:
    """Return whether an extraction entry is operating-system metadata."""
    name = entry.name
    return name in {"__MACOSX", ".DS_Store"} or name.startswith("._")


def flatten_single_root_directory(destination_dir: str | Path) -> None:
    """Flatten a single meaningful top-level directory after extraction.

    macOS archive metadata is removed before determining whether a single root
    directory contains all extracted content.
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


__all__ = [
    "DEFAULT_ARCHIVE_ERROR_TYPES",
    "DEFAULT_MAX_FILE_COUNT",
    "ArchiveErrorTypes",
    "InsufficientDiskSpaceError",
    "InvalidArchiveError",
    "SafeZipArchive",
    "ZipBombDetectedError",
    "check_disk_headroom",
    "flatten_single_root_directory",
    "validate_zip_entries",
]
