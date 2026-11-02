# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset archive validation shared by the HTTP and Batch entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.data.archive_safety import SafeZipArchive, flatten_single_root_directory

from trainer.settings import get_settings

if TYPE_CHECKING:
    from pathlib import Path


def validate_and_extract(archive_path: Path, target_dir: Path) -> None:
    """Validate the ZIP and extract it into ``target_dir`` (blocking)."""
    settings = get_settings()
    safe = SafeZipArchive(archive_path, max_uncompressed_bytes=settings.max_uncompressed_bytes)
    safe.validate()
    target_dir.mkdir(parents=True, exist_ok=True)
    safe.extract_to(target_dir, min_free_bytes=settings.min_free_bytes)
    # Tolerate a single wrapping directory in uploaded snapshots.
    flatten_single_root_directory(target_dir)
