# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Staging-path helpers for dataset import archives.

An *opaque staging identifier* (a UUID string) is persisted in the job
payload instead of the raw filesystem path.  The actual path is derived
deterministically from the id at runtime so no in-memory map is needed.

Layout::

    <cache_dir>/imports/datasets/<staging_id>.zip
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from settings import get_settings

if TYPE_CHECKING:
    from pathlib import Path

    from schemas.dataset_import_job import DatasetImportJobPayload


def generate_staging_id() -> UUID:
    """Return a new random staging identifier (UUID v4)."""
    return uuid4()


def resolve_payload_archive_path(payload: DatasetImportJobPayload) -> Path:
    """Return the archive ``Path`` for a dataset import job payload."""

    settings = get_settings()
    staging_dir = settings.cache_dir / "imports" / "datasets"
    return staging_dir / f"{payload.archive_staging_id}.zip"
