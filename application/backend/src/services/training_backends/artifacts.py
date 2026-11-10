# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Validate and extract model artifacts downloaded from training providers."""

from pathlib import Path

from physicalai.data.archive_safety import SafeZipArchive


def extract_training_artifact(
    archive: Path, output_dir: Path, max_uncompressed_bytes: int, min_free_bytes: int
) -> None:
    """Extract an artifact only after validating its paths and expansion limits."""
    safe = SafeZipArchive(archive, max_uncompressed_bytes=max_uncompressed_bytes)
    safe.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe.extract_to(output_dir, min_free_bytes=min_free_bytes)
