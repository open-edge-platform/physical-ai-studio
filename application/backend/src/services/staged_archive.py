"""Lifecycle helpers for locally staged archive files."""

from __future__ import annotations

from pathlib import Path

from loguru import logger


def cleanup_staged_archive(archive_path: str | Path | None) -> None:
    """Remove a staged archive without masking the caller's primary failure."""
    if archive_path is None:
        return

    try:
        Path(archive_path).unlink(missing_ok=True)
    except OSError as error:
        logger.warning("Failed to clean staged archive '{}': {}", archive_path, error)
