from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from schemas.dataset_import_job import ImportValidationReport

from .base import DatasetImportAdapter
from .lerobot_v3 import LeRobotV3Adapter

if TYPE_CHECKING:
    from physicalai.data.archive_safety import SafeZipArchive

REGISTERED_DATASET_IMPORT_ADAPTERS: tuple[DatasetImportAdapter, ...] = (LeRobotV3Adapter(),)


def get_registered_dataset_import_adapters() -> list[DatasetImportAdapter]:
    """Return fresh adapter list for worker/runtime usage."""
    return list(REGISTERED_DATASET_IMPORT_ADAPTERS)


def get_supported_dataset_import_formats() -> set[str]:
    """Single source of truth for currently supported import format hints."""
    return {"auto", *[adapter.source.value for adapter in REGISTERED_DATASET_IMPORT_ADAPTERS]}


@dataclass
class DatasetAdapterSelectionResult:
    """Structured result from :func:`select_dataset_import_adapter`.

    Exactly one of *adapter* or *report* will be set:
    - Success: ``adapter`` is the matched :class:`DatasetImportAdapter`, ``report`` is ``None``.
    - Failure: ``adapter`` is ``None``, ``report`` carries the actionable error(s).
    """

    adapter: DatasetImportAdapter | None
    report: ImportValidationReport | None


def select_dataset_import_adapter(
    adapters: list[DatasetImportAdapter],
    format_hint: str,
    archive: SafeZipArchive,
) -> DatasetAdapterSelectionResult:
    """Select the appropriate adapter given a *format_hint* and an open *archive*.

    Returns a :class:`DatasetAdapterSelectionResult` — never raises for semantic
    mismatches or unknown hints.
    """
    for adapter in adapters:
        if format_hint not in ("auto", adapter.source.value):
            continue

        matched, report = adapter.detect(archive)
        if matched:
            return DatasetAdapterSelectionResult(adapter=adapter, report=None)

        if format_hint == adapter.source.value:
            return DatasetAdapterSelectionResult(adapter=None, report=report)

    report = ImportValidationReport()
    report.add_error(
        "No supported dataset format was detected. "
        "Supported formats are LeRobot v2 and LeRobot v3. "
        "Ensure the archive contains the required metadata (meta/info.json) "
        "and the expected data layout (data/ directory with Parquet episode files)."
    )
    return DatasetAdapterSelectionResult(adapter=None, report=report)


__all__ = [
    "DatasetAdapterSelectionResult",
    "DatasetImportAdapter",
    "LeRobotV3Adapter",
    "get_registered_dataset_import_adapters",
    "get_supported_dataset_import_formats",
    "select_dataset_import_adapter",
]
