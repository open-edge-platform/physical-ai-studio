from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from schemas import Dataset
from schemas.dataset_import_job import (
    DatasetImportJobPayload,
    DatasetImportSource,
    DatasetManifest,
    ImportValidationReport,
)

if TYPE_CHECKING:
    from uuid import UUID

    from services.archive_safety import SafeZipArchive


class DatasetImportAdapter(ABC):
    source: DatasetImportSource = DatasetImportSource.UNKNOWN

    @abstractmethod
    def detect(self, archive: SafeZipArchive) -> tuple[bool, ImportValidationReport]:
        """Return (matched, report) where matched is True if this adapter can process the archive.

        The report may contain error messages explaining why the archive did not match,
        which callers may surface when the adapter was explicitly requested.
        """

    @abstractmethod
    def build_draft(
        self,
        archive: SafeZipArchive,
        payload: DatasetImportJobPayload,
    ) -> tuple[DatasetManifest, ImportValidationReport]:
        """Parse source archive and return draft manifest plus validation report."""

    @abstractmethod
    def validate_pre_commit(self, payload: DatasetImportJobPayload) -> ImportValidationReport:
        """Validate the finalized payload immediately before committing.

        Implementations must enforce adapter-specific requirements (for example,
        source manifest consistency and finalize input fields). Generic shared
        checks are handled by the import workflow before adapter validation.
        """

    @abstractmethod
    async def commit(self, payload: DatasetImportJobPayload, project_id: UUID, archive: SafeZipArchive) -> Dataset:
        """Execute extraction and register dataset in DB."""
