import asyncio
import os
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from api.dependencies import get_dataset_import_service, get_job_service, get_project_id
from exceptions import InvalidArchiveError, ZipBombDetectedError
from schemas import Job
from schemas.base_job import JobType
from schemas.dataset_import_job import DatasetImportFinalizeInput, DatasetImportJobPayload, ImportStep
from schemas.job import DatasetImportJob
from services.archive_safety import SafeZipArchive, check_disk_headroom, cleanup_staged_archive
from services.dataset_import.adapters import get_supported_dataset_import_formats
from services.dataset_import.service import DatasetImportService
from services.dataset_import.staging import resolve_payload_archive_path
from services.job_service import JobService
from settings import get_settings

router = APIRouter(prefix="/api/projects/{project_id}/imports", tags=["Imports"])

ProjectID = Annotated[UUID, Depends(get_project_id)]

_UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def _write_archive_to_disk(file: UploadFile, destination: Path) -> None:
    """Synchronous helper: write upload to *destination* in chunks."""
    with destination.open("wb") as out:
        while chunk := file.file.read(_UPLOAD_CHUNK_SIZE):
            out.write(chunk)


async def _persist_uploaded_archive(file: UploadFile, payload: DatasetImportJobPayload) -> Path:
    """Persist an uploaded archive under the given staging id without blocking the event loop."""
    destination = resolve_payload_archive_path(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(_write_archive_to_disk, file, destination)
    return destination


def _resolve_upload_size_estimate(archive: UploadFile, content_length_header: str | None, fallback: int) -> int:
    """Best-effort upload-size estimate used for disk-headroom checks.

    Prefer the actual staged file size from the uploaded file object. If that
    cannot be determined, fall back to the request Content-Length header.
    """
    actual_size: int | None = None
    try:
        current_pos = archive.file.tell()
        archive.file.seek(0, os.SEEK_END)
        actual_size = archive.file.tell()
        archive.file.seek(current_pos)
    except (AttributeError, OSError, ValueError):
        actual_size = None

    if actual_size is not None and actual_size >= 0:
        return actual_size

    if content_length_header is not None:
        content_length: int | None = None
        try:
            content_length = int(content_length_header)
        except ValueError:
            content_length = None

        if content_length is not None and content_length >= 0:
            return content_length

    return fallback


async def get_awaiting_upload_job(
    project_id: ProjectID,
    job_id: UUID,
    job_service: Annotated[JobService, Depends(get_job_service)],
) -> DatasetImportJob:
    """Dependency: fetch and validate a dataset import job that is awaiting archive upload."""
    job = await job_service.get_job_by_id(job_id)

    if job.project_id != project_id or job.type != JobType.DATASET_IMPORT:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset import job not found")

    if not isinstance(job.payload, DatasetImportJobPayload):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Dataset import job payload is invalid")

    if job.payload.step != ImportStep.AWAITING_ARCHIVE_UPLOAD:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Archive can only be uploaded when job is in '{ImportStep.AWAITING_ARCHIVE_UPLOAD}' step",
        )

    return job


@router.post("/datasets:prepare", status_code=status.HTTP_202_ACCEPTED)
async def prepare_dataset_import_job(
    project_id: ProjectID,
    dataset_import_service: Annotated[DatasetImportService, Depends(get_dataset_import_service)],
    format_hint: Annotated[str, Form()] = "auto",
    dataset_name: Annotated[str, Form()] = "",
) -> Job:
    """Phase 1: Create a dataset import job immediately, before the archive is uploaded.

    Returns a Job with step=awaiting_archive_upload. Use the returned job_id in the phase-2
    upload endpoint to attach the archive and queue processing.
    """
    supported_format_hints = get_supported_dataset_import_formats()
    if format_hint not in supported_format_hints:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid format_hint '{format_hint}'. Expected one of: {sorted(supported_format_hints)}",
        )

    if not dataset_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="dataset_name must be a non-empty string",
        )

    return await dataset_import_service.prepare_dataset_import_job(
        project_id=project_id, format_hint=format_hint, dataset_name=dataset_name.strip()
    )


@router.put("/datasets/{job_id}:upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_dataset_import_archive(
    request: Request,
    project_id: ProjectID,
    job_id: UUID,
    archive: Annotated[UploadFile, File(description="Dataset archive ZIP")],
    job: Annotated[DatasetImportJob, Depends(get_awaiting_upload_job)],
    dataset_import_service: Annotated[DatasetImportService, Depends(get_dataset_import_service)],
) -> Job:
    """Phase 2: Upload the archive and attach it to an existing import job.

    The job must be in the awaiting_archive_upload step (created via POST /datasets:prepare).
    After this call the job transitions to step=queued_for_detection and status=pending so the
    worker can pick it up for processing.
    """
    if not archive.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing archive filename")

    if not archive.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only ZIP archives are supported"
        )

    # Check if there is enough space for dataset import
    settings = get_settings()
    cache_dir = settings.cache_dir / "imports" / "datasets"
    upload_size_estimate = _resolve_upload_size_estimate(
        archive, request.headers.get("content-length"), settings.data_import_max_upload_bytes
    )

    check_disk_headroom(cache_dir, upload_size_estimate, settings.data_import_min_free_bytes)

    # Make sure we are safe against zip bomb attacks
    uploaded_archive_path = await _persist_uploaded_archive(archive, job.payload)
    try:
        safe_archive = SafeZipArchive(
            uploaded_archive_path,
            max_uncompressed_bytes=settings.data_import_max_uncompressed_bytes,
        )
        await asyncio.to_thread(safe_archive.validate)
    except (ZipBombDetectedError, InvalidArchiveError):
        cleanup_staged_archive(uploaded_archive_path)
        raise

    return await dataset_import_service.attach_dataset_import_archive(
        project_id=project_id,
        job_id=job_id,
        uploaded_archive_name=archive.filename,
    )


@router.post("/datasets/{job_id}:finalize", status_code=status.HTTP_202_ACCEPTED)
async def finalize_dataset_import_job(
    project_id: ProjectID,
    job_id: UUID,
    payload: DatasetImportFinalizeInput,
    dataset_import_service: Annotated[DatasetImportService, Depends(get_dataset_import_service)],
) -> Job:
    """Finalize staged dataset import by providing required user inputs."""
    return await dataset_import_service.finalize_dataset_import_job(
        project_id=project_id,
        job_id=job_id,
        finalize_input=payload,
    )


@router.post("/datasets/{job_id}:cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_dataset_import_job(
    project_id: ProjectID,
    job_id: UUID,
    dataset_import_service: Annotated[DatasetImportService, Depends(get_dataset_import_service)],
) -> Job:
    """Cancel dataset import while it is awaiting user review."""
    return await dataset_import_service.cancel_dataset_import_job(project_id=project_id, job_id=job_id)
