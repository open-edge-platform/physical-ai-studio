# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trainer HTTP API."""

from __future__ import annotations

import asyncio
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from uuid import UUID  # noqa: TC003

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from loguru import logger
from physicalai.data.archive_safety import (
    InsufficientDiskSpaceError,
    InvalidArchiveError,
    SafeZipArchive,
    ZipBombDetectedError,
    check_disk_headroom,
    flatten_single_root_directory,
)
from sse_starlette.sse import EventSourceResponse

from trainer.schemas import (
    CancelResponse,
    DatasetTransfer,
    JobState,
    SubmitJobRequest,
    SubmitJobResponse,
    TrainerJobStatus,
)
from trainer.settings import get_settings

if TYPE_CHECKING:
    from trainer.queue_worker import QueueManager

router = APIRouter(prefix="/jobs")

_TERMINAL = {TrainerJobStatus.COMPLETED, TrainerJobStatus.FAILED, TrainerJobStatus.CANCELED}
_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")

_ARTIFACT_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB


class _ChunkedFileResponse(FileResponse):
    """FileResponse with a larger streaming chunk size for large artifacts."""

    chunk_size = _ARTIFACT_CHUNK_SIZE


def _manager(request: Request) -> QueueManager:
    return request.app.state.queue_manager


def _get_job_id(job_id: UUID) -> str:
    """Validate a route job ID and return its canonical storage representation."""
    return str(job_id)


JobId = Annotated[str, Depends(_get_job_id)]


@dataclass(frozen=True)
class _ResolvedJob:
    """A validated trainer job and the manager that owns it."""

    id: str
    manager: QueueManager
    state: JobState


def _get_job(request: Request, job_id: JobId) -> _ResolvedJob:
    """Load a job or return the standard not-found response."""
    manager = _manager(request)
    state = manager.store.get(job_id)
    if state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return _ResolvedJob(id=job_id, manager=manager, state=state)


ResolvedJob = Annotated[_ResolvedJob, Depends(_get_job)]


def _get_http_dataset_job(job: ResolvedJob) -> _ResolvedJob:
    """Return an awaiting HTTP-transfer job or raise the appropriate HTTP error."""
    if job.state.status != TrainerJobStatus.AWAITING_DATASET:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job is not awaiting a dataset upload")

    submitted = job.manager.store.get_request(job.id)
    if submitted is None or submitted.dataset_transfer != DatasetTransfer.HTTP:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job does not use http dataset transfer")
    return job


HttpDatasetJob = Annotated[_ResolvedJob, Depends(_get_http_dataset_job)]


def _dataset_dir(job_id: str) -> Path:
    """Return the extraction directory for a job's uploaded dataset."""
    return get_settings().datasets_dir / job_id


def _upload_path(job_id: str) -> Path:
    """Return the staged partial ZIP path for an HTTP dataset upload."""
    return get_settings().datasets_dir / f"{job_id}.zip.part"


async def _stream_body_to_disk(request: Request, destination: Path, *, append: bool) -> int:
    """Stream the raw request body to ``destination`` and return bytes written."""
    written = 0
    with destination.open("ab" if append else "wb") as out:
        async for chunk in request.stream():
            out.write(chunk)
            written += len(chunk)
    return written


def _upload_offset(job_id: str) -> int:
    """Return the number of staged upload bytes for a job."""
    try:
        return _upload_path(job_id).stat().st_size
    except FileNotFoundError:
        return 0


def _parse_upload_range(request: Request, job_id: str) -> tuple[int, int | None, int | None, bool]:
    """Validate a resumable upload range and return its write parameters."""
    content_range = request.headers.get("content-range")
    if not content_range:
        return 0, None, None, False
    match = _CONTENT_RANGE_RE.fullmatch(content_range)
    if match is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Content-Range header")
    start, end, total = (int(value) for value in match.groups())
    if total <= 0 or start > end or end >= total:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Content-Range bounds")
    offset = _upload_offset(job_id)
    if start != offset:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Upload offset does not match staged dataset",
            headers={"Upload-Offset": str(offset)},
        )
    return start, end, total, start > 0


def _validate_range_body_size(written: int, start: int, end: int | None, total: int | None) -> None:
    """Ensure the body exactly fills the declared byte range."""
    if end is None or total is None or written != end - start + 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Content-Range length does not match body")


def _remaining_upload_bytes(request: Request, start: int, total: int | None) -> int | None:
    """Return the declared bytes still needed to stage an upload, when known."""
    if total is not None:
        return total - start
    content_length = request.headers.get("content-length")
    if content_length is not None and content_length.isdecimal():
        return int(content_length)
    return None


def _check_upload_disk_headroom(request: Request, start: int, total: int | None) -> None:
    """Reserve space for a declared incoming ZIP chunk, when its size is known."""
    remaining_bytes = _remaining_upload_bytes(request, start, total)
    if remaining_bytes is not None:
        settings = get_settings()
        check_disk_headroom(settings.datasets_dir, remaining_bytes, settings.min_free_bytes)


def _validate_and_extract(archive_path: Path, target_dir: Path) -> None:
    """Validate the ZIP and extract it into ``target_dir`` (blocking)."""
    settings = get_settings()
    safe = SafeZipArchive(archive_path, max_uncompressed_bytes=settings.max_uncompressed_bytes)
    safe.validate()
    target_dir.mkdir(parents=True, exist_ok=True)
    safe.extract_to(target_dir, min_free_bytes=settings.min_free_bytes)
    # Tolerate a single wrapping directory in uploaded snapshots.
    flatten_single_root_directory(target_dir)


@router.post("", response_model=SubmitJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_job(body: SubmitJobRequest, request: Request) -> SubmitJobResponse:
    """Enqueue a job, awaiting upload for HTTP transfers."""
    manager = _manager(request)
    job_id = manager.store.create(body)
    state = manager.store.get(job_id)
    job_status = state.status if state is not None else TrainerJobStatus.QUEUED
    return SubmitJobResponse(remote_job_id=job_id, status=job_status)


@router.put("/{job_id}/dataset", response_model=JobState, status_code=status.HTTP_202_ACCEPTED)
async def upload_dataset(job_id: HttpDatasetJob, request: Request, response: Response) -> JobState:
    """Validate an awaiting HTTP job's ZIP upload and queue it."""
    manager = job_id.manager
    state = job_id.state

    content_type = request.headers.get("content-type", "")
    if "zip" not in content_type.lower():
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Dataset must be uploaded as application/zip",
        )

    settings = get_settings()
    target_dir = _dataset_dir(job_id.id)
    archive_path = _upload_path(job_id.id)
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)

    start, end, total, append = _parse_upload_range(request, job_id.id)

    completed_upload = False
    try:
        _check_upload_disk_headroom(request, start, total)
        written = await _stream_body_to_disk(request, archive_path, append=append)
        if total is not None:
            if end is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Content-Range header")
            _validate_range_body_size(written, start, end, total)
            next_offset = end + 1
            response.headers["Upload-Offset"] = str(next_offset)
            if next_offset < total:
                return state
        await asyncio.to_thread(_validate_and_extract, archive_path, target_dir)
        completed_upload = True
    except (ZipBombDetectedError, InvalidArchiveError) as exc:
        _cleanup_upload(archive_path, target_dir)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except InsufficientDiskSpaceError as exc:
        _cleanup_upload(archive_path, target_dir)
        raise HTTPException(status_code=status.HTTP_507_INSUFFICIENT_STORAGE, detail=str(exc)) from exc
    finally:
        if completed_upload:
            archive_path.unlink(missing_ok=True)

    manager.store.mark_dataset_ready(job_id.id)
    updated = manager.store.get(job_id.id)
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if updated.status != TrainerJobStatus.QUEUED:
        # Job was canceled/failed while the upload was in-flight; don't leak disk.
        _cleanup_upload(archive_path, target_dir)
    return updated


def _cleanup_upload(archive_path: Path, target_dir: Path) -> None:
    """Best-effort removal of a failed upload's staged ZIP and extraction."""
    archive_path.unlink(missing_ok=True)
    if target_dir.exists():
        try:
            shutil.rmtree(target_dir)
        except OSError as exc:
            logger.warning("Failed to clean up dataset dir {}: {}", target_dir, exc)


@router.head("/{job_id}/dataset", status_code=status.HTTP_204_NO_CONTENT)
async def get_dataset_upload_offset(job_id: HttpDatasetJob) -> Response:
    """Return the staged byte offset for an interrupted HTTP dataset upload."""
    return Response(status_code=status.HTTP_204_NO_CONTENT, headers={"Upload-Offset": str(_upload_offset(job_id.id))})


@router.get("/{job_id}", response_model=JobState)
async def get_job(job_id: ResolvedJob) -> JobState:
    """Return the current job state."""
    return job_id.state


@router.get("/{job_id}/events")
async def job_events(job_id: ResolvedJob, request: Request) -> EventSourceResponse:
    """Stream job state changes until the job reaches a terminal state."""
    manager = job_id.manager

    async def _event_stream():  # noqa: ANN202
        last: str | None = None
        while True:
            if await request.is_disconnected():
                break
            state = manager.store.get(job_id.id)
            if state is None:
                break
            payload = state.model_dump_json()
            if payload != last:
                yield {"event": "state", "data": payload}
                last = payload
            if state.status in _TERMINAL:
                break
            await asyncio.sleep(1.0)

    return EventSourceResponse(_event_stream())


@router.get("/{job_id}/artifact")
async def get_artifact(job_id: ResolvedJob) -> FileResponse:
    """Download the trained model archive."""
    if job_id.state.status != TrainerJobStatus.COMPLETED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Artifact not ready")

    artifact = job_id.manager.store.get_artifact(job_id.id)
    if artifact is None or not Path(artifact).is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact missing")
    return _ChunkedFileResponse(artifact, media_type="application/zip", filename=f"{job_id.id}.zip")


@router.post("/{job_id}/cancel", response_model=CancelResponse)
async def cancel_job(job_id: ResolvedJob) -> CancelResponse:
    """Request cancellation of a queued or running job."""
    if job_id.state.status not in _TERMINAL:
        job_id.manager.request_cancel(job_id.id)
    final = job_id.manager.store.get(job_id.id)
    if final is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return CancelResponse(remote_job_id=final.remote_job_id, status=final.status)
