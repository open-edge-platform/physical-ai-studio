# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trainer HTTP API."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from trainer.archive_safety import (
    InsufficientDiskSpaceError,
    InvalidArchiveError,
    SafeZipArchive,
    ZipBombDetectedError,
    check_disk_headroom,
    flatten_single_root_directory,
)
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


def _manager(request: Request) -> QueueManager:
    return request.app.state.queue_manager


def _dataset_dir(job_id: str) -> Path:
    """Return the extraction directory for a job's uploaded dataset."""
    return get_settings().datasets_dir / job_id


async def _stream_body_to_disk(request: Request, destination: Path) -> None:
    """Stream the raw request body to ``destination`` in chunks."""
    with destination.open("wb") as out:
        async for chunk in request.stream():
            out.write(chunk)


def _validate_and_extract(archive_path: Path, target_dir: Path) -> None:
    """Validate the ZIP and extract it into ``target_dir`` (blocking)."""
    settings = get_settings()
    safe = SafeZipArchive(archive_path, max_uncompressed_bytes=settings.max_uncompressed_bytes)
    safe.validate()
    target_dir.mkdir(parents=True, exist_ok=True)
    safe.extract_to(target_dir, min_free_bytes=settings.min_free_bytes)
    # Studio zips the snapshot at its root, but tolerate a single wrapping folder.
    flatten_single_root_directory(target_dir)


@router.post("", response_model=SubmitJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_job(body: SubmitJobRequest, request: Request) -> SubmitJobResponse:
    """Enqueue a training job.

    An hf-transfer job is queued immediately; an http-transfer job waits in
    ``awaiting_dataset`` until its dataset is uploaded via ``PUT /jobs/{id}/dataset``.
    """
    manager = _manager(request)
    job_id = manager.store.create(body)
    state = manager.store.get(job_id)
    job_status = state.status if state is not None else TrainerJobStatus.QUEUED
    return SubmitJobResponse(remote_job_id=job_id, status=job_status)


@router.put("/{job_id}/dataset", response_model=JobState, status_code=status.HTTP_202_ACCEPTED)
async def upload_dataset(job_id: str, request: Request) -> JobState:
    """Accept the dataset ZIP for an awaiting-dataset job and queue it.

    The body is the raw ZIP (``Content-Type: application/zip``). The job must
    have been submitted with http transfer and still be in the
    ``awaiting_dataset`` state. After validation and extraction the job
    transitions to ``queued`` for the worker to pick up.
    """
    manager = _manager(request)
    state = manager.store.get(job_id)
    if state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if state.status != TrainerJobStatus.AWAITING_DATASET:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job is not awaiting a dataset upload")

    submitted = manager.store.get_request(job_id)
    if submitted is None or submitted.dataset_transfer != DatasetTransfer.HTTP:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job does not use http dataset transfer")

    content_type = request.headers.get("content-type", "")
    if "zip" not in content_type.lower():
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Dataset must be uploaded as application/zip",
        )

    settings = get_settings()
    target_dir = _dataset_dir(job_id)
    archive_path = settings.datasets_dir / f"{job_id}.zip"
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)

    try:
        check_disk_headroom(settings.datasets_dir, settings.max_uncompressed_bytes, settings.min_free_bytes)
        await _stream_body_to_disk(request, archive_path)
        await asyncio.to_thread(_validate_and_extract, archive_path, target_dir)
    except (ZipBombDetectedError, InvalidArchiveError) as exc:
        _cleanup_upload(archive_path, target_dir)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except InsufficientDiskSpaceError as exc:
        _cleanup_upload(archive_path, target_dir)
        raise HTTPException(status_code=status.HTTP_507_INSUFFICIENT_STORAGE, detail=str(exc)) from exc
    finally:
        archive_path.unlink(missing_ok=True)

    manager.store.mark_dataset_ready(job_id)
    updated = manager.store.get(job_id)
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return updated


def _cleanup_upload(archive_path: Path, target_dir: Path) -> None:
    """Best-effort removal of a failed upload's staged ZIP and extraction."""
    archive_path.unlink(missing_ok=True)
    if target_dir.exists():
        try:
            shutil.rmtree(target_dir)
        except OSError as exc:
            logger.warning("Failed to clean up dataset dir {}: {}", target_dir, exc)


@router.get("/{job_id}", response_model=JobState)
async def get_job(job_id: str, request: Request) -> JobState:
    """Return the current job state."""
    state = _manager(request).store.get(job_id)
    if state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return state


@router.get("/{job_id}/events")
async def job_events(job_id: str, request: Request) -> EventSourceResponse:
    """Stream job state changes until the job reaches a terminal state."""
    manager = _manager(request)
    if manager.store.get(job_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    async def _event_stream():  # noqa: ANN202
        last: str | None = None
        while True:
            if await request.is_disconnected():
                break
            state = manager.store.get(job_id)
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
async def get_artifact(job_id: str, request: Request) -> FileResponse:
    """Download the trained model archive."""
    manager = _manager(request)
    state = manager.store.get(job_id)
    if state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if state.status != TrainerJobStatus.COMPLETED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Artifact not ready")

    artifact = manager.store.get_artifact(job_id)
    if artifact is None or not Path(artifact).is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact missing")
    return FileResponse(artifact, media_type="application/zip", filename=f"{job_id}.zip")


@router.post("/{job_id}/cancel", response_model=CancelResponse)
async def cancel_job(job_id: str, request: Request) -> CancelResponse:
    """Request cancellation of a queued or running job."""
    manager = _manager(request)
    state = manager.store.get(job_id)
    if state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if state.status not in _TERMINAL:
        manager.request_cancel(job_id)
    final = manager.store.get(job_id)
    if final is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return CancelResponse(remote_job_id=job_id, status=final.status)
