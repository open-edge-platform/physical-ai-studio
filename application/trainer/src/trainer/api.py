# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trainer HTTP API."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from trainer.schemas import (
    CancelResponse,
    JobState,
    SubmitJobRequest,
    SubmitJobResponse,
    TrainerJobStatus,
)

if TYPE_CHECKING:
    from trainer.queue_worker import QueueManager

router = APIRouter(prefix="/jobs")

_TERMINAL = {TrainerJobStatus.COMPLETED, TrainerJobStatus.FAILED, TrainerJobStatus.CANCELED}


def _manager(request: Request) -> QueueManager:
    return request.app.state.queue_manager


@router.post("", response_model=SubmitJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_job(body: SubmitJobRequest, request: Request) -> SubmitJobResponse:
    """Enqueue a training job."""
    manager = _manager(request)
    job_id = manager.store.create(body)
    return SubmitJobResponse(remote_job_id=job_id, status=TrainerJobStatus.QUEUED)


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
