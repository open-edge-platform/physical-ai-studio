# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Log streaming API endpoints.

Provides endpoints to discover available log sources and to stream log file
contents in real-time via Server-Sent Events.

Source types:
    - application: The main app log (catch-all for non-worker logs)
    - worker: Per-class worker logs (training, inference, etc.)
    - job: Per-job logs created during training/import/export runs
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette import EventSourceResponse

from api.dependencies import get_log_service
from schemas.logs import LogSource
from services import LogService

router = APIRouter(prefix="/api/logs", tags=["Logs"])


@router.get("/sources")
async def get_log_sources(
    log_service: Annotated[LogService, Depends(get_log_service)],
) -> list[LogSource]:
    """Return all available log sources.

    Sources are grouped by type: application, worker, job.
    """
    return await log_service.get_log_sources()


@router.get("/{source_id}/stream")
async def stream_logs(
    source_id: str,
    log_service: Annotated[LogService, Depends(get_log_service)],
) -> EventSourceResponse:
    """Stream log lines from the given source via Server-Sent Events.

    The connection stays open and new lines are pushed as they are written
    to the log file. Close the connection from the client side to stop.
    """
    path = log_service.resolve_source_path(source_id)

    if path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown log source: {source_id}",
        )

    if not await log_service.source_exists(path):
        # Missing/empty file is treated as an empty stream so the UI can render
        # "No logs available" without entering an error/retry state.
        return EventSourceResponse(log_service.empty_log_stream())

    return EventSourceResponse(log_service.tail_log_file(path))
