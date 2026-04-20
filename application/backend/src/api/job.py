from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from loguru import logger
from sse_starlette import EventSourceResponse

from api.dependencies import (
    get_event_processor_ws,
    get_job_id,
    get_job_service,
    get_model_metrics_service,
    get_scheduler,
)
from core.scheduler import Scheduler
from exceptions import ResourceNotFoundError, ResourceType
from schemas import Job
from schemas.base_job import JobStatus
from schemas.job import TrainJobPayload
from services import JobService, ModelMetricsService
from services.event_processor import EventProcessor, EventType

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])


@router.get("")
async def list_jobs(
    job_service: Annotated[JobService, Depends(get_job_service)],
) -> list[Job]:
    """Fetch all jobs."""
    return await job_service.get_job_list()


@router.get("/{job_id}")
async def get_job(
    job_id: Annotated[UUID, Depends(get_job_id)],
    job_service: Annotated[JobService, Depends(get_job_service)],
) -> Job:
    """Fetch one job by id."""
    return await job_service.get_job_by_id(job_id)


@router.delete("/{job_id}")
async def delete_job(
    job_id: Annotated[UUID, Depends(get_job_id)],
    job_service: Annotated[JobService, Depends(get_job_service)],
) -> None:
    """Delete a job. Only allows deleting failed jobs"""
    await job_service.delete_job(job_id)


@router.post(":train")
async def submit_train_job(
    job_service: Annotated[JobService, Depends(get_job_service)],
    payload: Annotated[TrainJobPayload, Body()],
) -> Job:
    """Endpoint to submit a training job"""
    return await job_service.submit_train_job(payload=payload)


@router.post("/{job_id}:interrupt")
async def interrupt_job(
    job_id: Annotated[UUID, Depends(get_job_id)],
    job_service: Annotated[JobService, Depends(get_job_service)],
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
) -> None:
    """Endpoint to interrupt job"""
    job = await job_service.get_job_by_id(job_id)
    if job is None:
        raise ResourceNotFoundError(ResourceType.JOB, job_id)

    if job.status == JobStatus.RUNNING:
        scheduler.training_interrupt_event.set()
    await job_service.update_job_status(job_id, status=JobStatus.CANCELED)


@router.get("/{job_id}/model_metrics")
async def get_model_job_metrics(
    job_id: Annotated[UUID, Depends(get_job_id)],
    job_service: Annotated[JobService, Depends(get_job_service)],
    model_metrics_service: Annotated[ModelMetricsService, Depends(get_model_metrics_service)],
) -> EventSourceResponse:
    """Get model running metrics if job is a model job"""
    job = await job_service.get_job_by_id(job_id)
    metrics_path = await model_metrics_service.get_model_job_metrics_path(job)
    if metrics_path.exists():
        return EventSourceResponse(model_metrics_service.tail_csv_file(metrics_path))
    return EventSourceResponse(model_metrics_service.empty_metrics_stream())


@router.get("/ws", tags=["WebSocket"], summary="Job updates (WebSocket)", status_code=426)
async def jobs_websocket_openapi() -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/ws")
async def jobs_websocket(
    websocket: WebSocket,
    event_processor: Annotated[EventProcessor, Depends(get_event_processor_ws)],
) -> None:
    """Robot control websocket."""
    await websocket.accept()

    async def send_data(event: EventType, payload: Job):
        """Pass job update through to websocket."""
        await websocket.send_json(
            {
                "event": event,
                "data": payload.model_dump(mode="json"),
            }
        )

    event_processor.subscribe([EventType.JOB_UPDATE], send_data)

    try:
        while True:
            # TODO implement subscribing to specific events
            data = await websocket.receive_json("text")
            logger.info(f"Received websocket message: {data}")

    except WebSocketDisconnect:
        logger.info("Except: disconnected!")

    event_processor.unsubscribe([EventType.JOB_UPDATE], send_data)
    logger.info("Jobs Websocket handling done...")
