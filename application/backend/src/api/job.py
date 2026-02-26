from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from loguru import logger

from api.dependencies import get_event_processor_ws, get_job_service, get_scheduler, validate_uuid
from core.scheduler import Scheduler
from schemas import Job
from schemas.job import JobStatus, TrainJobPayload
from services import JobService
from services.event_processor import EventProcessor, EventType

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])


@router.get("")
async def list_jobs(
    job_service: Annotated[JobService, Depends(get_job_service)],
) -> list[Job]:
    """Fetch all jobs."""
    return await job_service.get_job_list()


@router.post(":train")
async def submit_train_job(
    job_service: Annotated[JobService, Depends(get_job_service)],
    payload: Annotated[TrainJobPayload, Body()],
) -> Job:
    """Endpoint to submit a training job"""
    return await job_service.submit_train_job(payload=payload)


@router.post("/{job_id}:interrupt")
async def interrupt_job(
    job_id: Annotated[UUID, Depends(validate_uuid)],
    job_service: Annotated[JobService, Depends(get_job_service)],
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
) -> None:
    """Endpoint to interrupt job"""
    job = await job_service.get_job_by_id(job_id)
    if job is not None:
        if job.status == JobStatus.RUNNING:
            scheduler.training_interrupt_event.set()
        await job_service.update_job_status(job_id, status=JobStatus.CANCELED)


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
