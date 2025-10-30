from typing import Annotated

from uuid import UUID
from fastapi import APIRouter, Depends, Body, WebSocket, WebSocketDisconnect

from api.dependencies import get_job_service, validate_uuid, get_scheduler, get_event_processor_ws
from schemas import Job
from schemas.job import TrainJobPayload, JobStatus
from services import JobService
from core.scheduler import Scheduler
from services.event_processor import EventType, EventProcessor

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])


@router.get("")
async def list_jobs(
    job_service: Annotated[JobService, Depends(get_job_service)],
) -> list[Job]:
    """Fetch all jobs."""
    return await job_service.get_job_list()

@router.post("/train")
async def submit_train_job(
    job_service: Annotated[JobService, Depends(get_job_service)],
    payload: Annotated[TrainJobPayload, Body()],
) -> Job:
    """Endpoint to submit a training job"""
    return await job_service.submit_train_job(payload=payload)

@router.post("/{job_id}/interrupt")
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


@router.websocket("/ws")
async def jobs_websocket(  # noqa: C901
    websocket: WebSocket,
    event_processor: Annotated[EventProcessor, Depends(get_event_processor_ws)],
) -> None:
    """Robot control websocket."""
    await websocket.accept()

    async def send_data(event, payload):
        print("sending data: ")
        await websocket.send_json({
            "event": event,
            "data": payload.model_dump(mode="json"),
        })

    event_processor.subscribe([EventType.JOB_UPDATE], send_data)

    try:
        while True:
            data = await websocket.receive_json("text")
            print(data)

    except WebSocketDisconnect:
        print("Except: disconnected!")

    event_processor.unsubscribe([EventType.JOB_UPDATE], send_data)
    print("websocket handling done...")
