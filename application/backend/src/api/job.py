from typing import Annotated

from uuid import UUID
from fastapi import APIRouter, Depends, Body

from api.dependencies import get_job_service, validate_uuid, get_scheduler
from schemas import Job
from schemas.job import TrainJobPayload, JobSubmitted, JobStatus
from services import JobService
from core.scheduler import Scheduler

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
) -> JobSubmitted:
    """Endpoint to submit a training job"""
    return await job_service.submit_train_job(payload=payload)

@router.post("/{job_id}")
async def submit_train_job(
    job_id: Annotated[UUID, Depends(validate_uuid)],
    job_service: Annotated[JobService, Depends(get_job_service)],
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
) -> None:
    """Endpoint to interrupt job"""
    job = await job_service.get_job_by_id(job_id)
    if job.status == JobStatus.RUNNING:
        scheduler.training_interrupt_event.set()
