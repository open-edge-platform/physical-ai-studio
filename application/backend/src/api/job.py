from typing import Annotated

from fastapi import APIRouter, Depends, Body

from api.dependencies import get_job_service
from schemas import Job
from schemas.job import TrainJobPayload, JobSubmitted
from services import JobService

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
