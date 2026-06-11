import asyncio
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from physicalai.export.backends import ExportBackend
from sse_starlette import EventSourceResponse
from starlette import status
from starlette.background import BackgroundTask

from api.dependencies import (
    get_dataset_service,
    get_job_service,
    get_model_download_service,
    get_model_id,
    get_model_metrics_service,
    get_model_service,
)
from api.utils import safe_archive_name
from exceptions import ResourceNotFoundError, ResourceType
from internal_datasets.utils import get_internal_dataset
from schemas import ModelDetailResponse
from schemas.job import TrainJob
from services import DatasetService, JobService, ModelDownloadService, ModelMetricsService, ModelService

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("/{model_id}")
async def get_model_by_id(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
    job_service: Annotated[JobService, Depends(get_job_service)],
) -> ModelDetailResponse:
    """Get model by id with per-backend export details and training job info."""
    model = await model_service.get_model_by_id(model_id)
    exports = model_service.get_backend_details(model)
    hparams = model_service.get_hparams(model)

    training_job: TrainJob | None = None
    if model.train_job_id is not None:
        job = await job_service.get_job_by_id(model.train_job_id)
        training_job = job if isinstance(job, TrainJob) else None

    training_summary = model_service.get_training_summary(training_job)

    return ModelDetailResponse(
        model=model,
        exports=exports,
        training_summary=training_summary,
        hparams=hparams,
    )


@router.get("/{model_id}/tasks")
async def get_tasks_of_model(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> list[str]:
    """Get availabe tasks for model."""
    model = await model_service.get_model_by_id(model_id)
    if model.dataset_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model has no dataset associated.")
    dataset = await dataset_service.get_dataset_by_id(model.dataset_id)
    return get_internal_dataset(dataset).get_tasks()


@router.get("/{model_id}/download")
async def model_download_endpoint(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
    model_download_service: Annotated[ModelDownloadService, Depends(get_model_download_service)],
    include_snapshot: bool = False,
) -> FileResponse:
    """Download model folder as a zip archive.

    By default the dataset snapshot that was used for training is excluded
    from the archive.  Pass ``include_snapshot=true`` to include it.
    """
    model = await model_service.get_model_by_id(model_id)
    model_path = Path(model.path).resolve()

    if not model_path.exists() or not model_path.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model path not found.")

    archive_path = await asyncio.to_thread(
        model_download_service.create_model_archive,
        model_path,
        include_snapshot=include_snapshot,
    )
    filename = f"{safe_archive_name(model.name, fallback='model')}.zip"
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=filename,
        background=BackgroundTask(archive_path.unlink, missing_ok=True),
    )


@router.get("/{model_id}/exports/{backend}/download")
async def download_model_backend(
    model_id: Annotated[UUID, Depends(get_model_id)],
    backend: ExportBackend,
    model_service: Annotated[ModelService, Depends(get_model_service)],
    model_download_service: Annotated[ModelDownloadService, Depends(get_model_download_service)],
) -> FileResponse:
    """Download a single backend export as a zip archive."""
    model = await model_service.get_model_by_id(model_id)
    if backend.value not in model.available_backends:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backend '{backend.value}' is not available for this model.",
        )

    export_dir = Path(model.path) / "exports" / backend.value
    if not export_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Export directory for backend '{backend.value}' not found on disk.",
        )

    archive_path = await asyncio.to_thread(model_download_service.create_backend_archive, export_dir, backend.value)
    filename = f"{safe_archive_name(model.name, fallback='model')}_{backend.value}.zip"
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=filename,
        background=BackgroundTask(archive_path.unlink, missing_ok=True),
    )


@router.get("/{model_id}/metrics")
async def stream_metrics(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
    model_metrics_service: Annotated[ModelMetricsService, Depends(get_model_metrics_service)],
) -> EventSourceResponse:
    """Get an EventSourceResponse from the metrics of a model."""
    model = await model_service.get_model_by_id(model_id)
    metrics_path = await model_metrics_service.get_model_metrics_path(model)
    if metrics_path.exists():
        return EventSourceResponse(model_metrics_service.tail_csv_file(metrics_path))
    return EventSourceResponse(model_metrics_service.empty_metrics_stream())


@router.delete("/{model_id}")
async def remove_model(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> None:
    """Fetch all projects."""
    model = await model_service.get_model_by_id(model_id)
    if model is None:
        raise ResourceNotFoundError(ResourceType.MODEL, model_id)
    await model_service.delete_model(model)
