from pathlib import Path
from typing import Annotated
from uuid import UUID
import csv
import json

from collections.abc import AsyncGenerator
from sse_starlette import EventSourceResponse, ServerSentEvent
from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from starlette import status
from starlette.background import BackgroundTask

from api.dependencies import get_dataset_service, get_model_download_service, get_model_id, get_model_service
from api.utils import safe_archive_name
from exceptions import ResourceNotFoundError, ResourceType
from internal_datasets.utils import get_internal_dataset
from schemas import Model
from services import DatasetService, ModelDownloadService, ModelService
from loguru import logger
from pathlib import Path
import asyncio
import anyio

from api.dependencies import get_dataset_service, get_model_service, get_model_id, get_model_metrics_service
from exceptions import ResourceNotFoundError, ResourceType
from internal_datasets.utils import get_internal_dataset
from schemas import Model
from services import DatasetService, ModelService, ModelMetricsService

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("/{model_id}")
async def get_model_by_id(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> Model:
    """Get model by id."""
    return await model_service.get_model_by_id(model_id)


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

    archive_path = model_download_service.create_model_archive(model_path, include_snapshot=include_snapshot)
    filename = f"{safe_archive_name(model.name, fallback='model')}.zip"
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


@router.delete("")
async def remove_model(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> None:
    """Fetch all projects."""
    model = await model_service.get_model_by_id(model_id)
    if model is None:
        raise ResourceNotFoundError(ResourceType.MODEL, model_id)
    await model_service.delete_model(model)
