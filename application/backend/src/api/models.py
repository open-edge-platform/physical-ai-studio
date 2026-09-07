import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from physicalai.config import to_yaml
from physicalai.export.backends import ExportBackend
from sse_starlette import EventSourceResponse
from starlette import status
from starlette.background import BackgroundTask

from api.dependencies import (
    EnvironmentServiceDep,
    get_dataset_service,
    get_job_service,
    get_model_download_service,
    get_model_id,
    get_model_metrics_service,
    get_model_service,
    get_robot_catalog_service,
    get_robot_client_factory,
    get_robot_manager_service,
)
from api.utils import safe_archive_name
from exceptions import ResourceNotFoundError, ResourceType
from internal_datasets.utils import get_internal_read_dataset
from robots.robot_client_factory import RobotClientFactory
from runtime.config_builder import (
    RUNTIME_FPS,
    build_runtime_config,
    policy_source_fragment,
    runtime_config_change_me,
    runtime_export_readme,
)
from schemas import Model, ModelDetailResponse
from schemas.job import TrainJob
from services import DatasetService, JobService, ModelDownloadService, ModelMetricsService, ModelService
from services.environment_service import EnvironmentService

router = APIRouter(prefix="/api/models", tags=["Models"])


async def _runtime_recipe_texts(
    *,
    model: Model,
    environment_id: UUID,
    backend: ExportBackend,
    device: str,
    task: str | None,
    environment_service: EnvironmentService,
    robot_client_factory: RobotClientFactory,
) -> tuple[str, str]:
    """Build runtime.yaml and README for a model download that includes a recipe."""
    if backend.value not in model.available_backends:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backend '{backend.value}' is not available for this model.",
        )

    environment = await environment_service.get_environment_by_id(model.project_id, environment_id)
    if len(environment.robots) != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Runtime export requires exactly one follower robot",
        )

    fragment = policy_source_fragment(
        export_dir=f"./exports/{backend.value}",
        backend=backend.value,
        device=device,
        task=(task.strip() or None) if task else None,
    )
    try:
        document = await build_runtime_config(
            follower=environment.robots[0].robot,
            leader=None,
            cameras=environment.cameras,
            robot_factory=robot_client_factory,
            fps=RUNTIME_FPS,
            allow_stored_port=True,
            action_source=fragment,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    unresolved = runtime_config_change_me(document)
    comments = "".join(f"# CHANGE_ME: replace machine-specific device path {path}\n" for path in unresolved)
    return comments + to_yaml(document), runtime_export_readme(document, unresolved=unresolved)


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
    return get_internal_read_dataset(dataset).get_tasks()


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
async def download_model_backend(  # noqa: PLR0913
    request: Request,
    model_id: Annotated[UUID, Depends(get_model_id)],
    backend: ExportBackend,
    model_service: Annotated[ModelService, Depends(get_model_service)],
    model_download_service: Annotated[ModelDownloadService, Depends(get_model_download_service)],
    environment_service: EnvironmentServiceDep,
    environment_id: UUID | None = None,
    device: str | None = None,
    task: str | None = None,
) -> FileResponse:
    """Download a single backend export as a zip archive.

    Pass ``environment_id`` and ``device`` to include ``runtime.yaml`` and a
    README so the zip runs with ``physicalai run``. Weights stay under
    ``exports/<backend>/`` so the recipe's ``export_dir`` resolves.
    """
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

    if environment_id is None:
        archive_path = await asyncio.to_thread(model_download_service.create_backend_archive, export_dir, backend.value)
        filename = f"{safe_archive_name(model.name, fallback='model')}_{backend.value}.zip"
    else:
        if device is None or not device.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Runtime export requires device",
            )
        # Resolve hardware only when building a recipe. FastAPI would otherwise
        # require a live robot manager for weights-only downloads too.
        factory_override = request.app.dependency_overrides.get(get_robot_client_factory)
        if factory_override is not None:
            robot_client_factory = factory_override()
        else:
            robot_client_factory = get_robot_client_factory(
                get_robot_manager_service(request),
                get_robot_catalog_service(),
            )
        runtime_yaml, readme = await _runtime_recipe_texts(
            model=model,
            environment_id=environment_id,
            backend=backend,
            device=device,
            task=task,
            environment_service=environment_service,
            robot_client_factory=robot_client_factory,
        )
        archive_path = await asyncio.to_thread(
            model_download_service.create_runtime_export,
            export_dir,
            backend.value,
            runtime_yaml=runtime_yaml,
            readme=readme,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        filename = f"studio-runtime-{safe_archive_name(model.name, fallback='model')}-{stamp}.zip"

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
