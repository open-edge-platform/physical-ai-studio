from functools import lru_cache
from typing import Annotated, cast
from uuid import UUID

from fastapi import Depends, status
from fastapi.exceptions import HTTPException
from fastapi.requests import HTTPConnection
from sqlalchemy.ext.asyncio import AsyncSession

from core.scheduler import Scheduler
from db.engine import get_async_db_session
from robots.robot_client_factory import RobotClientFactory
from services import (
    DatasetDownloadService,
    DatasetService,
    EpisodeThumbnailService,
    ModelDownloadService,
    ModelMetricsService,
    ModelService,
    ProjectCameraService,
    ProjectService,
    ProjectThumbnailService,
    RemoteServerService,
    RemoteTrainerService,
    RobotService,
)
from services.camera_claims import CameraClaimRegistry
from services.dataset_import.service import DatasetImportService
from services.environment_service import EnvironmentService
from services.event_processor import EventProcessor
from services.job_service import JobService
from services.log_service import LogService
from services.robot_catalog_service import RobotCatalogService
from services.snapshot_service import SnapshotService
from services.system_service import SystemService
from settings import Settings, get_settings
from utils.serial_robot_tools import RobotConnectionManager

SettingsDep = Annotated[Settings, Depends(get_settings)]
AsyncSessionDep = Annotated[AsyncSession, Depends(get_async_db_session)]


def is_valid_uuid(identifier: str) -> bool:
    """Check if a given string identifier is formatted as a valid UUID.

    :param identifier: String to check
    :return: True if valid UUID, False otherwise
    """
    try:
        UUID(identifier)
    except ValueError:
        return False
    return True


@lru_cache
def get_system_service() -> SystemService:
    """Provide a SystemService instance for querying system hardware."""
    return SystemService()


SystemServiceDep = Annotated[SystemService, Depends(get_system_service)]


def get_project_service(session: AsyncSessionDep) -> ProjectService:
    """Provide a ProjectService instance for managing projects."""
    return ProjectService(session)


ProjectServiceDep = Annotated[ProjectService, Depends(get_project_service)]


def get_remote_trainer_service(session: AsyncSessionDep) -> RemoteTrainerService:
    """Provide a request-scoped service for configured remote trainers."""
    return RemoteTrainerService(session)


RemoteTrainerServiceDep = Annotated[RemoteTrainerService, Depends(get_remote_trainer_service)]


@lru_cache
def get_robot_catalog_service() -> RobotCatalogService:
    """Provide a RobotCatalogService instance for the robot catalog."""
    return RobotCatalogService()


RobotCatalogServiceDep = Annotated[RobotCatalogService, Depends(get_robot_catalog_service)]


def get_remote_server_service(session: AsyncSessionDep) -> RemoteServerService:
    """Provide a request-scoped service for SSH-provisioned training servers."""
    return RemoteServerService(session)


RemoteServerServiceDep = Annotated[RemoteServerService, Depends(get_remote_server_service)]


def get_robot_service(session: AsyncSessionDep, catalog_service: RobotCatalogServiceDep) -> RobotService:
    """Provide a RobotService instance for managing robots in a project."""
    return RobotService(session, catalog_service.registry)


RobotServiceDep = Annotated[RobotService, Depends(get_robot_service)]


def get_robot_manager_service(request: HTTPConnection) -> RobotConnectionManager:
    """Provide a RobotConnectionManager instance."""
    robot_manager = getattr(request.app.state, "robot_manager", None)

    if robot_manager is None:
        raise RuntimeError("Robot manager not initialized")

    return cast("RobotConnectionManager", robot_manager)


RobotConnectionManagerDep = Annotated[RobotConnectionManager, Depends(get_robot_manager_service)]


def get_robot_client_factory(
    robot_manager: RobotConnectionManagerDep,
    catalog_service: RobotCatalogServiceDep,
) -> RobotClientFactory:
    """Provide a RobotClientFactory bound to the application robot manager.

    Request scoped: the factory is a thin wrapper around the shared
    RobotConnectionManager and is the seam used to fake robot hardware in tests.
    """
    return RobotClientFactory(robot_manager=robot_manager, catalog_registry=catalog_service.registry)


RobotClientFactoryDep = Annotated[RobotClientFactory, Depends(get_robot_client_factory)]


def get_camera_service(session: AsyncSessionDep, catalog_service: RobotCatalogServiceDep) -> ProjectCameraService:
    """Provide a ProjectCameraService instance for managing cameras in a project."""
    return ProjectCameraService(session, catalog_service.registry)


ProjectCameraServiceDep = Annotated[ProjectCameraService, Depends(get_camera_service)]


def get_environment_service(session: AsyncSessionDep, catalog_service: RobotCatalogServiceDep) -> EnvironmentService:
    """Provide a EnvironmentService instance for managing environments in a project."""
    return EnvironmentService(session, catalog_service.registry)


EnvironmentServiceDep = Annotated[EnvironmentService, Depends(get_environment_service)]


def get_dataset_service(session: AsyncSessionDep) -> DatasetService:
    """Provides a DatasetService instance for managing datasets."""
    return DatasetService(session)


DatasetServiceDep = Annotated[DatasetService, Depends(get_dataset_service)]


@lru_cache
def get_dataset_download_service() -> DatasetDownloadService:
    """Provides a DatasetDownloadService instance for dataset exports."""
    return DatasetDownloadService()


DatasetDownloadServiceDep = Annotated[DatasetDownloadService, Depends(get_dataset_download_service)]


@lru_cache
def get_episode_thumbnail_service() -> EpisodeThumbnailService:
    """Provides a service for building episode thumbnails."""
    return EpisodeThumbnailService()


EpisodeThumbnailServiceDep = Annotated[EpisodeThumbnailService, Depends(get_episode_thumbnail_service)]


def get_project_thumbnail_service(
    episode_thumbnail_service: EpisodeThumbnailServiceDep,
) -> ProjectThumbnailService:
    """Provides a service for building project thumbnails."""
    return ProjectThumbnailService(episode_thumbnail_service=episode_thumbnail_service)


ProjectThumbnailServiceDep = Annotated[ProjectThumbnailService, Depends(get_project_thumbnail_service)]


def get_model_service(session: AsyncSessionDep) -> ModelService:
    """Provides a ModelService instance for managing models."""
    return ModelService(session)


ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]


def get_model_metrics_service(settings: SettingsDep) -> ModelMetricsService:
    """Provides a ModelMetricsService instance for reading training metrics.

    Not cached: the constructor is trivial, and caching on a request-scoped
    argument would retain every request for the lifetime of the process.
    """
    return ModelMetricsService(settings=settings)


ModelMetricsServiceDep = Annotated[ModelMetricsService, Depends(get_model_metrics_service)]


@lru_cache
def get_model_download_service() -> ModelDownloadService:
    """Provides a ModelDownloadService instance for model exports."""
    return ModelDownloadService()


ModelDownloadServiceDep = Annotated[ModelDownloadService, Depends(get_model_download_service)]


def get_job_service(session: AsyncSessionDep) -> JobService:
    """Provides a JobService instance for managing jobs."""
    return JobService(session, RemoteTrainerService(session), RemoteServerService(session))


JobServiceDep = Annotated[JobService, Depends(get_job_service)]


def get_dataset_import_service(session: AsyncSessionDep) -> DatasetImportService:
    """Provides a DatasetImportService instance for dataset import jobs."""
    return DatasetImportService(session)


DatasetImportServiceDep = Annotated[DatasetImportService, Depends(get_dataset_import_service)]


def get_snapshot_service(session: AsyncSessionDep) -> SnapshotService:
    """Provide a request-scoped snapshot service."""
    return SnapshotService(session)


SnapshotServiceDep = Annotated[SnapshotService, Depends(get_snapshot_service)]


def get_log_service(settings: SettingsDep, job_service: JobServiceDep) -> LogService:
    """Provides a LogService instance for managing logs."""
    return LogService(settings=settings, job_service=job_service)


LogServiceDep = Annotated[LogService, Depends(get_log_service)]


def get_project_id(project_id: str) -> UUID:
    """Initialize and validates a project ID."""
    if not is_valid_uuid(project_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid project ID")
    return UUID(project_id)


def get_dataset_id(dataset_id: str) -> UUID:
    """Initialize and validates a dataset ID."""
    if not is_valid_uuid(dataset_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid dataset ID")
    return UUID(dataset_id)


def get_model_id(model_id: str) -> UUID:
    """Initialize and validates a model ID."""
    if not is_valid_uuid(model_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid model ID")
    return UUID(model_id)


def get_robot_id(robot_id: str) -> UUID:
    """Initialize and validates a robot ID."""
    if not is_valid_uuid(robot_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid robot ID")
    return UUID(robot_id)


def get_camera_id(camera_id: str) -> UUID:
    """Initialize and validates a camera ID."""
    if not is_valid_uuid(camera_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid camera ID")
    return UUID(camera_id)


def get_job_id(job_id: str) -> UUID:
    """Initialize and validates a project ID."""
    if not is_valid_uuid(job_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid job ID")
    return UUID(job_id)


def get_environment_id(environment_id: str) -> UUID:
    """Initialize and validates an environment ID."""
    if not is_valid_uuid(environment_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid environment ID")
    return UUID(environment_id)


def get_remote_server_id(remote_server_id: str) -> UUID:
    """Initialize and validate a remote server ID."""
    if not is_valid_uuid(remote_server_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid remote server ID")
    return UUID(remote_server_id)


def get_scheduler(request: HTTPConnection) -> Scheduler:
    """Provide the global Scheduler instance.

    Typed as HTTPConnection so it resolves for both HTTP routes and WebSocket endpoints.
    """
    return request.app.state.scheduler


SchedulerDep = Annotated[Scheduler, Depends(get_scheduler)]


def get_event_processor_ws(request: HTTPConnection) -> EventProcessor:
    """Provide the global event_processor instance for WebSocket."""
    return request.app.state.event_processor


EventProcessorDep = Annotated[EventProcessor, Depends(get_event_processor_ws)]


def get_camera_claim_registry(request: HTTPConnection) -> CameraClaimRegistry:
    """Return the process-local camera claim registry."""
    registry = getattr(request.app.state, "camera_claim_registry", None)
    if registry is None:
        raise RuntimeError("Camera claim registry is not initialized")
    return cast("CameraClaimRegistry", registry)


CameraClaimRegistryDep = Annotated[CameraClaimRegistry, Depends(get_camera_claim_registry)]
