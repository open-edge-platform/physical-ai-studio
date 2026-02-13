from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import Depends, status
from fastapi.exceptions import HTTPException
from fastapi.requests import HTTPConnection

from core.scheduler import Scheduler
from services import DatasetService, JobService, ModelService, ProjectCameraService, ProjectService, RobotService
from services.environment_service import EnvironmentService
from services.event_processor import EventProcessor
from services.robot_calibration_service import RobotCalibrationService
from settings import get_settings
from utils.serial_robot_tools import RobotConnectionManager
from workers.camera_worker_registry import CameraWorkerRegistry
from workers.robot_worker_registry import RobotWorkerRegistry


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
def get_project_service() -> ProjectService:
    """Provide a ProjectService instance for managing projects."""
    return ProjectService()


@lru_cache
def get_robot_service() -> RobotService:
    """Provide a RobotService instance for managing robots in a project."""
    return RobotService()


def get_robot_manager_service(request: HTTPConnection) -> RobotConnectionManager:
    """Provide a RobotConnectionManager instance."""
    robot_manager = getattr(request.app.state, "robot_manager", None)

    if robot_manager is None:
        raise RuntimeError("Robot manager not initialized")

    return robot_manager


RobotConnectionManagerDep = Annotated[RobotConnectionManager, Depends(get_robot_manager_service)]


def get_robot_calibration_service(robot_manager: RobotConnectionManagerDep) -> RobotCalibrationService:
    """Provide a RobotCalibrationService instance for managing robot calibrations."""
    return RobotCalibrationService(robot_manager, settings=get_settings())


RobotCalibrationServiceDep = Annotated[RobotCalibrationService, Depends(get_robot_calibration_service)]


@lru_cache
def get_camera_service() -> ProjectCameraService:
    """Provide a ProjectCameraService instance for managing cameras in a project."""
    return ProjectCameraService()


@lru_cache
def get_environment_service() -> EnvironmentService:
    """Provide a EnvironmentService instance for managing environments in a project."""
    return EnvironmentService()


@lru_cache
def get_dataset_service() -> DatasetService:
    """Provides a DatasetService instance for managing datasets."""
    return DatasetService()


@lru_cache
def get_model_service() -> ModelService:
    """Provides a ModelService instance for managing models."""
    return ModelService()


@lru_cache
def get_job_service() -> JobService:
    """Provides a JobService instance for managing jobs."""
    return JobService()


def get_project_id(project_id: str) -> UUID:
    """Initialize and validates a project ID."""
    if not is_valid_uuid(project_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid project ID")
    return UUID(project_id)


def get_robot_id(robot_id: str) -> UUID:
    """Initialize and validates a robot ID."""
    if not is_valid_uuid(robot_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid robot ID")
    return UUID(robot_id)


def get_calibration_id(calibration_id: str) -> UUID:
    """Initialize and validates a calibration ID."""
    if not is_valid_uuid(calibration_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid calibration ID")
    return UUID(calibration_id)


def get_camera_id(camera_id: str) -> UUID:
    """Initialize and validates a camera ID."""
    if not is_valid_uuid(camera_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid camera ID")
    return UUID(camera_id)


def get_environment_id(environment_id: str) -> UUID:
    """Initialize and validates an environment ID."""
    if not is_valid_uuid(environment_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid environment ID")
    return UUID(environment_id)


def validate_uuid(uuid: str) -> UUID:
    """Initialize and validates UUID."""
    if not is_valid_uuid(uuid):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid ID")
    return UUID(uuid)


def get_scheduler(request: HTTPConnection) -> Scheduler:
    """Provide the global Scheduler instance."""
    return request.app.state.scheduler


def get_scheduler_ws(request: HTTPConnection) -> Scheduler:
    """Provide the global Scheduler instance for WebSocket."""
    return request.app.state.scheduler


def get_event_processor_ws(request: HTTPConnection) -> EventProcessor:
    """Provide the global event_processor instance for WebSocket."""
    return request.app.state.event_processor


def get_camera_registry(request: HTTPConnection) -> CameraWorkerRegistry:
    """Dependency to get camera worker registry."""
    registry = getattr(request.app.state, "camera_registry", None)
    if registry is None:
        raise RuntimeError("Camera worker registry not initialized")
    return registry


def get_robot_registry(request: HTTPConnection) -> RobotWorkerRegistry:
    """Dependency to get robot worker registry."""
    registry = getattr(request.app.state, "robot_registry", None)
    if registry is None:
        raise RuntimeError("Robot worker registry not initialized")
    return registry


CameraRegistryDep = Annotated[CameraWorkerRegistry, Depends(get_camera_registry)]
RobotRegistryDep = Annotated[RobotWorkerRegistry, Depends(get_robot_registry)]
