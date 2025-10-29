from functools import lru_cache
from uuid import UUID

from fastapi import Request, WebSocket, status
from fastapi.exceptions import HTTPException

from core.scheduler import Scheduler
from services import DatasetService, ProjectService, ModelService, JobService
from services.event_processor import EventProcessor
from webrtc.manager import WebRTCManager


def is_valid_uuid(identifier: str) -> bool:
    """
    Check if a given string identifier is formatted as a valid UUID.

    :param identifier: String to check
    :return: True if valid UUID, False otherwise
    """
    try:
        UUID(identifier)
    except ValueError:
        return False
    return True


def get_webrtc_manager(request: Request) -> WebRTCManager:
    """Provide the global WebRTCManager instance from FastAPI application's state."""
    return request.app.state.webrtc_manager


@lru_cache
def get_project_service() -> ProjectService:
    """Provide a ProjectService instance for managing projects."""
    return ProjectService()


@lru_cache
def get_dataset_service() -> DatasetService:
    """Provides a DatasetService instance for managing datasets."""
    return DatasetService()

@lru_cache
def get_model_service() -> DatasetService:
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

def validate_uuid(uuid: str) -> UUID:
    """Initialize and validates UUID."""
    if not is_valid_uuid(uuid):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid ID")
    return UUID(uuid)


def get_scheduler(request: Request) -> Scheduler:
    """Provide the global Scheduler instance."""
    return request.app.state.scheduler

def get_scheduler_ws(request: WebSocket) -> Scheduler:
    """Provide the global Scheduler instance for WebSocket."""
    return request.app.state.scheduler

def get_event_processor_ws(request: WebSocket) -> EventProcessor:
    """Provide the global event_processor instance for WebSocket."""
    return request.app.state.event_processor
