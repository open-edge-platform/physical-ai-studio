from functools import lru_cache
from uuid import UUID

from fastapi import Request, status
from fastapi.exceptions import HTTPException

from services import ProjectService, DatasetService
from webrtc.manager import WebRTCManager


def is_valid_uuid(identifier: str) -> bool:
    """
    Check if a given string identifier is formatted as a valid UUID

    :param identifier: String to check
    :return: True if valid UUID, False otherwise
    """
    try:
        UUID(identifier)
    except ValueError:
        return False
    return True


def get_webrtc_manager(request: Request) -> WebRTCManager:
    """Provides the global WebRTCManager instance from FastAPI application's state."""
    return request.app.state.webrtc_manager

@lru_cache
def get_project_service() -> ProjectService:
    """Provides a ProjectService instance for managing projects."""
    return ProjectService()

@lru_cache
def get_dataset_service() -> DatasetService:
    """Provides a ProjectService instance for managing projects."""
    return DatasetService()


def get_project_id(project_id: str) -> UUID:
    """Initializes and validates a project ID"""
    if not is_valid_uuid(project_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid project ID")
    return UUID(project_id)
