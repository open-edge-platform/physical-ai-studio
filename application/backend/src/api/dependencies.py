from functools import lru_cache

from fastapi import Request

from services.project_service import ProjectService
from webrtc.manager import WebRTCManager


def get_webrtc_manager(request: Request) -> WebRTCManager:
    """Provides the global WebRTCManager instance from FastAPI application's state."""
    return request.app.state.webrtc_manager

@lru_cache
def get_project_service() -> ProjectService:
    """Provides a ProjectService instance for managing projects."""
    return ProjectService()