from fastapi import Request
from functools import lru_cache
from webrtc.manager import WebRTCManager
from services.project_service import ProjectService


def get_webrtc_manager(request: Request) -> WebRTCManager:
    """Provides the global WebRTCManager instance from FastAPI application's state."""
    return request.app.state.webrtc_manager

@lru_cache
def get_project_service() -> ProjectService:
    """Provides a ProjectService instance for managing projects."""
    return ProjectService()