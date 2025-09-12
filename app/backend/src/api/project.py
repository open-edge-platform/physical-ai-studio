from fastapi import APIRouter
from uuid import uuid4
from schemas import ProjectConfig, CameraConfig,RobotConfig

from storage.storage import load_projects, write_project

router = APIRouter()

@router.get("")
async def get_projects() -> list[ProjectConfig]:
    """Get all projects"""

    return load_projects()

@router.put("")
async def create_project(project: ProjectConfig) -> str:
    """Create a new project"""
    write_project(project)
    return project.id
