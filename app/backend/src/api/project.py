from fastapi import APIRouter

from schemas import ProjectConfig
from storage.storage import load_project, load_projects, write_project

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

@router.get("/{id}")
async def get_project(id: str) -> ProjectConfig:
    """Get project by id"""

    return load_project(id)