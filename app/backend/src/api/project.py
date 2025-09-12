from fastapi import APIRouter

from schemas import ProjectConfig, Dataset
from storage.storage import load_project, load_projects, write_project
from utils.dataset import get_dataset

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

@router.get("/{project_id}/datasets/{repo}/{id}")
async def get_dataset_of_project(project_id: str, repo: str, id: str) -> Dataset:
    """Get dataset of project by id"""
    repo_id = f"{repo}/{id}"
    return get_dataset(repo_id)
