from fastapi import APIRouter
from fastapi.exceptions import HTTPException

from schemas import Dataset, ProjectConfig
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
    project =  load_project(project_id)
    repo_id = f"{repo}/{id}"
    if repo_id in project.datasets:
        return get_dataset(repo_id)
    raise HTTPException(status_code=404, detail="Dataset not found")
