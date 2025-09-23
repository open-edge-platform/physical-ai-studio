from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException

from api.dependencies import get_project_service
from schemas import Dataset, Project
from services.project_service import ProjectService
from storage.storage import load_project, write_project
from utils.dataset import get_dataset

router = APIRouter()


@router.get("")
async def list_projects(project_service: Annotated[ProjectService, Depends(get_project_service)]) -> list[Project]:
    """Fetch all projects"""
    projects =  project_service.list_projects()
    print(projects)
    return projects

@router.put("")
async def create_project(
    project: Project,
    project_service: Annotated[ProjectService, Depends(get_project_service)]) -> Project:
    """Create a new project"""
    return project_service.create_project(project)


@router.get("/{id}")
async def get_project(id: str, project_service: Annotated[ProjectService, Depends(get_project_service)]) -> Project:
    """Get project by id"""

    return project_service.get_project_by_id(id)


@router.get("/{project_id}/datasets/{repo}/{id}")
async def get_dataset_of_project(project_id: str, repo: str, id: str) -> Dataset:
    """Get dataset of project by id"""
    project = load_project(project_id)
    repo_id = f"{repo}/{id}"
    if repo_id in project.datasets:
        return get_dataset(repo_id)
    raise HTTPException(status_code=404, detail="Dataset not found")
