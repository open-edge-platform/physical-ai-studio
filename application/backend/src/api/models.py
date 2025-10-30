from typing import Annotated

from fastapi import APIRouter, Depends

from uuid import UUID
from api.dependencies import get_model_service, get_project_id, validate_uuid
from schemas import Model
from services import ModelService
from pathlib import Path
from exceptions import ResourceNotFoundError, ResourceType

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("/{project_id}")
async def list_models(
    project_id: Annotated[UUID, Depends(get_project_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> list[Model]:
    """Fetch all projects."""
    return await model_service.get_project_models(project_id)

@router.delete("")
async def remove_model(
    model_id: Annotated[UUID, Depends(validate_uuid)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> None:
    """Fetch all projects."""
    model = await model_service.get_model_by_id(model_id)
    if model is None:
        raise ResourceNotFoundError(ResourceType.MODEL, model_id)
    model_path = Path(model.path).expanduser()
    model_path.unlink()
    model_path.parent.rmdir()
    await model_service.delete_model(model_id)
