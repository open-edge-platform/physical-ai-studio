from typing import Annotated

from fastapi import APIRouter, Depends

from uuid import UUID
from api.dependencies import get_model_service, get_project_id
from schemas import Model
from services import ModelService

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("/{project_id}")
async def list_models(
    project_id: Annotated[UUID, Depends(get_project_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> list[Model]:
    """Fetch all projects."""
    return await model_service.get_project_models(project_id)
