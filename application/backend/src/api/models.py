from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from api.dependencies import get_model_service, validate_uuid
from exceptions import ResourceNotFoundError, ResourceType
from services import ModelService
from schemas import Model

router = APIRouter(prefix="/api/models", tags=["Models"])

@router.get("/{model_id}")
async def get_model_by_id(
    model_id: Annotated[UUID, Depends(validate_uuid)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> Model:
    """Get model by id."""
    return await model_service.get_model_by_id(model_id)


@router.delete("")
async def remove_model(
    model_id: Annotated[UUID, Depends(validate_uuid)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> None:
    """Fetch all projects."""
    model = await model_service.get_model_by_id(model_id)
    if model is None:
        raise ResourceNotFoundError(ResourceType.MODEL, model_id)
    await model_service.delete_model(model)
