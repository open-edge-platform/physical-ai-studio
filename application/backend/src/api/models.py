from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_model_service
from schemas import Model
from services import ModelService

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("")
async def list_models(
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> list[Model]:
    """Fetch all projects."""
    return await model_service.get_model_list()
