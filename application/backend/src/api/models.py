from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.exceptions import HTTPException

from api.dependencies import get_model_service
from schemas import LeRobotDatasetInfo, Project, TeleoperationConfig, Model
from services import ModelService
from services.base import ResourceInUseError, ResourceNotFoundError
from utils.dataset import build_dataset_from_lerobot_dataset, build_project_config_from_dataset

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("")
async def list_models(
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> list[Model]:
    """Fetch all projects."""
    return model_service.list_models()
