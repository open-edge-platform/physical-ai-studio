from fastapi import APIRouter
from pydantic import BaseModel

from settings import get_settings

router = APIRouter(prefix="/api/settings", tags=["Settings"])


class UserSettings(BaseModel):
    geti_action_dataset_path: str


@router.get("")
async def get_user_settings() -> UserSettings:
    """Get user settings"""
    settings = get_settings()
    return UserSettings(geti_action_dataset_path=str(settings.datasets_dir))
