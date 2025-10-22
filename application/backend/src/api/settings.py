from fastapi import APIRouter
from pydantic import BaseModel

from storage.storage import GETI_ACTION_DATASETS

router = APIRouter(prefix="/api/settings", tags=["Settings"])


class UserSettings(BaseModel):
    geti_action_dataset_path: str


@router.get("")
async def get_user_settings() -> UserSettings:
    """Get user settings"""
    return UserSettings(geti_action_dataset_path=str(GETI_ACTION_DATASETS))
