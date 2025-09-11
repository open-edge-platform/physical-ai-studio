from fastapi import APIRouter, FastAPI
from typing import List, Literal

router = APIRouter()


@router.get("/cameras")
async def get_cameras() -> List[dict]:
    return [{"hello": "world"}]
