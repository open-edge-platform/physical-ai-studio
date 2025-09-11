from fastapi import APIRouter

router = APIRouter()


@router.get("/cameras")
async def get_cameras() -> list[dict]:
    """Get cameras example request"""
    return [{"hello": "world"}]
