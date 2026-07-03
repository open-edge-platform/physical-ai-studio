from fastapi import APIRouter
from fastapi.responses import FileResponse

from robots.catalog.assets import resolve_robot_asset_path, resolve_robot_urdf_path
from schemas.robot import RobotType

router = APIRouter(prefix="/api/robots/catalog", tags=["Robot Catalog"])


@router.get("/{robot_type}/urdf")
async def get_robot_catalog_urdf(robot_type: RobotType) -> FileResponse:
    """Return the URDF file for a catalog robot type."""
    resolved_path = resolve_robot_urdf_path(robot_type=robot_type)
    return FileResponse(resolved_path)


@router.get("/{robot_type}/{asset_path:path}")
async def get_robot_catalog_asset(
    robot_type: RobotType,
    asset_path: str,
) -> FileResponse:
    """Return an asset file referenced by a catalog robot URDF."""
    resolved_path = resolve_robot_asset_path(robot_type=robot_type, asset_path=asset_path)
    return FileResponse(resolved_path)
