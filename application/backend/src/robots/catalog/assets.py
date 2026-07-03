from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, status

from schemas.robot import RobotType

BUILTIN_ROBOT_ASSETS_ROOT = Path(__file__).resolve().parents[2] / "static" / "robot-assets"

_ROBOT_URDF_PATHS = {
    RobotType.SO101_FOLLOWER: Path("SO101/so101_new_calib.urdf"),
    RobotType.SO101_LEADER: Path("SO101/so101_new_calib.urdf"),
    RobotType.TROSSEN_WIDOWXAI_FOLLOWER: Path("widowx/urdf/generated/wxai/wxai_follower.urdf"),
    RobotType.TROSSEN_WIDOWXAI_LEADER: Path("widowx/urdf/generated/wxai/wxai_follower.urdf"),
    RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER: Path("widowx/urdf/generated/stationary_ai.urdf"),
    RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER: Path("widowx/urdf/generated/stationary_ai.urdf"),
}


def get_builtin_robot_assets_root() -> Path:
    """Return the backend-owned directory for built-in robot assets."""
    return BUILTIN_ROBOT_ASSETS_ROOT


def builtin_robot_assets_are_available() -> bool:
    """Return whether all built-in robot URDF assets are present locally."""
    root = get_builtin_robot_assets_root()
    return all((root / relative_path).is_file() for relative_path in set(_ROBOT_URDF_PATHS.values()))


def resolve_robot_urdf_path(robot_type: RobotType) -> Path:
    """Resolve the local URDF file for a supported catalog robot type."""
    try:
        urdf_relative_path = _ROBOT_URDF_PATHS[robot_type]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="URDF is unavailable for the requested robot type."
        ) from None

    return _resolve_robot_path(relative_path=urdf_relative_path)


def resolve_robot_asset_path(robot_type: RobotType, asset_path: str) -> Path:
    """Resolve a local asset file referenced by a robot URDF."""
    relative_path = Path(asset_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access to the requested file is forbidden.")

    try:
        package_root = Path(_ROBOT_URDF_PATHS[robot_type].parts[0])
    except (IndexError, KeyError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Assets are unavailable for the requested robot type."
        ) from None

    return _resolve_robot_path(relative_path=package_root / relative_path)


def _resolve_robot_path(relative_path: Path) -> Path:
    root = get_builtin_robot_assets_root().resolve()

    requested_path = (root / relative_path).resolve()
    if not requested_path.is_relative_to(root):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access to the requested file is forbidden.")
    if not requested_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    return requested_path
