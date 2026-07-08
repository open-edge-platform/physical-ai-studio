from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, status

from robots.catalog.types import RobotCatalogDefinition

from . import so101, widowxai

BUILTIN_ROBOT_ASSETS_ROOT = Path(__file__).resolve().parents[2] / "static" / "robot-assets"


def get_builtin_robot_assets_root() -> Path:
    """Return the backend-owned directory for built-in robot assets."""
    return BUILTIN_ROBOT_ASSETS_ROOT


def builtin_robot_assets_are_available() -> bool:
    """Return whether all built-in robot URDF assets are present locally."""
    root = get_builtin_robot_assets_root()
    definitions = so101.get_definitions() + widowxai.get_definitions()

    return all((root / Path(definition.urdf_relative_path)).is_file() for definition in definitions)


def resolve_robot_urdf_path(definition: RobotCatalogDefinition) -> Path:
    """Resolve the local URDF file for a supported catalog robot type."""
    return _resolve_robot_path(asset_path=Path(definition.urdf_relative_path))


def resolve_robot_asset_path(definition: RobotCatalogDefinition, asset_path: Path) -> Path:
    """Resolve a local asset file referenced by a robot URDF."""
    if asset_path.is_absolute() or ".." in asset_path.parts:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access to the requested file is forbidden.")

    urdf_relative = Path(definition.urdf_relative_path)
    if len(urdf_relative.parts) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Assets are unavailable for the requested robot type."
        )

    package_root = Path(urdf_relative.parts[0])
    return _resolve_robot_path(asset_path=package_root / asset_path)


def _resolve_robot_path(asset_path: Path) -> Path:
    root = get_builtin_robot_assets_root().resolve()

    requested_path = (root / asset_path).resolve()
    if not requested_path.is_relative_to(root):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access to the requested file is forbidden.")
    if not requested_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    return requested_path
