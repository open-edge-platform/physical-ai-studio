from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException, status

from . import so101, widowxai

if TYPE_CHECKING:
    from physicalai_studio_plugin import RobotCatalogDefinition

BUILTIN_ROBOT_ASSETS_ROOT = Path(__file__).resolve().parents[2] / "static" / "robot-assets"


def get_builtin_robot_assets_root() -> Path:
    """Return the backend-owned directory for built-in robot assets."""
    return BUILTIN_ROBOT_ASSETS_ROOT


def builtin_robot_assets_are_available() -> bool:
    """Return whether all built-in robot URDF assets are present locally."""
    root = get_builtin_robot_assets_root()
    definitions = so101.get_definitions() + widowxai.get_definitions()

    return all(
        definition.asset is not None and (root / definition.asset.urdf_relative_path).is_file()
        for definition in definitions
    )


def resolve_robot_urdf_path(definition: RobotCatalogDefinition) -> Path:
    """Resolve the local URDF file for a supported catalog robot type."""
    if definition.asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assets are unavailable for the requested robot type.",
        )
    return _resolve_robot_path(asset_path=definition.asset.urdf_relative_path, definition=definition)


def resolve_robot_relative_asset_path(definition: RobotCatalogDefinition, asset_path: Path) -> Path:
    """Resolve an asset path relative to the model's top-level asset directory."""
    if asset_path.is_absolute() or ".." in asset_path.parts:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access to the requested file is forbidden.")

    if definition.asset is None or not definition.asset.urdf_relative_path.parts:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assets are unavailable for the requested robot type.",
        )

    return _resolve_robot_path(
        asset_path=definition.asset.urdf_relative_path.parts[0] / asset_path,
        definition=definition,
    )


def _resolve_robot_path(asset_path: Path, definition: RobotCatalogDefinition) -> Path:
    asset = definition.asset
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assets are unavailable for the requested robot type.",
        )

    root = get_builtin_robot_assets_root().resolve() if asset.root_resolver is None else asset.root_resolver()

    requested_path = (root / asset_path).resolve()
    if not requested_path.is_relative_to(root):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access to the requested file is forbidden.")
    if not requested_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    return requested_path
