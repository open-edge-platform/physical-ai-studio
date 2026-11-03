from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException, status

if TYPE_CHECKING:
    from physicalai_studio_plugin import RobotCatalogDefinition

# Kept as a fallback for legacy third-party catalog definitions that do not
# provide an asset root resolver. Studio no longer ships built-in robot assets.
BUILTIN_ROBOT_ASSETS_ROOT = Path(__file__).resolve().parents[2] / "static" / "robot-assets"


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

    # Mesh URLs are resolved from their declared URDF package root. This is
    # usually the top-level directory named by the URDF path, but plugins can
    # resolve package roots inside their installed distributions.
    package_roots = tuple(definition.asset.packages.values())
    if len(package_roots) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Robot assets declare multiple package roots, which this endpoint does not support.",
        )
    asset_root = package_roots[0] if package_roots else definition.asset.urdf_relative_path.parts[0]
    return _resolve_robot_path(asset_path=asset_root / asset_path, definition=definition)


def _resolve_robot_path(asset_path: Path, definition: RobotCatalogDefinition) -> Path:
    asset = definition.asset
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assets are unavailable for the requested robot type.",
        )

    root = BUILTIN_ROBOT_ASSETS_ROOT.resolve() if asset.root_resolver is None else asset.root_resolver()

    requested_path = (root / asset_path).resolve()
    if not requested_path.is_relative_to(root):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access to the requested file is forbidden.")
    if not requested_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    return requested_path
