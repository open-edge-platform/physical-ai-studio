"""Plugin management API endpoints."""

from functools import lru_cache
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.dependencies import AsyncSessionDep, HealthServiceDep
from exceptions import ResourceInUseError, ResourceType
from plugins.plugin_manager import (
    PluginExtensionInfo,
    PluginInfo,
    PluginManager,
    PluginRobot,
    find_robot_types_in_use_async,
)

router = APIRouter(prefix="/api/plugins", tags=["Plugins"])


class PluginRobotResponse(BaseModel):
    type: str = Field(..., description="Stable robot type identifier")
    display_name: str = Field(..., description="Human-readable robot type label")
    role: Literal["follower", "leader"] = Field(..., description="Default robot role")
    installed: bool = Field(..., description="Whether the robot type is currently available in the catalog")


class PluginExtensionResponse(BaseModel):
    id: str = Field(..., description="Python distribution name")
    name: str = Field(..., description="Display name shown in the UI")
    description: str = Field(..., description="Short extension description")
    repo_url: str | None = Field(default=None, description="Project repository URL")
    installed: bool = Field(..., description="Whether the extension distribution is installed")
    installed_version: str | None = Field(default=None, description="Installed extension version")


class PluginResponse(BaseModel):
    id: str = Field(..., description="Python distribution name")
    name: str = Field(..., description="Display name shown in the UI")
    description: str = Field(..., description="Short plugin description")
    category: str = Field(..., description="Robot catalog category label")
    source: Literal["internal", "first_party", "external"] = Field(..., description="Plugin ownership classification")
    repo_url: str | None = Field(default=None, description="Project repository URL")
    installed: bool = Field(..., description="Whether the plugin distribution is installed")
    installed_version: str | None = Field(default=None, description="Installed plugin version")
    in_use_robot_count: int = Field(..., description="Number of persisted robots using this plugin's robot types")
    robots: list[PluginRobotResponse] = Field(..., description="Robot types contributed by the plugin")
    extensions: list[PluginExtensionResponse] = Field(
        default_factory=list,
        description="Optional add-ons gated behind this plugin being installed",
    )


class PluginOperationResponse(BaseModel):
    restart_required: bool = Field(default=True, description="A server restart is required to activate the change")


@lru_cache
def get_plugin_manager() -> PluginManager:
    """Provide a shared PluginManager instance."""
    return PluginManager()


PluginManagerDep = Annotated[PluginManager, Depends(get_plugin_manager)]


def _to_robot_response(robot: PluginRobot) -> PluginRobotResponse:
    return PluginRobotResponse(
        type=robot.type,
        display_name=robot.display_name,
        role=robot.role,
        installed=robot.installed,
    )


def _to_extension_response(extension: PluginExtensionInfo) -> PluginExtensionResponse:
    return PluginExtensionResponse(
        id=extension.id,
        name=extension.name,
        description=extension.description,
        repo_url=extension.repo_url,
        installed=extension.installed,
        installed_version=extension.installed_version,
    )


def _to_response(plugin: PluginInfo, in_use_robot_count: int) -> PluginResponse:
    return PluginResponse(
        id=plugin.id,
        name=plugin.name,
        description=plugin.description,
        category=plugin.category,
        source=plugin.source,
        repo_url=plugin.repo_url,
        installed=plugin.installed,
        installed_version=plugin.installed_version,
        in_use_robot_count=in_use_robot_count,
        robots=[_to_robot_response(robot) for robot in plugin.robots],
        extensions=[_to_extension_response(extension) for extension in plugin.extensions],
    )


@router.get("")
async def list_plugins(
    plugin_manager: PluginManagerDep,
    session: AsyncSessionDep,
) -> list[PluginResponse]:
    """List available and installed plugins with their robot types."""
    plugins = plugin_manager.list_plugins()
    robot_types_by_plugin = {plugin.id: [robot.type for robot in plugin.robots] for plugin in plugins}
    all_robot_types = sorted({type_ for types in robot_types_by_plugin.values() for type_ in types})
    in_use_robot_types = set(await find_robot_types_in_use_async(session, all_robot_types))

    return [
        _to_response(
            plugin,
            in_use_robot_count=sum(1 for type_ in robot_types_by_plugin[plugin.id] if type_ in in_use_robot_types),
        )
        for plugin in plugins
    ]


@router.post("/{plugin_id}/install")
async def install_plugin(
    plugin_id: str,
    plugin_manager: PluginManagerDep,
    health_service: HealthServiceDep,
) -> PluginOperationResponse:
    """Install a plugin distribution and require a server restart to activate."""
    plugin_manager.install(plugin_id)
    health_service.mark_plugin_restart_required()
    return PluginOperationResponse()


@router.post("/{plugin_id}/uninstall")
async def uninstall_plugin(
    plugin_id: str,
    plugin_manager: PluginManagerDep,
    session: AsyncSessionDep,
    health_service: HealthServiceDep,
) -> PluginOperationResponse:
    """Uninstall a plugin distribution after checking no robots reference its types."""
    in_use_robot_types = await find_robot_types_in_use_async(session, plugin_manager.robot_types(plugin_id))
    if in_use_robot_types:
        plugin = plugin_manager.get(plugin_id)
        raise ResourceInUseError(
            ResourceType.PLUGIN,
            plugin_id,
            message=(
                f"Cannot uninstall '{plugin.name}': {len(in_use_robot_types)} robot(s) use type(s) "
                f"{', '.join(in_use_robot_types)}. Delete those robots first."
            ),
        )
    plugin_manager.uninstall(plugin_id)
    health_service.mark_plugin_restart_required()
    return PluginOperationResponse()
