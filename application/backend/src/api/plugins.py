"""Plugin management API endpoints."""

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.dependencies import HealthServiceDep
from plugins.plugin_manager import PluginInfo, PluginManager

router = APIRouter(prefix="/api/plugins", tags=["Plugins"])


class PluginResponse(BaseModel):
    id: str = Field(..., description="Python distribution name")
    name: str = Field(..., description="Display name shown in the UI")
    description: str = Field(..., description="Short plugin description")
    repo_url: str | None = Field(default=None, description="Project repository URL")
    installed: bool = Field(..., description="Whether the plugin distribution is installed")
    installed_version: str | None = Field(default=None, description="Installed plugin version")
    robot_count: int = Field(..., description="Number of robot types contributed by the plugin")


class PluginOperationResponse(BaseModel):
    restart_required: bool = Field(default=True, description="A server restart is required to activate the change")


class InstallPluginRequest(BaseModel):
    plugin_id: str = Field(..., description="Python distribution name of the plugin to install")


@lru_cache
def get_plugin_manager() -> PluginManager:
    """Provide a shared PluginManager instance."""
    return PluginManager()


PluginManagerDep = Annotated[PluginManager, Depends(get_plugin_manager)]


def _to_response(plugin: PluginInfo) -> PluginResponse:
    return PluginResponse(
        id=plugin.id,
        name=plugin.name,
        description=plugin.description,
        repo_url=plugin.repo_url,
        installed=plugin.installed,
        installed_version=plugin.installed_version,
        robot_count=len(plugin.robots),
    )


@router.get("")
async def list_plugins(
    plugin_manager: PluginManagerDep,
) -> list[PluginResponse]:
    """List available and installed plugins with their robot types."""
    plugins = plugin_manager.list_plugins()
    return [_to_response(plugin) for plugin in plugins]


@router.post("")
async def install_plugin(
    request: InstallPluginRequest,
    plugin_manager: PluginManagerDep,
    health_service: HealthServiceDep,
) -> PluginOperationResponse:
    """Install a plugin distribution and require a server restart to activate."""
    await plugin_manager.install(request.plugin_id)
    health_service.mark_plugin_restart_required()
    return PluginOperationResponse()


@router.delete("/{plugin_id}")
async def uninstall_plugin(
    plugin_id: str,
    plugin_manager: PluginManagerDep,
    health_service: HealthServiceDep,
) -> PluginOperationResponse:
    """Uninstall a plugin distribution and require a server restart to activate the change."""
    await plugin_manager.uninstall(plugin_id)
    health_service.mark_plugin_restart_required()
    return PluginOperationResponse()
