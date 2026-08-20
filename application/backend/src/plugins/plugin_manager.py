"""Discovery, installation, and uninstallation of robot catalog plugins."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Literal

from exceptions import PluginOperationError, ResourceNotFoundError, ResourceType
from robots.catalog.registry import RobotCatalogRegistry

from .manifest import PluginExtensionEntry, PluginManifestEntry, PluginSource, load_plugin_manifest

if TYPE_CHECKING:
    from importlib.metadata import Distribution

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm.session import Session


@dataclass
class PluginRobot:
    """A robot type contributed by a plugin, enriched with install state."""

    type: str
    display_name: str
    role: Literal["follower", "leader"]
    installed: bool


@dataclass
class PluginExtensionInfo:
    """An optional add-on gated behind its parent plugin, with install state."""

    id: str
    name: str
    description: str
    repo_url: str | None
    installed: bool
    installed_version: str | None


@dataclass
class PluginInfo:
    """Full plugin status combining the manifest with runtime discovery."""

    id: str
    name: str
    description: str
    category: str
    source: PluginSource
    repo_url: str | None
    installed: bool
    installed_version: str | None
    robots: list[PluginRobot]
    extensions: list[PluginExtensionInfo]


class PluginManager:
    """Manage robot catalog plugins backed by the shipped manifest."""

    def __init__(
        self,
        manifest: list[PluginManifestEntry] | None = None,
        registry: RobotCatalogRegistry | None = None,
    ) -> None:
        self._manifest = manifest if manifest is not None else load_plugin_manifest()
        self._registry = registry

    @property
    def registry(self) -> RobotCatalogRegistry:
        """Return the catalog registry, discovering plugin entry points once."""
        if self._registry is None:
            self._registry = RobotCatalogRegistry()
        return self._registry

    def list_plugins(self) -> list[PluginInfo]:
        """Return manifest plugins merged with installed distribution state."""
        return [self._to_info(entry, self._installed_dist(entry.id)) for entry in self._manifest]

    def get(self, plugin_id: str) -> PluginInfo:
        """Return a single manifest plugin or extension, raising if unknown."""
        for entry in self._manifest:
            if entry.id == plugin_id:
                return self._to_info(entry, self._installed_dist(plugin_id))
            for extension in entry.extensions:
                if extension.id == plugin_id:
                    return self._to_extension_info(extension)
        raise ResourceNotFoundError(ResourceType.PLUGIN, plugin_id)

    def robot_types(self, plugin_id: str) -> list[str]:
        """Return every robot type a plugin contributes (manifest plus installed catalog types)."""
        entry, extension = self._resolve(plugin_id)
        if extension is not None:
            return []
        types = [robot.type for robot in entry.robots]
        if self._installed_dist(plugin_id) is not None:
            types.extend(self.registry.robot_types_for_distribution(entry.id))
        return types

    def install(self, plugin_id: str) -> None:
        """Install a plugin or extension distribution into the active environment."""
        entry, extension = self._resolve(plugin_id)
        if extension is not None and self._installed_dist(entry.id) is None:
            raise PluginOperationError(
                f"Install '{entry.name}' first before installing the extension '{extension.name}'."
            )
        if self._installed_dist(plugin_id) is not None:
            raise PluginOperationError(f"Plugin '{plugin_id}' is already installed.")
        install_source = extension.install_source if extension is not None else entry.install_source
        self._run(["uv", "pip", "install", "--python", sys.executable, install_source])

    def uninstall(self, plugin_id: str) -> None:
        """Uninstall a plugin or extension distribution from the active environment."""
        self._resolve(plugin_id)
        if self._installed_dist(plugin_id) is None:
            raise PluginOperationError(f"Plugin '{plugin_id}' is not installed.")
        self._run(["uv", "pip", "uninstall", "--python", sys.executable, plugin_id])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve(self, plugin_id: str) -> tuple[PluginManifestEntry, PluginExtensionEntry | None]:
        """Resolve a plugin id to its manifest entry and, if applicable, its extension."""
        for entry in self._manifest:
            if entry.id == plugin_id:
                return entry, None
            for extension in entry.extensions:
                if extension.id == plugin_id:
                    return entry, extension
        raise ResourceNotFoundError(ResourceType.PLUGIN, plugin_id)

    @staticmethod
    def _installed_dist(plugin_id: str) -> Distribution | None:
        try:
            return metadata.distribution(plugin_id)
        except metadata.PackageNotFoundError:
            return None

    def _to_info(self, entry: PluginManifestEntry, dist: Distribution | None) -> PluginInfo:
        installed = dist is not None
        definitions = self.registry.list_definitions()
        definitions_by_type = {definition.type: definition for definition in definitions}

        robots = [
            PluginRobot(
                type=robot.type,
                display_name=robot.display_name,
                role=robot.role,
                installed=robot.type in definitions_by_type,
            )
            for robot in entry.robots
        ]

        if installed:
            for robot_type in self.registry.robot_types_for_distribution(entry.id):
                definition = definitions_by_type.get(robot_type)
                if definition is None:
                    continue
                robots.append(
                    PluginRobot(
                        type=definition.type,
                        display_name=definition.display_name,
                        role=definition.role,
                        installed=True,
                    )
                )

        return PluginInfo(
            id=entry.id,
            name=entry.name,
            description=entry.description,
            category=entry.category,
            source=entry.source,
            repo_url=entry.repo_url,
            installed=installed,
            installed_version=dist.version if dist is not None else None,
            robots=robots,
            extensions=[self._to_extension_info(extension) for extension in entry.extensions],
        )

    def _to_extension_info(self, extension: PluginExtensionEntry) -> PluginExtensionInfo:
        dist = self._installed_dist(extension.id)
        return PluginExtensionInfo(
            id=extension.id,
            name=extension.name,
            description=extension.description,
            repo_url=extension.repo_url,
            installed=dist is not None,
            installed_version=dist.version if dist is not None else None,
        )

    @staticmethod
    def _run(command: list[str]) -> None:
        """Run a subprocess, raising a user-facing error on failure."""
        command_preview = " ".join(command)
        try:
            # Command is assembled from the curated manifest and the active interpreter, not user input.
            result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)  # noqa: S603
        except (subprocess.SubprocessError, OSError) as error:
            raise PluginOperationError(f"Failed to run `{command_preview}`: {error}") from error
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise PluginOperationError(f"`{command_preview}` failed: {detail}")


def find_robot_types_in_use_sync(session: Session, robot_types: list[str]) -> list[str]:
    """Return which of the given robot types are persisted across projects."""
    from sqlalchemy import select

    from db.schema import ProjectRobotDB

    if not robot_types:
        return []
    rows = session.execute(select(ProjectRobotDB.type).where(ProjectRobotDB.type.in_(robot_types))).scalars().all()
    return sorted(set(rows))


async def find_robot_types_in_use_async(session: AsyncSession, robot_types: list[str]) -> list[str]:
    """Return which of the given robot types are persisted across projects."""
    from sqlalchemy import select

    from db.schema import ProjectRobotDB

    if not robot_types:
        return []
    rows = await session.execute(select(ProjectRobotDB.type).where(ProjectRobotDB.type.in_(robot_types)))
    return sorted(set(rows.scalars().all()))
