"""Discovery, installation, and uninstallation of robot catalog plugins."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Literal

from exceptions import PluginOperationError, ResourceNotFoundError, ResourceType
from robots.catalog.registry import RobotCatalogRegistry

from .manifest import PluginManifestEntry, load_plugin_manifest

if TYPE_CHECKING:
    from importlib.metadata import Distribution


@dataclass
class PluginRobot:
    """A robot type contributed by a plugin, enriched with install state."""

    type: str
    display_name: str
    role: Literal["follower", "leader"]
    installed: bool


@dataclass
class PluginInfo:
    """Full plugin status combining the manifest with runtime discovery."""

    id: str
    name: str
    description: str
    repo_url: str | None
    installed: bool
    installed_version: str | None
    robots: list[PluginRobot]


class PluginManager:
    """Manage robot catalog plugins backed by the shipped manifest."""

    def __init__(
        self,
        manifest: list[PluginManifestEntry] | None = None,
        registry: RobotCatalogRegistry | None = None,
    ) -> None:
        self._manifest = manifest if manifest is not None else load_plugin_manifest()
        self._registry = registry
        # Serializes install/uninstall so concurrent requests cannot race `uv pip`
        # against the same environment. Install/uninstall run in a worker thread
        # so a long download does not block the event loop; the lock is acquired
        # before the pre-checks so the observed installed state cannot change
        # underneath an in-flight operation.
        self._operation_lock = asyncio.Lock()

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
        """Return a single manifest plugin, raising if unknown."""
        return self._to_info(self._resolve(plugin_id), self._installed_dist(plugin_id))

    def robot_types(self, plugin_id: str) -> list[str]:
        """Return every robot type a plugin contributes (manifest plus installed catalog types)."""
        entry = self._resolve(plugin_id)
        types = [robot.type for robot in entry.robots]
        if self._installed_dist(plugin_id) is not None:
            types.extend(self.registry.robot_types_for_distribution(entry.id))
        return list(dict.fromkeys(types))

    async def install(self, plugin_id: str) -> None:
        """Install a plugin distribution into the active environment.

        Serialized per manager instance: only one install/uninstall can touch
        the environment at a time. The subprocess runs in a worker thread so a
        long download does not block the event loop. The installed state is
        re-read on the next process (``importlib.metadata`` is process-cached),
        which the restart-required flow already accounts for.
        """
        async with self._operation_lock:
            entry = self._resolve(plugin_id)
            if self._installed_dist(plugin_id) is not None:
                raise PluginOperationError(f"Plugin '{plugin_id}' is already installed.")
            await asyncio.to_thread(
                self._run,
                ["uv", "pip", "install", "--python", sys.executable, entry.install_source],
            )

    async def uninstall(self, plugin_id: str) -> None:
        """Uninstall a plugin distribution from the active environment.

        Serialized per manager instance; see ``install``.
        """
        async with self._operation_lock:
            self._resolve(plugin_id)
            if self._installed_dist(plugin_id) is None:
                raise PluginOperationError(f"Plugin '{plugin_id}' is not installed.")
            await asyncio.to_thread(
                self._run,
                ["uv", "pip", "uninstall", "--python", sys.executable, plugin_id],
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve(self, plugin_id: str) -> PluginManifestEntry:
        """Resolve a plugin id to its manifest entry, raising if unknown."""
        for entry in self._manifest:
            if entry.id == plugin_id:
                return entry
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
        known_robot_types = {robot.type for robot in robots}

        if installed:
            for robot_type in self.registry.robot_types_for_distribution(entry.id):
                definition = definitions_by_type.get(robot_type)
                if definition is None or robot_type in known_robot_types:
                    continue
                robots.append(
                    PluginRobot(
                        type=definition.type,
                        display_name=definition.display_name,
                        role=definition.role,
                        installed=True,
                    )
                )
                known_robot_types.add(robot_type)

        return PluginInfo(
            id=entry.id,
            name=entry.name,
            description=entry.description,
            repo_url=entry.repo_url,
            installed=installed,
            installed_version=dist.version if dist is not None else None,
            robots=robots,
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
