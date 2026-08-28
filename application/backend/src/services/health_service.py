"""Process-local backend health and restart state."""

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from uuid import uuid4

CATALOG_PLUGIN_ENTRYPOINT_GROUP = "physicalai.studio.catalog_plugins"


def _installed_catalog_plugins() -> set[tuple[str, str, str]]:
    """Return the installed catalog plugin entry points and distribution versions."""
    plugins: set[tuple[str, str, str]] = set()
    for plugin in entry_points(group=CATALOG_PLUGIN_ENTRYPOINT_GROUP):
        distribution = plugin.dist
        distribution_version = distribution.version if distribution is not None else "unknown"
        plugins.add((plugin.name, plugin.value, distribution_version))
    return plugins


@dataclass
class HealthService:
    """Expose the current server instance and pending plugin restart state."""

    instance_id: str = field(default_factory=lambda: str(uuid4()))
    plugin_restart_required: bool = False
    _installed_catalog_plugins: set[tuple[str, str, str]] = field(default_factory=_installed_catalog_plugins)

    def refresh_plugin_restart_required(self) -> None:
        """Record catalog plugin installation or version changes after startup."""
        if _installed_catalog_plugins() != self._installed_catalog_plugins:
            self.mark_plugin_restart_required()

    def mark_plugin_restart_required(self) -> None:
        """Record that installed plugin changes require process replacement."""
        self.plugin_restart_required = True
