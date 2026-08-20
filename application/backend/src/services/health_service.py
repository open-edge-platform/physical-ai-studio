"""Process-local backend health and restart state."""

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class HealthService:
    """Expose the current server instance and pending plugin restart state."""

    instance_id: str = field(default_factory=lambda: str(uuid4()))
    plugin_restart_required: bool = False

    def mark_plugin_restart_required(self) -> None:
        """Record that installed plugin changes require process replacement."""
        self.plugin_restart_required = True
