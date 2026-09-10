"""Plugin manifest schema and loader.

The manifest is a curated list of known robot catalog plugins shipped with
Studio. Each entry describes a plugin distribution, where it is installed
from, and the robot types it contributes. ``PluginManager`` merges the
manifest with the runtime entry-point discovery to report available versus
installed plugins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"


class ManifestRobot(BaseModel):
    """A robot type contributed by a plugin, as declared in the manifest."""

    type: str = Field(..., description="Stable robot type identifier")
    display_name: str = Field(..., description="Human-readable robot type label")
    role: Literal["follower", "leader"] = Field(..., description="Default robot role")


class PluginManifestEntry(BaseModel):
    """A known plugin distribution and the robots it contributes."""

    id: str = Field(..., description="Python distribution name (PyPI normalised)")
    name: str = Field(..., description="Display name shown in the UI")
    description: str = Field(..., description="Short plugin description")
    repo_url: str | None = Field(default=None, description="Project repository URL")
    install_source: str = Field(..., description="Install spec passed to `uv pip install`")
    robots: list[ManifestRobot] = Field(default_factory=list, description="Robot types contributed by the plugin")


def load_plugin_manifest(path: Path = MANIFEST_PATH) -> list[PluginManifestEntry]:
    """Load and validate the plugin manifest file."""
    with path.open(encoding="utf-8") as manifest_file:
        data = json.load(manifest_file)
    return [PluginManifestEntry.model_validate(entry) for entry in data]
