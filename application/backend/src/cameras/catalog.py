"""Local camera contracts, to be promoted to the shared plugin SDK."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    import builtins
    from collections.abc import Awaitable, Callable

    from physicalai.capture import Camera


class CameraClaimPolicy(StrEnum):
    """Ownership of a camera's physical source."""

    SHARED_MATCHING_SETTINGS = "shared_matching_settings"
    EXCLUSIVE = "exclusive"
    SHARED_SOURCE_MULTI_VIEW = "shared_source_multi_view"


@dataclass(frozen=True)
class CameraDeviceInfo:
    """Stable identity and display name of a discovered camera."""

    fingerprint: dict[str, Any]
    hardware_name: str


@dataclass(frozen=True)
class CameraFormat:
    """One supported stream profile."""

    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class CatalogCamera[PayloadT: BaseModel]:
    """Validated descriptor passed to a builder, independent of Studio's API models."""

    type: str
    fingerprint: dict[str, Any]
    payload: PayloadT


class CatalogCameraFactory(Protocol):
    """Reserved for application services needed by camera builders."""


class CameraProbe[PayloadT: BaseModel](Protocol):
    """Hardware queries; implementations keep blocking SDK work off the event loop."""

    async def discover(self, *, only_usable: bool = True) -> list[CameraDeviceInfo]:
        """Discover cameras; false also includes devices currently in use."""
        ...

    async def query_formats(self, fingerprint: dict[str, Any], payload: PayloadT | None = None) -> list[CameraFormat]:
        """Query profiles without starting a capture stream."""
        ...

    async def is_online(self, fingerprint: dict[str, Any], payload: PayloadT) -> bool:
        """Report whether the selected source is attached."""
        ...


@dataclass(frozen=True)
class CameraCatalogDefinition[PayloadT: BaseModel]:
    """A registered camera type and its construction/schema contract.

    Builders return unconnected, export-config-decorated Runtime cameras. Studio
    wraps their recipes in SharedCamera and owns connection and claim lifetimes.
    """

    type: str
    display_name: str
    camera_payload: builtins.type[PayloadT]
    camera_builder: Callable[[CatalogCamera[PayloadT], CatalogCameraFactory], Awaitable[Camera]] | None = None
    probe: CameraProbe[PayloadT] | None = None
    claim_policy: CameraClaimPolicy = CameraClaimPolicy.SHARED_MATCHING_SETTINGS
