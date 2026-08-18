from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from exceptions import CameraSettingsConflictError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from schemas.project_camera import Camera


@dataclass(frozen=True, slots=True)
class CameraClaim:
    fingerprint: str
    settings: tuple[int, int, int]
    holder: str
    project_id: UUID
    project_name: str


def settings_from_camera(camera: Camera) -> tuple[int, int, int]:
    """Return (width, height, fps) from a camera row, using 0 for unset values."""
    payload = camera.payload
    return (
        int(getattr(payload, "width", None) or 0),
        int(getattr(payload, "height", None) or 0),
        int(getattr(payload, "fps", None) or 0),
    )


class CameraClaimRegistry:
    """Pin camera settings for the life of an API process.

    In-memory is acceptable here, unlike the robot guard. A camera claim
    protects against a concurrent misconfiguration inside one API process. A
    robot guard has to survive an API restart because a detached session keeps
    driving the arm — which is why that one reads the on-disk lock file.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._claims: dict[str, dict[str, CameraClaim]] = {}

    def claim(self, claims: Sequence[CameraClaim]) -> None:
        """Pin every camera in ``claims``, or pin none of them."""
        with self._lock:
            for incoming in claims:
                holders = self._claims.get(incoming.fingerprint)
                if not holders:
                    continue
                pinned = next(iter(holders.values()))
                if pinned.settings != incoming.settings:
                    raise CameraSettingsConflictError(
                        project_name=pinned.project_name,
                        pinned=pinned.settings,
                        requested=incoming.settings,
                    )
            for incoming in claims:
                holders = self._claims.setdefault(incoming.fingerprint, {})
                holders[incoming.holder] = incoming

    def release(self, holder: str) -> None:
        with self._lock:
            empty: list[str] = []
            for fingerprint, holders in self._claims.items():
                holders.pop(holder, None)
                if not holders:
                    empty.append(fingerprint)
            for fingerprint in empty:
                del self._claims[fingerprint]

    def holder_of(self, fingerprint: str) -> CameraClaim | None:
        with self._lock:
            holders = self._claims.get(fingerprint)
            if not holders:
                return None
            return next(iter(holders.values()))
