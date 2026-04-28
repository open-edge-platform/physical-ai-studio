# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serializable camera construction spec for transport endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from physicalai.capture.camera import Camera


@dataclass(frozen=True)
class CameraSpec:
    """Config payload describing how to construct a camera instance.

    Attributes:
        camera_type: Logical camera type passed to :func:`create_camera`.
        camera_kwargs: Keyword arguments forwarded to the camera constructor.
    """

    camera_type: str
    camera_kwargs: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # JSON serialization (used by subprocess publisher protocol)
    # ------------------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            Dictionary with ``camera_type`` and ``camera_kwargs`` keys.
        """
        return {"camera_type": self.camera_type, "camera_kwargs": self.camera_kwargs}

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> CameraSpec:
        """Deserialize from a JSON dictionary.

        Args:
            data: Dictionary with ``camera_type`` and optional
                ``camera_kwargs`` keys.

        Returns:
            A new :class:`CameraSpec` instance.
        """
        return cls(
            camera_type=data["camera_type"],
            camera_kwargs=data.get("camera_kwargs", {}),
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def build(self) -> Camera:
        """Instantiate the camera described by this spec.

        Returns:
            A new camera instance configured from ``camera_type`` and
            ``camera_kwargs``.
        """
        from physicalai.capture.factory import create_camera  # noqa: PLC0415

        return create_camera(self.camera_type, **self.camera_kwargs)
