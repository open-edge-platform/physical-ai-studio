"""Factory for building camera instances from backend camera configs.

Maps backend driver names to physicalai.capture Config recipes and
filters per-driver kwargs so that only constructor-safe parameters reach the camera.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from loguru import logger
from physicalai.capture import CameraType, ColorMode, SharedCamera
from physicalai.config import Config

if TYPE_CHECKING:
    from schemas.project_camera import Camera

MIGRATED_DRIVERS: frozenset[str] = frozenset({"usb_camera", "realsense", "basler", "ipcam"})

DRIVER_KEY_MAP: dict[str, str] = {
    "uvc": "usb_camera",
    "realsense": "realsense",
    "basler": "basler",
    "ipcam": "ipcam",
}

_DRIVER_TO_CAMERA_TYPE: dict[str, CameraType] = {
    "usb_camera": CameraType.UVC,
    "realsense": CameraType.REALSENSE,
    "basler": CameraType.BASLER,
    "ipcam": CameraType.IP,
}

_DRIVER_TO_CLASS_PATH: dict[str, str] = {
    "usb_camera": "physicalai.capture.UVCCamera",
    "realsense": "physicalai.capture.RealSenseCamera",
    "basler": "physicalai.capture.BaslerCamera",
    "ipcam": "physicalai.capture.IPCamera",
}

# Per-driver kwargs that are safe to pass through to the nested camera recipe.
_ALLOWED_KWARGS: dict[str, frozenset[str]] = {
    "usb_camera": frozenset({"width", "height", "fps"}),
    "realsense": frozenset({"width", "height", "fps"}),
    "basler": frozenset({"width", "height", "fps"}),
    "ipcam": frozenset({"width", "height", "fps", "url"}),
}


def _get_fingerprint_dict(fingerprint_str: str) -> dict[str, Any]:
    try:
        d = json.loads(fingerprint_str)
        if isinstance(d, dict):
            return d
    except ValueError:  # This is not a valid fingerprint
        pass
    logger.error(f"Could not convert `{fingerprint_str}` into a valid fingerprint dict")
    raise ValueError("Unable to parse fingerprint into a dict")


def build_camera_config(config: Camera) -> Config:
    """Build a camera configuration from a config schema."""
    class_path = _DRIVER_TO_CLASS_PATH[config.driver]
    allowed = _ALLOWED_KWARGS.get(config.driver, frozenset())

    payload = config.payload.model_dump()
    init_args: dict[str, Any] = {k: v for k, v in payload.items() if k in allowed and v is not None}

    fingerprint = _get_fingerprint_dict(config.fingerprint)

    if config.driver == "usb_camera":
        init_args["device"] = fingerprint
    elif config.driver != "ipcam":
        init_args["serial_number"] = fingerprint["serial"]

    return Config(class_path, init_args)


def is_migrated(driver: str) -> bool:
    """Return True if *driver* is supported by physicalai.capture."""
    return driver in MIGRATED_DRIVERS


def driver_to_camera_type(driver: str) -> CameraType:
    """Convert a backend driver name to a CameraType enum value."""
    try:
        return _DRIVER_TO_CAMERA_TYPE[driver]
    except KeyError:
        msg = f"unsupported driver {driver!r}; expected one of {sorted(MIGRATED_DRIVERS)}"
        raise ValueError(msg) from None


def build_shared_camera(
    config: Camera,
    *,
    validate_on_connect: bool = False,
    overwrite_settings: bool = False,
    idle_timeout: float = 5.0,
) -> SharedCamera:
    """Build a SharedCamera from a backend Camera schema.

    Args:
        config: Backend camera configuration (discriminated union).
        validate_on_connect: If ``True``, :meth:`~SharedCamera.connect` raises
            :class:`~physicalai.capture.errors.CaptureError` when an
            existing publisher's resolution does not match the requested
            ``width``/``height``. Use ``False`` for preview streams
            (tolerates mismatch) and ``True`` for recording / inference
            when the initial attachment must match the requested config.
        overwrite_settings: If ``True``, attempt to reconfigure the publisher
            to match requested settings when a config mismatch is detected.
            Requires a publisher that supports the control channel (v2+).
        idle_timeout: Seconds with zero subscribers before the publisher
            self-exits.  Preview-class callers should use a short value
            (e.g. 0.5) for fast turnover on resolution changes;
            record-class callers should use a longer value (e.g. 5.0).

    Returns:
        A configured (but not yet connected) SharedCamera instance.
    """
    if config.driver not in _DRIVER_TO_CLASS_PATH:
        msg = f"unsupported driver {config.driver!r}; expected one of {sorted(MIGRATED_DRIVERS)}"
        raise ValueError(msg)

    camera_config = build_camera_config(config)
    logger.debug(f"camera config for {config.name}: {camera_config}")

    return SharedCamera(
        camera=camera_config,
        color_mode=ColorMode.RGB,
        validate_on_connect=validate_on_connect,
        overwrite_settings=overwrite_settings,
        idle_timeout=idle_timeout,
    )
