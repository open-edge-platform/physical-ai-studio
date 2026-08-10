from __future__ import annotations

from pathlib import Path


def resolve_serial_device(device: str) -> str:
    """Prefer the stable serial by-id alias for a robot device."""
    return _stable_device_path(device, Path("/dev/serial/by-id"))


def resolve_camera_device(device: str) -> str:
    """Prefer the stable v4l by-id alias for a camera device."""
    return _stable_device_path(device, Path("/dev/v4l/by-id"))


def _stable_device_path(device: str, directory: Path) -> str:
    try:
        target = Path(device).resolve()
        for candidate in directory.iterdir():
            if candidate.resolve() == target:
                return str(candidate)
    except OSError:
        pass
    return device
