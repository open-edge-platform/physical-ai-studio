from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from exceptions import CameraSettingsConflictError
from services.camera_claims import CameraClaim, CameraClaimRegistry


def _claim(
    fingerprint: dict[str, Any],
    settings: tuple[int, int, int],
    *,
    holder: str = "rt-a",
    project_name: str = "Alpha",
) -> CameraClaim:
    return CameraClaim(
        fingerprint=fingerprint,
        settings=settings,
        holder=holder,
        project_id=uuid4(),
        project_name=project_name,
    )


def test_identical_settings_share() -> None:
    registry = CameraClaimRegistry()
    first = _claim({"serial": "cam"}, (640, 480, 30), holder="rt-a", project_name="Alpha")
    second = _claim({"serial": "cam"}, (640, 480, 30), holder="rt-b", project_name="Beta")

    registry.claim([first])
    registry.claim([second])

    assert registry.holder_of({"serial": "cam"}) is not None


def test_fingerprint_key_order_does_not_create_a_second_claim() -> None:
    registry = CameraClaimRegistry()
    registry.claim([_claim({"serial": "cam", "bus": "usb-1"}, (640, 480, 30))])

    with pytest.raises(CameraSettingsConflictError):
        registry.claim([_claim({"bus": "usb-1", "serial": "cam"}, (1280, 720, 30), holder="rt-b")])


def test_different_settings_are_rejected_naming_the_holder() -> None:
    registry = CameraClaimRegistry()
    registry.claim([_claim({"serial": "cam"}, (640, 480, 30), project_name="Alpha")])

    with pytest.raises(CameraSettingsConflictError, match="Alpha") as exc_info:
        registry.claim([_claim({"serial": "cam"}, (1280, 720, 30), holder="rt-b", project_name="Beta")])

    assert "640x480@30" in exc_info.value.message
    assert "1280x720@30" in exc_info.value.message


def test_a_partial_claim_is_rolled_back() -> None:
    registry = CameraClaimRegistry()
    registry.claim([_claim({"serial": "cam-a"}, (640, 480, 30), holder="rt-a")])

    with pytest.raises(CameraSettingsConflictError):
        registry.claim(
            [
                _claim({"serial": "cam-b"}, (640, 480, 30), holder="rt-b"),
                _claim({"serial": "cam-c"}, (640, 480, 30), holder="rt-b"),
                _claim({"serial": "cam-a"}, (1280, 720, 30), holder="rt-b"),
            ]
        )

    assert registry.holder_of({"serial": "cam-b"}) is None
    assert registry.holder_of({"serial": "cam-c"}) is None
    assert registry.holder_of({"serial": "cam-a"}) is not None
    assert registry.holder_of({"serial": "cam-a"}).holder == "rt-a"


def test_release_allows_new_settings() -> None:
    registry = CameraClaimRegistry()
    registry.claim([_claim({"serial": "cam"}, (640, 480, 30), holder="rt-a")])
    registry.release("rt-a")
    registry.claim([_claim({"serial": "cam"}, (1280, 720, 30), holder="rt-b")])

    held = registry.holder_of({"serial": "cam"})
    assert held is not None
    assert held.settings == (1280, 720, 30)


def test_a_stale_generation_does_not_release() -> None:
    registry = CameraClaimRegistry()
    first = registry.claim([_claim({"serial": "cam"}, (640, 480, 30), holder="rt-a")])
    second = registry.claim([_claim({"serial": "cam"}, (640, 480, 30), holder="rt-a")])

    registry.release("rt-a", generation=first)

    assert first != second
    assert registry.holder_of({"serial": "cam"}) is not None
    registry.release("rt-a", generation=second)
    assert registry.holder_of({"serial": "cam"}) is None
