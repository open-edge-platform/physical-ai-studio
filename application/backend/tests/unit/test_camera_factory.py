"""Camera recipes preserve explicitly requested RealSense depth streaming."""

from unittest.mock import patch

import pytest

from schemas.project_camera import CameraAdapter
from utils.camera_factory import build_camera_config, build_shared_camera


@pytest.mark.parametrize("output_type,enable_depth", [(None, False), ("color", False), ("depth", True), ("both", True)])
def test_realsense_depth_streaming_follows_requested_output(output_type: str | None, enable_depth: bool) -> None:
    payload: dict[str, int | str] = {"width": 640, "height": 480, "fps": 30}
    if output_type is not None:
        payload["output_type"] = output_type
    camera = CameraAdapter.validate_python(
        {
            "driver": "realsense",
            "name": "wrist",
            "fingerprint": {"serial": "test-serial"},
            "hardware_name": None,
            "payload": payload,
        }
    )

    recipe = build_camera_config(camera)
    assert recipe.class_path == "physicalai.capture.RealSenseCamera"
    assert recipe.init_args == {
        "serial_number": "test-serial",
        "width": 640,
        "height": 480,
        "fps": 30,
        "enable_depth": enable_depth,
    }

    with patch("utils.camera_factory.SharedCamera") as shared:
        build_shared_camera(camera)
    assert shared.call_args.kwargs["camera"] == recipe


def test_uvc_recipe_does_not_receive_a_depth_option() -> None:
    camera = CameraAdapter.validate_python(
        {
            "driver": "usb_camera",
            "name": "front",
            "fingerprint": {"serial": "test-serial"},
            "hardware_name": None,
            "payload": {"width": 640, "height": 480, "fps": 30},
        }
    )
    assert "enable_depth" not in build_camera_config(camera).init_args
