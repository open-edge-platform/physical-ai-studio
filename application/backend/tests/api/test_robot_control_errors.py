from api.robot_control import _websocket_error_payload
from exceptions import RobotDeviceAlreadyOwnedError


def test_websocket_error_payload_from_app_exception():
    payload = _websocket_error_payload(RobotDeviceAlreadyOwnedError())
    assert payload["event"] == "error"
    assert payload["error_code"] == "robot_device_already_owned"
    assert "already in use" in payload["message"].lower()


def test_websocket_error_payload_from_generic_exception():
    payload = _websocket_error_payload(RuntimeError("boom"))
    assert payload["event"] == "error"
    assert payload["error_code"] == "robot_connection_failed"
    assert payload["message"] == "boom"
