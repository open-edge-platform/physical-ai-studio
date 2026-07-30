import pytest
from physicalai.robot import (
    RobotDeviceAlreadyOwned,
    RobotError,
    RobotNameConflict,
    RobotProtocolMismatch,
    RobotTransportError,
)

from exceptions import (
    RobotDeviceAlreadyOwnedError,
    RobotNameConflictError,
    RobotProtocolMismatchError,
    SharedRobotTransportError,
)
from robots.shared_robot_errors import translate_robot_error


class TestTranslateRobotError:
    def test_device_already_owned_includes_device_id(self):
        result = translate_robot_error(
            RobotDeviceAlreadyOwned("ignored", phase="device_lock_contention", device_ids=("serial:ttyACM0",))
        )
        assert isinstance(result, RobotDeviceAlreadyOwnedError)
        assert result.error_code == "robot_device_already_owned"
        assert "serial:ttyACM0" in result.message

    def test_device_already_owned_includes_multiple_device_ids(self):
        result = translate_robot_error(
            RobotDeviceAlreadyOwned(
                "ignored",
                phase="device_lock_contention",
                device_ids=("serial:ttyACM0", "serial:ttyACM1"),
            )
        )
        assert isinstance(result, RobotDeviceAlreadyOwnedError)
        assert "serial:ttyACM0" in result.message
        assert "serial:ttyACM1" in result.message

    def test_device_already_owned_uses_default_without_device_ids(self):
        result = translate_robot_error(RobotDeviceAlreadyOwned(""))
        assert isinstance(result, RobotDeviceAlreadyOwnedError)
        assert "already in use" in result.message.lower()

    def test_device_already_owned_ignores_worker_traceback(self):
        verbose = (
            "failed to start robot owner: device 'serial:ttyACM0' is already locked\n"
            "--- worker traceback ---\nTraceback (most recent call last):\n  ..."
        )
        result = translate_robot_error(
            RobotDeviceAlreadyOwned(verbose, phase="device_lock_contention", device_ids=("serial:ttyACM0",))
        )
        assert isinstance(result, RobotDeviceAlreadyOwnedError)
        assert "traceback" not in result.message.lower()

    def test_name_conflict_uses_robot_name_from_context(self):
        result = translate_robot_error(RobotNameConflict("ignored"), robot_name="lab_arm")
        assert isinstance(result, RobotNameConflictError)
        assert result.error_code == "robot_name_conflict"
        assert "lab_arm" in result.message

    def test_name_conflict_uses_default_without_robot_name(self):
        result = translate_robot_error(RobotNameConflict("name taken"))
        assert isinstance(result, RobotNameConflictError)
        assert "this robot is" in result.message.lower()

    def test_protocol_mismatch_uses_default_message(self):
        # physicalai does not expose protocol versions as structured fields.
        result = translate_robot_error(
            RobotProtocolMismatch("owner of 'lab_arm' speaks protocol_version=2, this SharedRobot supports 1")
        )
        assert isinstance(result, RobotProtocolMismatchError)
        assert result.error_code == "robot_protocol_mismatch"
        assert "incompatible software version" in result.message.lower()
        assert "protocol_version" not in result.message

    def test_transport_error_uses_default_message(self):
        result = translate_robot_error(RobotTransportError("spawn failed", phase="connection_failed"))
        assert isinstance(result, SharedRobotTransportError)
        assert result.error_code == "robot_transport_error"
        assert result.message == "Could not connect to the robot. Check the connection and try again."

    def test_generic_robot_error_uses_default_message(self):
        result = translate_robot_error(RobotError("unexpected"))
        assert isinstance(result, SharedRobotTransportError)
        assert result.message == "Could not connect to the robot. Check the connection and try again."

    def test_passthrough_non_robot_error(self):
        original = ValueError("not a robot error")
        assert translate_robot_error(original) is original

    def test_passthrough_already_translated(self):
        original = RobotDeviceAlreadyOwnedError(device_ids=("serial:ttyACM0",))
        assert translate_robot_error(original) is original


@pytest.mark.parametrize(
    ("exc", "kwargs", "error_code"),
    [
        (RobotDeviceAlreadyOwned("owned", device_ids=("serial:ttyACM0",)), {}, "robot_device_already_owned"),
        (RobotNameConflict("conflict"), {"robot_name": "arm"}, "robot_name_conflict"),
        (RobotProtocolMismatch("mismatch"), {}, "robot_protocol_mismatch"),
        (RobotTransportError("transport"), {}, "robot_transport_error"),
    ],
)
def test_translate_error_codes(exc: Exception, kwargs: dict, error_code: str):
    result = translate_robot_error(exc, **kwargs)
    assert getattr(result, "error_code") == error_code
