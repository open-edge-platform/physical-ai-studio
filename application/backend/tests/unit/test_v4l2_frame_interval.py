import ctypes
import errno
import importlib
import sys
from unittest.mock import patch

import pytest

from utils import v4l2_frame_interval as v4l2
from utils.v4l2_frame_interval import (
    VIDIOC_G_PARM,
    VIDIOC_S_PARM,
    _V4L2StreamParm,
    fix_device_frame_interval,
    fix_zero_frame_intervals,
)


class FakeDevice:
    """Simulates the G_PARM/S_PARM ioctls of a single V4L2 node."""

    def __init__(self, numerator=0, denominator=1, capability=0x1000, g_parm_error=None, s_parm_error=None):
        self.numerator = numerator
        self.denominator = denominator
        self.capability = capability
        self.g_parm_error = g_parm_error
        self.s_parm_error = s_parm_error
        self.s_parm_calls = []

    def ioctl(self, _fd, request, parm):
        capture = parm.parm.capture
        if request == VIDIOC_G_PARM:
            if self.g_parm_error:
                raise OSError(self.g_parm_error, "g_parm")
            capture.capability = self.capability
            capture.timeperframe.numerator = self.numerator
            capture.timeperframe.denominator = self.denominator
        elif request == VIDIOC_S_PARM:
            if self.s_parm_error:
                raise OSError(self.s_parm_error, "s_parm")
            requested = (capture.timeperframe.numerator, capture.timeperframe.denominator)
            self.s_parm_calls.append(requested)
            self.numerator, self.denominator = requested
        else:
            raise AssertionError(f"unexpected ioctl {request:#x}")
        return 0


@pytest.fixture
def fake_fd():
    with patch.object(v4l2.os, "open", return_value=42) as mock_open, patch.object(v4l2.os, "close") as mock_close:
        yield mock_open, mock_close


def _run(device: FakeDevice, fps=30):
    with patch.object(v4l2.sys, "platform", "linux"), patch("fcntl.ioctl", side_effect=device.ioctl):
        return fix_device_frame_interval("/dev/video0", fps)


def test_struct_matches_kernel_abi():
    # struct v4l2_streamparm is a u32 type followed by a 200-byte union.
    assert ctypes.sizeof(_V4L2StreamParm) == 204
    assert VIDIOC_G_PARM == 0xC0CC5615
    assert VIDIOC_S_PARM == 0xC0CC5616


def test_repairs_zero_interval(fake_fd):
    device = FakeDevice(numerator=0, denominator=1)
    assert _run(device, fps=30) is True
    assert device.s_parm_calls == [(1, 30)]
    fake_fd[1].assert_called_once_with(42)


def test_leaves_valid_interval_untouched(fake_fd):
    device = FakeDevice(numerator=1, denominator=30)
    assert _run(device) is False
    assert device.s_parm_calls == []


def test_skips_nodes_without_g_parm(fake_fd):
    device = FakeDevice(g_parm_error=errno.EINVAL)
    assert _run(device) is False
    assert device.s_parm_calls == []
    fake_fd[1].assert_called_once_with(42)


def test_skips_devices_without_timeperframe_capability(fake_fd):
    device = FakeDevice(numerator=0, denominator=1, capability=0)
    assert _run(device) is False
    assert device.s_parm_calls == []


def test_s_parm_failure_returns_false(fake_fd):
    device = FakeDevice(numerator=0, denominator=1, s_parm_error=errno.EBUSY)
    assert _run(device) is False


def test_unopenable_device_returns_false():
    with (
        patch.object(v4l2.sys, "platform", "linux"),
        patch.object(v4l2.os, "open", side_effect=PermissionError(errno.EACCES, "denied")),
    ):
        assert fix_device_frame_interval("/dev/video0") is False


def test_fix_zero_frame_intervals_scans_video_nodes(tmp_path):
    for name in ("video0", "video1", "media0"):
        (tmp_path / name).touch()
    repaired = {str(tmp_path / "video1")}
    with (
        patch.object(v4l2.sys, "platform", "linux"),
        patch.object(v4l2, "fix_device_frame_interval", side_effect=lambda dev, _fps: str(dev) in repaired) as fix,
    ):
        assert fix_zero_frame_intervals(tmp_path) == [str(tmp_path / "video1")]
    assert [str(call.args[0]) for call in fix.call_args_list] == [str(tmp_path / "video0"), str(tmp_path / "video1")]


def test_fix_zero_frame_intervals_noop_off_linux(tmp_path):
    (tmp_path / "video0").touch()
    with patch.object(v4l2.sys, "platform", "darwin"), patch.object(v4l2, "fix_device_frame_interval") as fix:
        assert fix_zero_frame_intervals(tmp_path) == []
    fix.assert_not_called()


def test_importable_and_noop_without_fcntl(tmp_path):
    """On Windows there is no fcntl module; importing and calling must still work."""
    (tmp_path / "video0").touch()
    with patch.dict(sys.modules, {"fcntl": None}), patch.object(sys, "platform", "win32"):
        module = importlib.reload(v4l2)
        try:
            assert module.fix_device_frame_interval(tmp_path / "video0") is False
            assert module.fix_zero_frame_intervals(tmp_path) == []
        finally:
            importlib.reload(v4l2)
