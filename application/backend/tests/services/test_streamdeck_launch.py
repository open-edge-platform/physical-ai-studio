import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from exceptions import BaseException as APIError
from services.streamdeck_service import StreamDeckService


def test_native_launch_uses_fixed_script_and_settings_once(monkeypatch, tmp_path):
    settings = SimpleNamespace(host="::1", port=7860, log_dir=tmp_path)
    process = Mock(pid=321)
    process.poll.return_value = None
    spawn = Mock(return_value=process)
    monkeypatch.setattr("services.streamdeck_service.get_settings", lambda: settings)
    monkeypatch.setattr("services.streamdeck_service.shutil.which", lambda name: "/tools/uv")
    monkeypatch.setattr(StreamDeckService, "native_launch_available", staticmethod(lambda: True))
    monkeypatch.setattr("services.streamdeck_service.subprocess.Popen", spawn)
    monkeypatch.setenv("HTTP_SERVER_PORT", "9999")
    service = StreamDeckService()
    service.start()
    service.start()
    spawn.assert_called_once()
    assert spawn.call_args.args[0] == [
        "/tools/uv",
        "run",
        "--no-project",
        str(service.bridge_path()),
        "--url",
        "http://[::1]:9999",
    ]
    registry = Mock()
    assert service.status(registry).helper_starting
    process.poll.return_value = 1
    assert not service.status(registry).helper_starting
    assert "Helper exited" in service.status(registry).error


def test_container_and_public_backend_cannot_launch(monkeypatch):
    monkeypatch.setattr("services.streamdeck_service.get_settings", lambda: SimpleNamespace(host="0.0.0.0"))
    assert not StreamDeckService.native_launch_available()
    monkeypatch.setattr("services.streamdeck_service.get_settings", lambda: SimpleNamespace(host="127.0.0.1"))
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: str(path) == "/.dockerenv" or original_exists(path))
    assert not StreamDeckService.native_launch_available()
    with pytest.raises(APIError):
        StreamDeckService().start()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process group cleanup")
def test_shutdown_stops_only_owned_process_tree(monkeypatch):
    kill = Mock()
    monkeypatch.setattr("services.streamdeck_service.os.killpg", kill)
    service = StreamDeckService()
    service.close()
    kill.assert_not_called()
    process = Mock(pid=321)
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("helper", 5), None]
    service.process = process
    service.close()
    assert [call.args for call in kill.call_args_list] == [(321, signal.SIGTERM), (321, signal.SIGKILL)]
    kill.side_effect = ProcessLookupError
    service.close()


def test_windows_shutdown_terminates_the_owned_tree(monkeypatch):
    run = Mock()
    monkeypatch.setattr("services.streamdeck_service.sys.platform", "win32")
    monkeypatch.setattr("services.streamdeck_service.subprocess.run", run)
    service = StreamDeckService()
    service.process = Mock(pid=321)
    service.process.poll.return_value = None
    service.close()
    assert run.call_args.args[0] == ["taskkill", "/PID", "321", "/T", "/F"]
