# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Worker protocol behavior tests."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from physicalai import isolate
from physicalai.engine import Run, TrainingCancelledError, WorkerEnv, run_worker
from physicalai.engine import _environment, _runner

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _worker_env(tmp_path: Path, source: str) -> WorkerEnv:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='test-worker'\n", encoding="utf-8")
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_text(source, encoding="utf-8")
    dist_info = tmp_path / "demo_plugin-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.3\nName: demo-plugin\nVersion: 1.0\nRequires-Dist: lerobot[dataset]>=0.6.1,<0.7\n",
        encoding="utf-8",
    )
    return WorkerEnv(
        framework="demo",
        project=project,
        worker=worker,
        distribution="demo-plugin",
        package="lerobot",
    )


def _protocol_writer(*messages: str) -> str:
    return "\n".join(
        [
            "import json, os, sys",
            "fd = int(os.environ['PHYSICALAI_WORKER_PROTOCOL_FD'])",
            "prefix = bytes([30]) + b'physicalai-worker '",
            "def send(value): os.write(fd, prefix + (json.dumps(value) + '\\n').encode())",
            *[f"send({message})" for message in messages],
        ],
    )


class _Hooks:
    def __init__(self) -> None:
        self.steps: list[tuple[int, int | None, Mapping[str, float]]] = []
        self.stop_requested = False

    def on_step(self, step: int, total: int | None, metrics: Mapping[str, float]) -> None:
        self.steps.append((step, total, dict(metrics)))

    def should_stop(self) -> bool:
        return self.stop_requested


@pytest.mark.parametrize("use_override", [False, True])
def test_worker_runner_launches_managed_entrypoints_through_runtime_isolate(
    use_override: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _protocol_writer(
        "{'type': 'hello', 'protocol': 1, 'framework': 'demo', 'package': 'lerobot', 'version': '0.6.1'}",
        "{'type': 'result', 'output_dir': None, 'checkpoint': None}",
    )
    worker = tmp_path / "override_worker.py"
    worker.write_text(source, encoding="utf-8")
    env = _worker_env(tmp_path, source)
    monkeypatch.syspath_prepend(str(tmp_path))
    root = tmp_path / "prepared"
    project = root / "project"
    project.mkdir(parents=True)
    (project / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "lerobot"\nversion = "0.6.1"\n',
        encoding="utf-8",
    )
    python = root / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    declaration = _environment.to_isolate_environment(env, "cpu")
    isolated = isolate.PreparedEnvironment(path=root, key=declaration.key(), profile=declaration.profile)
    engine_prepared = _environment.PreparedEnvironment(
        framework=env.framework,
        name=env.name,
        accelerator="cpu",
        key=env.key(),
        distribution=env.distribution,
        isolate=isolated,
    )
    monkeypatch.setattr(_runner, "prepare_environment", lambda *_args: engine_prepared)
    real_start = isolate.start
    calls: list[tuple[object, object, tuple[str, ...]]] = []

    def track_start(
        prepared: isolate.PreparedEnvironment,
        entrypoint: object,
        args: Sequence[str] = (),
        **kwargs: object,
    ) -> isolate.Session:
        calls.append((prepared, entrypoint, tuple(args)))
        return real_start(prepared, entrypoint, args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(isolate, "start", track_start)

    result = run_worker(env, ["--steps=1"], device="cpu", worker=worker if use_override else None)

    assert result == Run(framework="demo", output_dir=None, checkpoint=None)
    assert len(calls) == 1
    assert calls[0][0] is isolated
    assert calls[0][1] == (worker if use_override else env.worker)
    assert calls[0][2] == ("--steps=1",)


def test_worker_runner_checks_handshake_reports_steps_and_returns_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Worker logs remain separate from protocol events while steps reach hooks."""
    monkeypatch.syspath_prepend(str(tmp_path))
    env = _worker_env(
        tmp_path,
        _protocol_writer(
            "{'type': 'hello', 'protocol': 1, 'framework': 'demo', 'package': 'lerobot', 'version': '0.6.1'}",
            "{'type': 'progress', 'step': 2, 'total': 5, 'metrics': {'loss': 0.25}}",
            "{'type': 'result', 'output_dir': 'run', 'checkpoint': 'run/checkpoint'}",
        )
        + "\nprint('native stdout')\nprint('native stderr', file=sys.stderr)\n",
    )
    hooks = _Hooks()

    with caplog.at_level(logging.INFO, logger="physicalai.engine._runner"):
        result = run_worker(env, [], device="cpu", hooks=hooks, interpreter=sys.executable)

    assert result == Run(framework="demo", output_dir=Path("run"), checkpoint=Path("run/checkpoint"))
    assert hooks.steps == [(2, 5, {"loss": 0.25})]
    assert "native stdout" in caplog.text
    assert "native stderr" in caplog.text


def test_worker_runner_reads_protocol_messages_after_progress_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Progress events remain readable when stderr emits carriage-return updates."""
    monkeypatch.syspath_prepend(str(tmp_path))
    worker = """\
import json, os, sys
fd = int(os.environ['PHYSICALAI_WORKER_PROTOCOL_FD'])
prefix = bytes([30]) + b'physicalai-worker '
def send(value): os.write(fd, prefix + (json.dumps(value) + '\\n').encode())
send({'type': 'hello', 'protocol': 1, 'framework': 'demo', 'package': 'lerobot', 'version': '0.6.1'})
os.write(2, b'Training: 1/5\\r')
send({'type': 'progress', 'step': 1, 'total': 1, 'metrics': {'loss': 0.25}})
send({'type': 'result', 'output_dir': None, 'checkpoint': None})
"""
    env = _worker_env(tmp_path, worker)
    hooks = _Hooks()

    with caplog.at_level(logging.INFO, logger="physicalai.engine._runner"):
        result = run_worker(env, [], device="cpu", hooks=hooks, interpreter=sys.executable)

    assert result == Run(framework="demo", output_dir=None, checkpoint=None)
    assert hooks.steps == [(1, 1, {"loss": 0.25})]
    assert "Training: 1/5" in caplog.text


def test_worker_runner_rejects_a_version_outside_plugin_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The worker fails before training when its framework version is unsupported."""
    monkeypatch.syspath_prepend(str(tmp_path))
    env = _worker_env(
        tmp_path,
        _protocol_writer(
            "{'type': 'hello', 'protocol': 1, 'framework': 'demo', 'package': 'lerobot', 'version': '0.7.0'}",
        ),
    )

    with pytest.raises(RuntimeError, match="outside the supported range"):
        run_worker(env, [], device="cpu", hooks=None, interpreter=sys.executable)


def test_worker_runner_finds_a_bare_interpreter_name_on_the_worker_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A bare name such as ``python3`` is found on the worker's PATH, not in the working directory."""
    monkeypatch.syspath_prepend(str(tmp_path))
    env = _worker_env(
        tmp_path,
        _protocol_writer(
            "{'type': 'hello', 'protocol': 1, 'framework': 'demo', 'package': 'lerobot', 'version': '0.6.1'}",
            "{'type': 'result', 'output_dir': None, 'checkpoint': None}",
        ),
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "custom-python"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    started: list[object] = []
    real_start = isolate.start

    def track_start(target: object, **kwargs: object) -> isolate.Session:
        started.append(target)
        return real_start(target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(isolate, "start", track_start)
    path = os.pathsep.join([str(bin_dir), os.environ.get("PATH", "")])

    result = run_worker(env, [], device="cpu", interpreter="custom-python", extra_env={"PATH": path})

    assert result == Run(framework="demo", output_dir=None, checkpoint=None)
    assert started == [isolate.ExternalInterpreter(wrapper)]


def test_worker_runner_rejects_a_bare_interpreter_name_missing_from_the_worker_path(tmp_path: Path) -> None:
    env = _worker_env(tmp_path, _protocol_writer())

    with pytest.raises(FileNotFoundError, match="'custom-python' was not found on PATH"):
        run_worker(env, [], device="cpu", interpreter="custom-python", extra_env={"PATH": str(tmp_path)})


def test_worker_runner_delivers_cancellation_and_reports_last_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Host cancellation reaches the worker over stdin and retains its last step."""
    monkeypatch.syspath_prepend(str(tmp_path))
    worker = "\n".join(
        [
            "import json, os, sys",
            "fd = int(os.environ['PHYSICALAI_WORKER_PROTOCOL_FD'])",
            "prefix = bytes([30]) + b'physicalai-worker '",
            "def send(value): os.write(fd, prefix + (json.dumps(value) + '\\n').encode())",
            "send({'type': 'hello', 'protocol': 1, 'framework': 'demo', 'package': 'lerobot', 'version': '0.6.1'})",
            "assert json.loads(sys.stdin.readline())['type'] == 'ack'",
            "send({'type': 'progress', 'step': 3, 'total': 5, 'metrics': {}})",
            "assert json.loads(sys.stdin.readline())['type'] == 'cancel'",
            "send({'type': 'cancelled', 'step': 4})",
        ],
    )
    env = _worker_env(tmp_path, worker)

    class _StopHooks(_Hooks):
        def should_stop(self) -> bool:
            return bool(self.steps)

    hooks = _StopHooks()
    with pytest.raises(TrainingCancelledError) as exc_info:
        run_worker(env, [], device="cpu", hooks=hooks, interpreter=sys.executable)

    assert exc_info.value.step == 4
    assert hooks.steps == [(3, 5, {})]


def test_worker_runner_rejects_invalid_protocol_before_accepting_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A progress event cannot replace the mandatory version handshake."""
    monkeypatch.syspath_prepend(str(tmp_path))
    env = _worker_env(
        tmp_path,
        _protocol_writer("{'type': 'progress', 'step': 1, 'total': 2, 'metrics': {}}"),
    )

    with pytest.raises(RuntimeError, match="first protocol message must be a hello"):
        run_worker(env, [], device="cpu", hooks=None, interpreter=sys.executable)


def test_worker_runner_reports_the_worker_error_when_it_exits_right_away(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A worker that fails immediately is reported by its own error, not as a handshake or exit failure."""
    monkeypatch.syspath_prepend(str(tmp_path))
    env = _worker_env(
        tmp_path,
        _protocol_writer(
            "{'type': 'hello', 'protocol': 1, 'framework': 'demo', 'package': 'lerobot', 'version': '0.6.1'}",
            "{'type': 'error', 'error': 'ValueError', 'message': 'bad dataset'}",
        )
        + "\nsys.exit(1)\n",
    )
    read_protocol = _runner._read_protocol

    def slow_reader(*args: object) -> None:
        # Let the worker exit before its events are read, as on a busy host.
        import time

        time.sleep(0.5)
        read_protocol(*args)

    monkeypatch.setattr(_runner, "_read_protocol", slow_reader)

    with pytest.raises(RuntimeError, match="demo worker raised ValueError: bad dataset"):
        run_worker(env, [], device="cpu", interpreter=sys.executable)


def test_worker_runner_shows_worker_output_when_it_fails_before_the_handshake(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A worker that dies before its handshake reports what it printed, such as its traceback."""
    monkeypatch.syspath_prepend(str(tmp_path))
    env = _worker_env(tmp_path, "import physicalai_missing_worker_dependency\n")

    with pytest.raises(_runner.WorkerProtocolError, match="No module named 'physicalai_missing_worker_dependency'"):
        run_worker(env, [], device="cpu", interpreter=sys.executable)


# The tests below characterize the user-managed interpreter path (``--interpreter``), which launches
# the worker itself rather than through physicalai.isolate. They pin what callers observe, so that path
# can move onto an isolate session without changing behaviour unnoticed.

_HELLO = "{'type': 'hello', 'protocol': 1, 'framework': 'demo', 'package': 'lerobot', 'version': '0.6.1'}"


def _hanging_worker(*messages: str, ignore_sigterm: bool = False) -> str:
    """Record the worker's pid in argv[1], send *messages*, then hang without reading stdin."""
    lines = [_protocol_writer(), "import signal, time", "from pathlib import Path"]
    if ignore_sigterm:
        lines.append("signal.signal(signal.SIGTERM, signal.SIG_IGN)")
    lines.append("Path(sys.argv[1]).write_text(str(os.getpid()))")
    lines.extend(f"send({message})" for message in messages)
    return "\n".join([*lines, "while True:", "    time.sleep(0.05)"])


def _has_exited(pid_file: Path) -> bool:
    try:
        os.kill(int(pid_file.read_text(encoding="utf-8")), 0)
    except ProcessLookupError:
        return True
    return False


@pytest.mark.parametrize("ignore_sigterm", [False, True], ids=["terminated", "killed"])
def test_interpreter_worker_that_ignores_cancellation_is_stopped(
    ignore_sigterm: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unresponsive worker is terminated after the cancel grace period, then killed, and reaped."""
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(_runner, "_CANCEL_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(_runner, "_TERMINATE_GRACE_SECONDS", 0.2)
    env = _worker_env(
        tmp_path,
        _hanging_worker(
            _HELLO,
            "{'type': 'progress', 'step': 2, 'total': 9, 'metrics': {}}",
            ignore_sigterm=ignore_sigterm,
        ),
    )
    pid_file = tmp_path / "worker.pid"

    class _StopHooks(_Hooks):
        def should_stop(self) -> bool:
            return bool(self.steps)

    with pytest.raises(TrainingCancelledError) as exc_info:
        run_worker(env, [str(pid_file)], device="cpu", hooks=_StopHooks(), interpreter=sys.executable)

    assert exc_info.value.step == 2
    assert _has_exited(pid_file)


def test_interpreter_worker_without_a_handshake_times_out_and_is_stopped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(_runner, "_STARTUP_TIMEOUT_SECONDS", 0.3)
    env = _worker_env(tmp_path, _hanging_worker())
    pid_file = tmp_path / "worker.pid"

    with pytest.raises(TimeoutError, match="'demo' did not complete its version handshake"):
        run_worker(env, [str(pid_file)], device="cpu", interpreter=sys.executable)

    assert _has_exited(pid_file)


def test_interpreter_worker_is_stopped_when_the_host_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An exception raised by host hooks stops the worker instead of leaving it running."""
    monkeypatch.syspath_prepend(str(tmp_path))
    env = _worker_env(tmp_path, _hanging_worker(_HELLO, "{'type': 'progress', 'step': 1, 'total': 9, 'metrics': {}}"))
    pid_file = tmp_path / "worker.pid"

    class _FailingHooks(_Hooks):
        def on_step(self, step: int, total: int | None, metrics: Mapping[str, float]) -> None:
            msg = "host failed"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="host failed"):
        run_worker(env, [str(pid_file)], device="cpu", hooks=_FailingHooks(), interpreter=sys.executable)

    assert _has_exited(pid_file)


def test_interpreter_worker_runs_in_the_host_environment_with_worker_variables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A user-managed interpreter keeps the host's variables, including PYTHONPATH, and gets no project."""
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "user-site"))
    monkeypatch.setenv("PHYSICALAI_TEST_HOST_VARIABLE", "kept")
    names = [
        "PYTHONPATH",
        "PHYSICALAI_TEST_HOST_VARIABLE",
        "PHYSICALAI_TEST_EXTRA",
        "PHYSICALAI_WORKER_PROTOCOL_FD",
        "PHYSICALAI_WORKER_FRAMEWORK",
        "PHYSICALAI_WORKER_PACKAGE",
        "PHYSICALAI_WORKER_DEVICE",
        "PHYSICALAI_WORKER_PROJECT",
    ]
    record = "\n".join([
        "from pathlib import Path",
        f"names = {names!r}",
        "Path(sys.argv[1]).write_text(json.dumps({name: os.environ.get(name) for name in names}))",
    ])
    env = _worker_env(
        tmp_path,
        _protocol_writer(_HELLO) + "\n" + record + "\n" + "send({'type': 'result', 'output_dir': None, 'checkpoint': None})",
    )
    seen = tmp_path / "environment.json"

    run_worker(
        env,
        [str(seen)],
        device="xpu",
        interpreter=sys.executable,
        extra_env={"PHYSICALAI_TEST_EXTRA": "extra"},
    )

    assert json.loads(seen.read_text(encoding="utf-8")) == {
        "PYTHONPATH": str(tmp_path / "user-site"),
        "PHYSICALAI_TEST_HOST_VARIABLE": "kept",
        "PHYSICALAI_TEST_EXTRA": "extra",
        "PHYSICALAI_WORKER_PROTOCOL_FD": "2",
        "PHYSICALAI_WORKER_FRAMEWORK": "demo",
        "PHYSICALAI_WORKER_PACKAGE": "lerobot",
        "PHYSICALAI_WORKER_DEVICE": "xpu",
        "PHYSICALAI_WORKER_PROJECT": None,
    }
