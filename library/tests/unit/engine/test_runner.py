# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Worker protocol behavior tests."""

from __future__ import annotations

import logging
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
        path=root,
        _isolate=isolated,
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
            "send({'type': 'progress', 'step': 3, 'total': 5, 'metrics': {}})",
            "message = sys.stdin.readline()",
            "assert json.loads(message)['type'] == 'cancel'",
            "send({'type': 'cancelled', 'step': 3})",
        ],
    )
    env = _worker_env(tmp_path, worker)

    class _StopHooks(_Hooks):
        def should_stop(self) -> bool:
            return bool(self.steps)

    hooks = _StopHooks()
    with pytest.raises(TrainingCancelledError) as exc_info:
        run_worker(env, [], device="cpu", hooks=hooks, interpreter=sys.executable)

    assert exc_info.value.step == 3
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
