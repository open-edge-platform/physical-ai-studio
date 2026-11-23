# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot worker protocol integration tests using a fake native package."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from physicalai.engine import Run, TrainHooks, WorkerEnv, run_worker
from physicalai_lerobot_policy_plugin.engine import LeRobotEngine


def _fake_lerobot_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[WorkerEnv, Path]:
    """Install a fake LeRobot whose trainer creates the run directory, and declare its worker."""
    fake_site = tmp_path / "site"
    train_module = fake_site / "lerobot" / "scripts" / "lerobot_train.py"
    config_module = fake_site / "lerobot" / "configs" / "train.py"
    train_module.parent.mkdir(parents=True)
    config_module.parent.mkdir(parents=True)
    (fake_site / "lerobot" / "__init__.py").write_text("", encoding="utf-8")
    (fake_site / "lerobot" / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    (fake_site / "lerobot" / "configs" / "__init__.py").write_text("", encoding="utf-8")
    config_module.write_text(
        "from pathlib import Path\n"
        "import os\n"
        "class TrainPipelineConfig:\n"
        "    def __init__(self):\n"
        "        self.steps = 2\n"
        "        self.output_dir = Path(os.environ['FAKE_RUN_OUTPUT'])\n"
        "    def validate(self):\n"
        "        return None\n",
        encoding="utf-8",
    )
    train_module.write_text(
        "from pathlib import Path\n"
        "from lerobot.configs.train import TrainPipelineConfig\n"
        "class Tracker:\n"
        "    steps = 0\n"
        "    def to_dict(self, use_avg=False):\n"
        "        return {'loss': 0.25, 'grad_norm': float('nan'), 'lr': float('inf'), 'steps': self.steps}\n"
        "def update_policy(tracker):\n"
        "    return tracker, None\n"
        "def main():\n"
        "    config = TrainPipelineConfig()\n"
        "    config.validate()\n"
        "    update_policy(Tracker())\n"
        "    (config.output_dir / 'checkpoints/last/pretrained_model').mkdir(parents=True)\n",
        encoding="utf-8",
    )
    plugin_metadata = fake_site / "demo_plugin-1.0.dist-info"
    plugin_metadata.mkdir()
    (plugin_metadata / "METADATA").write_text(
        "Metadata-Version: 2.3\nName: demo-plugin\nVersion: 1.0\n"
        "Requires-Dist: lerobot[dataset]>=0.6.1,<0.7\n",
        encoding="utf-8",
    )
    lerobot_metadata = fake_site / "lerobot-0.6.1.dist-info"
    lerobot_metadata.mkdir()
    (lerobot_metadata / "METADATA").write_text(
        "Metadata-Version: 2.3\nName: lerobot\nVersion: 0.6.1\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(fake_site))
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(fake_site), current_pythonpath))))
    output_dir = tmp_path / "native-run"
    monkeypatch.setenv("FAKE_RUN_OUTPUT", str(output_dir))
    project = tmp_path / "worker-project"
    project.mkdir()
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    env = WorkerEnv(
        framework="lerobot",
        project=project,
        worker=LeRobotEngine.env.worker,
        distribution="demo-plugin",
        package="lerobot",
    )

    return env, output_dir


def test_native_worker_reports_le_robot_update_metrics_and_run_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The real worker bridge reports native updates over the shared protocol.

    Non-finite metrics (for example a skipped AMP step) are dropped instead of stopping training.
    """
    env, output_dir = _fake_lerobot_worker(monkeypatch, tmp_path)

    class _Hooks:
        steps: list[tuple[int, int | None, Mapping[str, float]]]

        def __init__(self) -> None:
            self.steps = []

        def on_step(self, step: int, total: int | None, metrics: Mapping[str, float]) -> None:
            self.steps.append((step, total, dict(metrics)))

        @staticmethod
        def should_stop() -> bool:
            return False

    hooks: TrainHooks = _Hooks()
    run = run_worker(env, ["--steps=2"], device="cpu", hooks=hooks, interpreter=sys.executable)

    assert run == Run(
        framework="lerobot",
        output_dir=output_dir,
        checkpoint=output_dir / "checkpoints" / "last" / "pretrained_model",
    )
    assert hooks.steps == [(1, 2, {"loss": 0.25})]


def test_native_worker_waits_for_the_host_to_accept_its_handshake(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A worker whose version the host rejects never starts native training."""
    from packaging.version import Version

    from physicalai.engine import _runner
    from physicalai.engine._runner import WorkerProtocolError

    env, output_dir = _fake_lerobot_worker(monkeypatch, tmp_path)
    read_protocol = _runner._read_protocol

    def slow_reader(*args: object) -> None:
        # Validate the handshake late, as on a busy host, so an eager worker would start training.
        import time

        time.sleep(0.5)
        read_protocol(*args)

    monkeypatch.setattr(_runner, "_read_protocol", slow_reader)

    with pytest.raises(WorkerProtocolError):
        run_worker(env, [], device="cpu", interpreter=sys.executable, expected_version=Version("9.9.9"))

    assert not output_dir.exists()
