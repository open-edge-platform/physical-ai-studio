# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LeRobot-native framework adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from physicalai.engine import Engine


@pytest.mark.parametrize(
    "native_args",
    [
        ["--policy.type=act", "--dataset.repo_id=lerobot/pusht", "--steps=2"],
        ["--help"],
    ],
)
def test_physicalai_train_forwards_lerobot_arguments_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    native_args: list[str],
) -> None:
    from lerobot.scripts import lerobot_train
    from physicalai.cli.main import main as physicalai_main

    captured_argv: list[str] = []

    def native_main() -> None:
        captured_argv.extend(sys.argv)

    monkeypatch.setattr(lerobot_train, "main", native_main)
    host_argv = ["physicalai", "train", "lerobot", *native_args]
    monkeypatch.setattr(sys, "argv", host_argv)

    exit_code = physicalai_main(host_argv[1:])

    assert exit_code == 0
    assert captured_argv == ["lerobot-train", *native_args]
    assert sys.argv is host_argv
    assert sys.argv == host_argv


def test_engine_passes_the_original_train_pipeline_config_to_native_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts import lerobot_train
    from lerobot.utils import import_utils

    config = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/pusht"))
    events: list[object] = []
    monkeypatch.setattr(lerobot_train, "train", lambda native_config: events.append(native_config))
    monkeypatch.setattr(import_utils, "register_third_party_plugins", lambda: events.append("plugins"))

    scoped_argv = ["pytest", "unrelated", "--test-runner-flag"]
    monkeypatch.setattr(sys, "argv", scoped_argv)

    run = Engine().train(config)

    assert run.framework == "lerobot"
    assert events == ["plugins", config]
    assert events[1] is config
    assert sys.argv is scoped_argv
    assert sys.argv == ["pytest", "unrelated", "--test-runner-flag"]


def test_engine_cli_training_returns_native_run_and_restores_config_patch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.scripts import lerobot_train

    output_dir = tmp_path / "native-run"
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/pusht"),
        policy=ACTConfig(push_to_hub=False),
        output_dir=output_dir,
        steps=1,
    )

    def native_main() -> None:
        config.validate()
        (output_dir / "checkpoints" / "last" / "pretrained_model").mkdir(parents=True)

    monkeypatch.setattr(lerobot_train, "main", native_main)
    original_validate = TrainPipelineConfig.validate
    host_argv = ["physicalai", "train", "lerobot", "--policy.type=act"]
    monkeypatch.setattr(sys, "argv", host_argv)

    run = Engine().train(["--policy.type=act"], framework="lerobot")

    assert run.output_dir == output_dir
    assert run.checkpoint == output_dir / "checkpoints" / "last" / "pretrained_model"
    assert TrainPipelineConfig.validate is original_validate
    assert sys.argv is host_argv


def test_engine_cli_training_restores_config_patch_after_native_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.scripts import lerobot_train

    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/pusht"),
        policy=ACTConfig(push_to_hub=False),
        output_dir=tmp_path / "native-run",
        steps=1,
    )

    def native_main() -> None:
        config.validate()
        raise RuntimeError("native CLI failure")

    monkeypatch.setattr(lerobot_train, "main", native_main)
    original_validate = TrainPipelineConfig.validate
    host_argv = ["physicalai", "train", "lerobot", "--policy.type=act"]
    monkeypatch.setattr(sys, "argv", host_argv)

    with pytest.raises(RuntimeError, match="native CLI failure"):
        Engine().train(["--policy.type=act"], framework="lerobot")

    assert TrainPipelineConfig.validate is original_validate
    assert sys.argv is host_argv
