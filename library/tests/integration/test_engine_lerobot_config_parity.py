# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native LeRobot config parity through both command entry points."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig


@pytest.mark.integration
def test_train_config_json_matches_native_lerobot_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Both CLIs should resolve the same native settings without starting training."""
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts import lerobot_train
    from physicalai.cli.main import main as physicalai_main

    class _ValidationReached(Exception):
        def __init__(self, config: TrainPipelineConfig) -> None:
            self.config = config

    original_validate = TrainPipelineConfig.validate

    def capture_validated_config(config: TrainPipelineConfig) -> None:
        original_validate(config)
        raise _ValidationReached(config)

    monkeypatch.setattr(TrainPipelineConfig, "validate", capture_validated_config)
    flags = [
        "--policy.type=act",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=lerobot/pusht",
        "--steps=2",
        "--seed=19",
        "--batch_size=2",
        "--num_workers=0",
        f"--output_dir={tmp_path / 'native-run'}",
    ]

    native_argv = ["lerobot-train", *flags]
    monkeypatch.setattr(sys, "argv", native_argv)
    with pytest.raises(_ValidationReached) as native_capture:
        lerobot_train.main()

    physicalai_argv = ["physicalai", "train", "lerobot", *flags]
    monkeypatch.setattr(sys, "argv", physicalai_argv)
    with pytest.raises(_ValidationReached) as studio_capture:
        physicalai_main(physicalai_argv[1:])

    assert sys.argv is physicalai_argv
    assert native_capture.value.config is not studio_capture.value.config

    native_output = tmp_path / "native-config"
    studio_output = tmp_path / "studio-config"
    native_output.mkdir()
    studio_output.mkdir()
    native_capture.value.config.save_pretrained(native_output)
    studio_capture.value.config.save_pretrained(studio_output)

    native_config = json.loads((native_output / "train_config.json").read_text(encoding="utf-8"))
    studio_config = json.loads((studio_output / "train_config.json").read_text(encoding="utf-8"))
    assert studio_config == native_config
