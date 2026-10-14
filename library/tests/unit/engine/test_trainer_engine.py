# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""First-party Trainer Engine compatibility tests."""

from __future__ import annotations

from pathlib import Path

import lightning
import pytest

from physicalai.engine import Engine
from physicalai.train import Trainer


def test_trainer_remains_a_lightning_trainer_with_its_existing_constructor() -> None:
    trainer = Trainer(max_epochs=3, logger=False, default_root_dir="runs", num_sanity_val_steps=0)

    assert isinstance(trainer, lightning.Trainer)
    assert isinstance(trainer, Engine)
    assert trainer.max_epochs == 3
    assert trainer.logger is None
    assert Path(trainer.default_root_dir) == Path("runs")


def test_physicalai_engine_is_registered_and_create_preserves_trainer_options() -> None:
    engine_class = Engine.get("physicalai")
    trainer = Engine.create("physicalai", max_epochs=2, logger=False)

    assert engine_class is Trainer
    assert isinstance(trainer, Trainer)
    assert trainer.max_epochs == 2


def test_trainer_cli_aliases_the_existing_fit_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        calls.append(argv)
        return 9

    monkeypatch.setattr("physicalai.cli.main.main", fake_main)

    result = Trainer.cli(["--config", "act.yaml"], interpreter=Path("unused"), accelerator="xpu")

    assert result == 9
    assert calls == [["fit", "--config", "act.yaml"]]


def test_first_party_engine_exposes_policy_schema_and_capabilities() -> None:
    schema = Trainer.config_schema("act")
    capabilities = Trainer.capabilities("act")

    assert schema["type"] == "object"
    assert schema["title"] == "ACTConfig"
    assert "cpu" in capabilities.training_accelerators
    assert capabilities.export_formats


def test_studio_owned_first_party_run_path_remains_separate() -> None:
    with pytest.raises(NotImplementedError, match="run_training_job"):
        Trainer.run({}, dataset=Path("dataset"), output_dir=Path("output"), device="cpu")
