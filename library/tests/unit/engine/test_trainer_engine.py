# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""First-party Trainer Engine compatibility tests."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning
import pytest
import torch

from physicalai.engine import Engine
from physicalai.policies.base.policy import Policy
from physicalai.train import Trainer

if TYPE_CHECKING:
    from collections.abc import Callable

    from physicalai.data import Observation


def test_trainer_remains_a_lightning_trainer_with_its_existing_constructor() -> None:
    trainer = Trainer(max_epochs=3, logger=False, default_root_dir="runs", num_sanity_val_steps=0)

    assert isinstance(trainer, lightning.Trainer)
    assert isinstance(trainer, Engine)
    assert trainer.max_epochs == 3
    assert trainer.logger is None
    assert Path(trainer.default_root_dir) == Path("runs")


@pytest.mark.parametrize("method_name", ["fit", "validate", "test", "predict"])
def test_trainer_training_methods_are_unchanged_lightning_methods(method_name: str) -> None:
    trainer_method = getattr(Trainer, method_name)
    lightning_method = getattr(lightning.Trainer, method_name)

    assert trainer_method is lightning_method
    assert inspect.signature(trainer_method) == inspect.signature(lightning_method)


def test_trainer_export_forwards_policy_export_arguments() -> None:
    input_sample = {"state": torch.zeros(1, 2)}
    post_export_hooks: list[Callable[[str], None]] = [lambda path: None]
    output_path = Path("exports/act")
    backend_kwargs = {"opset_version": 17, "dynamic_axes": {"action": {0: "batch"}}}

    class _ExportPolicy(Policy):
        export_call: dict[str, Any] | None = None

        def forward(self, batch: Observation) -> torch.Tensor:
            return torch.zeros(1, 2)

        def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
            return torch.zeros(1, 2)

        def export(
            self,
            destination: str | Path,
            backend: str,
            input_sample: dict[str, torch.Tensor] | None = None,
            post_export_hooks: list[Callable[[str], None]] | None = None,
            **kwargs: Any,
        ) -> None:
            self.export_call = {
                "destination": destination,
                "backend": backend,
                "input_sample": input_sample,
                "post_export_hooks": post_export_hooks,
                "kwargs": kwargs,
            }

    policy = _ExportPolicy()

    result = Trainer.export(
        policy,
        output_path,
        backend="onnx",
        input_sample=input_sample,
        post_export_hooks=post_export_hooks,
        **backend_kwargs,
    )

    assert result == output_path
    assert policy.export_call == {
        "destination": output_path,
        "backend": "onnx",
        "input_sample": input_sample,
        "post_export_hooks": post_export_hooks,
        "kwargs": backend_kwargs,
    }
    assert policy.export_call["input_sample"] is input_sample
    assert policy.export_call["post_export_hooks"] is post_export_hooks


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

    result = Trainer.cli(["--trainer.accelerator=gpu", "--config", "act.yaml"], interpreter=Path("unused"))

    assert result == 9
    assert calls == [["fit", "--trainer.accelerator=gpu", "--config", "act.yaml"]]


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
