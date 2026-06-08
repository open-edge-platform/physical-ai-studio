# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for studio CLI entry-point subcommands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest
from jsonargparse import ArgumentParser, Namespace

from physicalai.cli import benchmark as benchmark_module
from physicalai.cli import export as export_module
from physicalai.cli import fit as fit_module
from physicalai.cli import predict as predict_module
from physicalai.cli import test as test_module
from physicalai.cli import validate as validate_module
from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

if TYPE_CHECKING:
    from typing import Protocol

    class _CliModule(Protocol):
        def register(self) -> SubcommandSpec: ...


_SUBCOMMAND_MODULES: dict[str, _CliModule] = {
    "fit": fit_module,
    "validate": validate_module,
    "test": test_module,
    "predict": predict_module,
    "benchmark": benchmark_module,
    "export": export_module,
}


def _library_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _act_config_path() -> Path:
    return _library_root() / "configs" / "physicalai" / "act.yaml"


def _libero_config_path() -> Path:
    return _library_root() / "configs" / "benchmark" / "libero.yaml"


class TestRegister:
    """ST-1 register() returns expected names and specs."""

    @pytest.mark.parametrize(("name", "module"), _SUBCOMMAND_MODULES.items())
    def test_register_returns_subcommand_spec(self, name: str, module: _CliModule) -> None:
        spec = module.register()
        assert isinstance(spec, SubcommandSpec)
        assert spec.name == name
        assert isinstance(spec.parser, ArgumentParser)
        assert spec.help


class TestHelp:
    """ST-2 parsers build and ``--help`` exits cleanly."""

    @pytest.mark.parametrize("module", _SUBCOMMAND_MODULES.values())
    def test_parser_help_exits_zero(self, module: _CliModule) -> None:
        parser = module.register().parser
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0


class TestConfigParsing:
    """ST-3 and ST-4 config parsing parity."""

    def test_fit_parser_accepts_existing_act_config(self) -> None:
        parser = fit_module.register().parser
        cfg = parser.parse_args([f"--config={_act_config_path()}"])
        assert cfg.model.class_path == "physicalai.policies.ACT"
        assert cfg.data.class_path == "physicalai.data.lerobot.LeRobotDataModule"
        assert cfg.trainer.max_epochs == 100

    def test_benchmark_parser_accepts_existing_libero_config(self) -> None:
        parser = benchmark_module.register().parser
        cfg = parser.parse_args(
            [
                f"--config={_libero_config_path()}",
                "--policy=physicalai.policies.ACT",
            ],
        )
        assert cfg.benchmark.class_path == "physicalai.benchmark.gyms.LiberoBenchmark"
        assert cfg.output_dir == "./results/benchmark/libero_10"


class TestDispatch:
    """ST-5 and ST-6 dispatch behavior."""

    def test_fit_dispatch_calls_trainer_fit(self) -> None:
        parser = fit_module.register().parser
        cfg = parser.parse_args([f"--config={_act_config_path()}"])
        trainer = MagicMock()
        model = object()
        datamodule = object()

        with patch.object(
            parser,
            "instantiate_classes",
            return_value=MagicMock(trainer=trainer, model=model, data=datamodule),
        ):
            exit_code = fit_module.run(cast(ArgumentParser, parser), cast(Namespace, cfg))

        assert exit_code == 0
        trainer.fit.assert_called_once_with(model=model, datamodule=datamodule)

    def test_benchmark_dispatch_calls_benchmark_evaluate(self, tmp_path: Path) -> None:
        parser = benchmark_module.register().parser
        cfg = parser.parse_args(
            [
                f"--config={_libero_config_path()}",
                "--policy=physicalai.policies.ACT",
                f"--output_dir={tmp_path}",
            ],
        )
        fake_policy = object()
        fake_results = MagicMock()
        fake_results.summary.return_value = "summary"
        fake_benchmark = MagicMock()
        fake_benchmark.evaluate.return_value = fake_results
        fake_benchmark.gyms = [MagicMock(), MagicMock()]

        with (
            patch.object(parser, "instantiate_classes", return_value=MagicMock(benchmark=fake_benchmark)),
            patch("physicalai.cli.benchmark.load_policy", return_value=(fake_policy, "cpu")),
            patch("builtins.print"),
        ):
            exit_code = benchmark_module.run(cast(ArgumentParser, parser), cast(Namespace, cfg))

        assert exit_code == 0
        fake_benchmark.evaluate.assert_called_once_with(policy=fake_policy)
        fake_results.to_json.assert_called_once_with(tmp_path / "results.json")
        fake_results.to_csv.assert_called_once_with(tmp_path / "results.csv")
        for gym in fake_benchmark.gyms:
            gym.close.assert_called_once_with()

    def test_export_dispatch_calls_policy_export(self) -> None:
        parser = export_module.register().parser
        cfg = parser.parse_args(
            [
                "--policy=physicalai.policies.ACT",
                "--ckpt_path=checkpoints/best.ckpt",
                "--backend=openvino",
                "--output_dir=exports/act",
            ],
        )
        fake_policy = MagicMock()

        with patch("physicalai.cli.export.load_policy", return_value=(fake_policy, "cpu")):
            exit_code = export_module.run(cast(ArgumentParser, parser), cast(Namespace, cfg))

        assert exit_code == 0
        fake_policy.export.assert_called_once_with("exports/act", backend="openvino")
