# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for engine registration, settings, training, and export wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from physicalai.engine import Engine, Run, TrainingEngine

from physicalai_gr00t_policy_plugin import engine, environments
from physicalai_gr00t_policy_plugin.config import Gr00tTrainConfig
from physicalai_gr00t_policy_plugin.engine import Gr00tEngine

from .test_package import LIBERO_MODALITY


def test_engine_is_discovered_as_a_training_engine_with_the_openvino_enablement() -> None:
    assert Engine.get("gr00t") is Gr00tEngine
    assert issubclass(Gr00tEngine, TrainingEngine)
    assert Gr00tEngine.policies() == ["gr00t_n1d7"]
    capabilities = Gr00tEngine.capabilities("gr00t_n1d7")
    assert capabilities.training_accelerators == ("cu128",)
    assert capabilities.export_formats == ("native", "openvino")
    assert capabilities.enablements == ("robotics-ai-suite-openvino",)


def test_config_renders_native_flags() -> None:
    config = Gr00tTrainConfig(
        dataset_path="/data/libero",
        output_dir="/runs/a",
        embodiment_tag="LIBERO_PANDA",
        max_steps=5,
        save_only_model=True,
        backbone_source="Qwen/Qwen3-VL-2B-Instruct",
        extra_args=("--warmup-ratio", "0.1"),
    )
    argv = config.to_argv()
    assert argv[:6] == ["--dataset-path", "/data/libero", "--output-dir", "/runs/a", "--embodiment-tag", "LIBERO_PANDA"]
    assert "--save-only-model" in argv
    assert "--no-tune-llm" in argv
    assert "--tune-projector" in argv
    assert "--modality-config-path" not in argv
    assert "--backbone-source" not in " ".join(argv)
    assert argv[-2:] == ["--warmup-ratio", "0.1"]


def test_settings_schema_and_studio_paths() -> None:
    schema = Gr00tEngine.config_schema("gr00t_n1d7")
    assert set(schema["required"]) == {"dataset_path", "output_dir", "embodiment_tag"}
    assert schema["properties"]["max_steps"] == {"type": "integer", "default": 10000}
    assert schema["properties"]["modality_config_path"]["type"] == ["string", "null"]
    assert schema["properties"]["extra_args"]["type"] == "array"
    config = Gr00tTrainConfig.from_settings(
        {"embodiment_tag": "LIBERO_PANDA", "max_steps": 3},
        dataset=Path("/d"),
        output_dir=Path("/o"),
    )
    assert (config.dataset_path, config.output_dir, config.max_steps) == ("/d", "/o", 3)
    with pytest.raises(ValueError, match="Unknown GR00T settings"):
        Gr00tTrainConfig.from_settings({"embodiment_tag": "x", "lr": 1}, dataset=Path("/d"), output_dir=Path("/o"))
    with pytest.raises(ValueError, match="Unknown gr00t policy"):
        Gr00tEngine.config_schema("act")


def test_training_runs_the_native_worker_through_core_and_records_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run"
    checkpoint.mkdir()
    (checkpoint / "physicalai_run.json").write_text('{"embodiment_tag": "libero_sim"}', encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def fake_run_worker(env: object, argv: list[str], **kwargs: Any) -> Run:  # ruff: ignore[any-type]
        calls.append({"env": env, "argv": argv, **kwargs})
        return Run(framework="gr00t", output_dir=checkpoint, checkpoint=checkpoint)

    monkeypatch.setattr(engine, "run_worker", fake_run_worker)
    run = Gr00tEngine(device="cuda:1").train(
        Gr00tTrainConfig(
            dataset_path="/d",
            output_dir=str(checkpoint),
            embodiment_tag="LIBERO_PANDA",
            backbone_source="Qwen/Qwen3-VL-2B-Instruct",
        ),
    )

    assert run.checkpoint == checkpoint
    assert calls[0]["env"] is engine.NATIVE
    assert calls[0]["worker"] is environments.TRAIN_WORKER
    assert calls[0]["interpreter"] is None
    assert calls[0]["extra_env"] == {
        "CUDA_VISIBLE_DEVICES": "1",
        "PHYSICALAI_GR00T_BACKBONE_SOURCE": "Qwen/Qwen3-VL-2B-Instruct",
    }
    record = json.loads((checkpoint / "physicalai_run.json").read_text(encoding="utf-8"))
    assert record["environment"] == environments.provenance(environments.NATIVE)
    with pytest.raises(ValueError, match="requires CUDA"):
        Gr00tEngine.run({"embodiment_tag": "x"}, dataset=Path("/d"), output_dir=Path("/o"), device="xpu")


def test_cli_passes_native_arguments_and_the_checkout_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(
        environments,
        "resolve",
        lambda _env, **_: (Path("/envs/default/python"), Path("/envs/default/project"), True),
    )
    monkeypatch.setattr(engine.subprocess, "call", lambda command, *, env, **_: seen.append((command, env)) or 3)

    assert Gr00tEngine.cli(["--help"]) == 3
    command, variables = seen[0]
    assert command[0] == "/envs/default/python"
    assert command[1].endswith("gr00t_train_worker.py")
    assert command[2:] == ["--help"]
    assert variables["PHYSICALAI_WORKER_PROJECT"] == "/envs/default/project"


def test_openvino_export_runs_the_patched_environment_and_packages_its_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "modality.json").write_text(json.dumps(LIBERO_MODALITY), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_name": "nvidia/Cosmos-Reason2-2B"}', encoding="utf-8")
    requests: list[dict[str, Any]] = []

    def fake_run_worker(env: object, argv: list[str], **kwargs: Any) -> Run:  # ruff: ignore[any-type]
        assert env is engine.OPENVINO
        assert kwargs["worker"] is environments.EXPORT_WORKER
        request = json.loads(argv[0])
        requests.append(request)
        ov_dir = Path(request["output_dir"])
        (ov_dir / "config").mkdir(parents=True)
        (ov_dir / "model_fp16.xml").write_text("<net/>", encoding="utf-8")
        (ov_dir / "physicalai_export.json").write_text(
            json.dumps({"precision": "fp16", "llm_lang_tokens": 64, "verification": {"cosine": 1.0}}),
            encoding="utf-8",
        )
        return Run(framework="gr00t", output_dir=ov_dir, checkpoint=None)

    monkeypatch.setattr(engine, "run_worker", fake_run_worker)
    output = Gr00tEngine.export_openvino(
        checkpoint,
        tmp_path / "package",
        dataset_path=dataset,
        embodiment_tag="LIBERO_PANDA",
    )

    artifact = output / "gr00t_n1d7.gr00t"
    descriptor = json.loads((artifact / "package.json").read_text(encoding="utf-8"))
    assert requests[0]["embodiment_tag"] == "LIBERO_PANDA"
    assert requests[0]["dataset_path"] == str(dataset)
    assert descriptor["environment"] == environments.provenance(environments.OPENVINO)
    assert descriptor["export"]["verification"] == {"cosine": 1.0}
    assert descriptor["backbone_source"] == "nvidia/Cosmos-Reason2-2B"
    assert not (artifact / "openvino" / "physicalai_export.json").exists()


def test_unknown_export_backends_are_rejected() -> None:
    with pytest.raises(ValueError, match="'native' or 'openvino'"):
        Gr00tEngine.export("/runs/a", "/exports/a", backend="onnx")
