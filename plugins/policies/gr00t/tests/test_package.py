# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Runtime package writing, reading, and pin checks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from physicalai_gr00t_policy_plugin import environments, package
from physicalai_gr00t_policy_plugin.package import IoSpec

NATIVE = environments.provenance(environments.NATIVE)
OPENVINO = environments.provenance(environments.OPENVINO)

if TYPE_CHECKING:
    from pathlib import Path

LIBERO_MODALITY = {
    "state": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 8},
    },
    "action": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 7},
    },
    "video": {
        "image": {"original_key": "observation.images.image"},
        "wrist_image": {"original_key": "observation.images.wrist_image"},
    },
    "annotation": {"human.action.task_description": {"original_key": "task_index"}},
}


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "modality.json").write_text(json.dumps(LIBERO_MODALITY), encoding="utf-8")
    return root


@pytest.fixture
def checkpoint(tmp_path: Path, dataset: Path) -> Path:
    root = tmp_path / "run"
    (root / "processor").mkdir(parents=True)
    (root / "checkpoint-3").mkdir()
    (root / "experiment_cfg").mkdir()
    (root / "config.json").write_text('{"model_type": "Gr00tN1d7"}', encoding="utf-8")
    (root / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (root / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (root / "training_args.bin").write_bytes(b"\x80\x04pickle")
    (root / "checkpoint-3" / "model-00001-of-00001.safetensors").write_bytes(b"duplicate")
    (root / "experiment_cfg" / "config.yaml").write_text("a: 1\n", encoding="utf-8")
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        (root / "processor" / name).write_text("{}", encoding="utf-8")
    run_info = {
        "embodiment_tag": "libero_sim",
        "dataset_path": str(dataset),
        "backbone_source": "nvidia/Cosmos-Reason2-2B",
        "backbone_override": False,
    }
    (root / package.RUN_INFO_NAME).write_text(json.dumps(run_info), encoding="utf-8")
    return root


def test_io_spec_derives_cameras_and_slices_from_modality(dataset: Path) -> None:
    spec = IoSpec.from_modality(dataset / "meta" / "modality.json")
    assert spec.cameras == {"image": "image", "wrist_image": "wrist_image"}
    assert spec.state["gripper"] == (6, 8)
    assert spec.action["gripper"] == (6, 7)
    assert (spec.state_dim, spec.action_dim) == (8, 7)
    assert IoSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_io_spec_rejects_invalid_slices() -> None:
    with pytest.raises(ValueError, match="Invalid slice"):
        IoSpec.from_dict({"cameras": {}, "state": {"x": [2, 1]}, "action": {}})


def test_native_package_copies_only_allowlisted_model_files(checkpoint: Path, dataset: Path, tmp_path: Path) -> None:
    spec = IoSpec.from_modality(dataset / "meta" / "modality.json")
    output = package.write_native_package(
        checkpoint,
        tmp_path / "native",
        io=spec,
        run_info=package.read_run_info(checkpoint),
        environment=NATIVE,
    )
    artifact = output / package.ARTIFACT_NAME
    copied = sorted(path.relative_to(artifact / "model").as_posix() for path in (artifact / "model").rglob("*.*"))
    assert "training_args.bin" not in copied
    assert not any(name.startswith("checkpoint-") for name in copied)
    assert {"config.json", "model-00001-of-00001.safetensors", "processor/statistics.json"} <= set(copied)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["artifacts"] == {"gr00t": package.ARTIFACT_NAME}
    assert manifest["model"]["callbacks"][0]["class_path"].endswith("Gr00tResetCallback")
    descriptor = package.load_descriptor(artifact)
    assert descriptor["format"] == package.FORMAT_NATIVE
    assert descriptor["embodiment_tag"] == "libero_sim"
    assert descriptor["environment"] == NATIVE
    assert descriptor["environment"]["source"]["patches"] == []


def test_openvino_package_records_export_and_completes_processor_config(
    checkpoint: Path,
    dataset: Path,
    tmp_path: Path,
) -> None:
    def export(ov_dir: Path) -> dict:
        (ov_dir / "config").mkdir(parents=True)
        (ov_dir / "model_fp16.xml").write_text("<net/>", encoding="utf-8")
        (ov_dir / "model_fp16.bin").write_bytes(b"ir")
        (ov_dir / "config" / "config.json").write_text("{}", encoding="utf-8")
        return {"precision": "fp16", "verification": {"cosine": 0.999999, "mae": 3.7e-5}, "llm_lang_tokens": 64}

    output = package.write_openvino_package(
        export,
        tmp_path / "openvino",
        checkpoint=checkpoint,
        io=IoSpec.from_modality(dataset / "meta" / "modality.json"),
        run_info=package.read_run_info(checkpoint),
        environment=OPENVINO,
    )
    artifact = output / package.ARTIFACT_NAME
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        assert (artifact / "openvino" / "config" / name).is_file()
    descriptor = package.load_descriptor(artifact)
    assert descriptor["format"] == package.FORMAT_OPENVINO
    assert descriptor["export"]["ir"] == ["model_fp16.xml"]
    assert descriptor["export"]["verification"]["cosine"] == pytest.approx(0.999999)
    patches = descriptor["environment"]["source"]["patches"][0]
    assert descriptor["environment"]["name"] == "robotics-ai-suite-openvino"
    assert patches["tree"] == "d657bcb89b8937495b8c849d7cfb8c9f3d1e5601"
    assert len(patches["files"]) == 8


def test_upstream_released_checkpoints_keep_root_processor_files(
    checkpoint: Path,
    dataset: Path,
    tmp_path: Path,
) -> None:
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        (checkpoint / "processor" / name).rename(checkpoint / name)
    (checkpoint / "processor").rmdir()
    output = package.write_native_package(
        checkpoint,
        tmp_path / "native",
        io=IoSpec.from_modality(dataset / "meta" / "modality.json"),
        run_info={"embodiment_tag": "LIBERO_PANDA"},
        environment=NATIVE,
    )
    model = output / package.ARTIFACT_NAME / "model"
    assert {"processor_config.json", "statistics.json", "embodiment_id.json"} <= {path.name for path in model.iterdir()}


def test_descriptors_without_an_environment_are_rejected(checkpoint: Path, dataset: Path, tmp_path: Path) -> None:
    output = package.write_native_package(
        checkpoint,
        tmp_path / "native",
        io=IoSpec.from_modality(dataset / "meta" / "modality.json"),
        run_info=package.read_run_info(checkpoint),
        environment=NATIVE,
    )
    descriptor_path = output / package.ARTIFACT_NAME / package.DESCRIPTOR_NAME
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    del descriptor["environment"]
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(ValueError, match="does not record its worker environment"):
        package.load_descriptor(output / package.ARTIFACT_NAME)


def test_existing_outputs_are_not_overwritten(checkpoint: Path, dataset: Path, tmp_path: Path) -> None:
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError):
        package.write_native_package(
            checkpoint,
            tmp_path / "taken",
            io=IoSpec.from_modality(dataset / "meta" / "modality.json"),
            run_info={},
            environment=NATIVE,
        )


def test_symlinked_checkpoint_files_are_rejected(checkpoint: Path, dataset: Path, tmp_path: Path) -> None:
    (checkpoint / "model-00002-of-00002.safetensors").symlink_to(checkpoint / "model-00001-of-00001.safetensors")
    with pytest.raises(ValueError, match="symbolic link"):
        package.write_native_package(
            checkpoint,
            tmp_path / "native",
            io=IoSpec.from_modality(dataset / "meta" / "modality.json"),
            run_info={},
            environment=NATIVE,
        )
