# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for converting old pi05 OpenVINO exports to take the start noise as an input."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
import onnx
import openvino as ov
import pytest
from onnx import TensorProto, helper
from physicalai.policies.pi05.noise_input import (
    BACKUP_SUFFIX,
    ConversionStatus,
    convert_openvino_export,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path

SHAPE = [1, 4, 3]
PI05_SPEC = {"type": "pi05", "image_resolution": [224, 224], "empty_cameras": 0}


def _write_export(export_dir: Path, *, random_noise: bool = True) -> Path:
    """Write a tiny OpenVINO export shaped like an old pi05 one: ``y = x + noise``.

    The noise comes from an ONNX ``RandomNormalLike`` converted by OpenVINO, which
    builds the same single-RandomUniform Box-Muller subgraph as the real pi05 export.
    """
    export_dir.mkdir(parents=True)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, SHAPE)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, SHAPE)
    if random_noise:
        nodes = [helper.make_node("RandomNormalLike", ["x"], ["n"]), helper.make_node("Add", ["x", "n"], ["y"])]
    else:
        nodes = [helper.make_node("Identity", ["x"], ["y"])]
    graph = helper.make_graph(nodes, "g", [x], [y])
    onnx_path = export_dir.parent / f"{export_dir.name}.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), onnx_path)
    ov.save_model(ov.convert_model(str(onnx_path)), export_dir / "pi05.xml")
    manifest = {
        "model": {
            "artifacts": {"openvino": "pi05.xml"},
            "preprocessors": [{"type": "normalize", "mode": "quantiles"}, dict(PI05_SPEC)],
        },
    }
    (export_dir / "manifest.json").write_text(json.dumps(manifest))
    (export_dir / "tokenizer.xml").write_text("<tokenizer/>")
    return export_dir


def _op_types(graph_path: Path) -> set[str]:
    return {op.get_type_name() for op in ov.Core().read_model(graph_path).get_ordered_ops()}


def _pi05_spec(export_dir: Path) -> dict:
    manifest = json.loads((export_dir / "manifest.json").read_text())
    return next(spec for spec in manifest["model"]["preprocessors"] if spec["type"] == "pi05")


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    return _write_export(tmp_path / "export")


class TestConvertOpenVINOExport:
    """Converting an export whose graph samples the start noise."""

    def test_in_place_takes_noise_input(self, export_dir: Path) -> None:
        result = convert_openvino_export(export_dir)

        assert result.status is ConversionStatus.CONVERTED
        assert result.noise_shape == (4, 3)
        assert "RandomUniform" in result.replaced_ops[0]
        assert "RandomUniform" not in _op_types(export_dir / "pi05.xml")
        assert _pi05_spec(export_dir) == {**PI05_SPEC, "noise_shape": [4, 3]}

        compiled = ov.Core().compile_model(export_dir / "pi05.xml", "CPU")
        x = np.ones(SHAPE, np.float32)
        noise = np.arange(12, dtype=np.float32).reshape(SHAPE)
        np.testing.assert_array_equal(compiled({"x": x, "noise": noise})[0], x + noise)

    def test_in_place_keeps_backup(self, export_dir: Path) -> None:
        result = convert_openvino_export(export_dir)

        backup = export_dir.with_name(export_dir.name + BACKUP_SUFFIX)
        assert result.backup_dir == backup
        assert sorted(path.name for path in backup.iterdir()) == ["manifest.json", "pi05.bin", "pi05.xml"]
        assert "RandomUniform" in _op_types(backup / "pi05.xml")
        assert "noise_shape" not in _pi05_spec(backup)
        assert sorted(path.name for path in export_dir.iterdir()) == [
            "manifest.json",
            "pi05.bin",
            "pi05.xml",
            "tokenizer.xml",
        ]

    def test_backup_dir(self, export_dir: Path, tmp_path: Path) -> None:
        convert_openvino_export(export_dir, backup_dir=tmp_path / "kept")

        assert "RandomUniform" in _op_types(tmp_path / "kept" / "pi05.xml")
        assert not export_dir.with_name(export_dir.name + BACKUP_SUFFIX).exists()

    def test_no_backup(self, export_dir: Path, tmp_path: Path) -> None:
        result = convert_openvino_export(export_dir, backup=False, backup_dir=tmp_path / "kept")

        assert result.backup_dir is None

        assert not export_dir.with_name(export_dir.name + BACKUP_SUFFIX).exists()
        assert not (tmp_path / "kept").exists()

    def test_output_dir_leaves_source_untouched(self, export_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "converted"
        result = convert_openvino_export(export_dir, out)

        assert result.export_dir == out
        assert "RandomUniform" in _op_types(export_dir / "pi05.xml")
        assert "noise_shape" not in _pi05_spec(export_dir)
        assert "RandomUniform" not in _op_types(out / "pi05.xml")
        assert _pi05_spec(out)["noise_shape"] == [4, 3]
        assert (out / "tokenizer.xml").read_text() == "<tokenizer/>"

    def test_refreshes_existing_checksums(self, export_dir: Path) -> None:
        names = ["manifest.json", "pi05.bin", "pi05.xml", "tokenizer.xml"]
        (export_dir / "SHA256SUMS").write_text("".join(f"{'0' * 64}  {name}\n" for name in names))

        convert_openvino_export(export_dir)

        rows = [line.split() for line in (export_dir / "SHA256SUMS").read_text().splitlines()]
        listed = {name: digest for digest, name in rows}
        for name in names[:3]:
            assert listed[name] == hashlib.sha256((export_dir / name).read_bytes()).hexdigest()
        assert listed["tokenizer.xml"] == "0" * 64

    def test_second_run_changes_nothing(self, export_dir: Path) -> None:
        convert_openvino_export(export_dir)
        before = {path.name: path.read_bytes() for path in export_dir.iterdir()}

        result = convert_openvino_export(export_dir)

        assert result.status is ConversionStatus.ALREADY_CONVERTED
        assert result.noise_shape == (4, 3)
        assert {path.name: path.read_bytes() for path in export_dir.iterdir()} == before

    def test_main(self, export_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main([str(export_dir), "--no-backup"])

        assert capsys.readouterr().out.startswith("converted:")
        assert not export_dir.with_name(export_dir.name + BACKUP_SUFFIX).exists()


class TestConvertOpenVINOExportErrors:
    """Exports the converter refuses, without writing anything."""

    def test_graph_without_noise(self, tmp_path: Path) -> None:
        export_dir = _write_export(tmp_path / "export", random_noise=False)

        with pytest.raises(ValueError, match="samples no start noise"):
            convert_openvino_export(export_dir)

    def test_manifest_without_pi05(self, export_dir: Path) -> None:
        manifest = json.loads((export_dir / "manifest.json").read_text())
        manifest["model"]["preprocessors"] = manifest["model"]["preprocessors"][:1]
        (export_dir / "manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="one `pi05` preprocessor"):
            convert_openvino_export(export_dir)

    def test_not_an_export(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"no manifest\.json"):
            convert_openvino_export(tmp_path)

    def test_output_dir_not_empty(self, export_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "converted"
        out.mkdir()
        (out / "keep.txt").write_text("x")

        with pytest.raises(FileExistsError, match="not empty"):
            convert_openvino_export(export_dir, out)

    def test_existing_backup(self, export_dir: Path) -> None:
        backup = export_dir.with_name(export_dir.name + BACKUP_SUFFIX)
        backup.mkdir()
        (backup / "pi05.xml").write_text("older backup")

        with pytest.raises(FileExistsError, match="not empty"):
            convert_openvino_export(export_dir)
        assert "RandomUniform" in _op_types(export_dir / "pi05.xml")
