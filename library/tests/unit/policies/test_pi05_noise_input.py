# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for converting old pi05 OpenVINO exports to take the start noise as an input."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import onnx
import openvino as ov
import pytest
from onnx import TensorProto, helper
from physicalai.policies.pi05.noise_input import (
    BACKUP_SUFFIX,
    ConversionStatus,
    convert_openvino_export,
    has_backup,
    main,
    RollbackError,
    read_noise_shape,
    restore_openvino_export,
)
from physicalai.policies.pi05 import noise_input


SHAPE = [1, 4, 3]
PI05_SPEC = {"type": "pi05", "image_resolution": [224, 224], "empty_cameras": 0}


def _scalar(name: str, value: float) -> onnx.NodeProto:
    return helper.make_node("Constant", [], [name], value=helper.make_tensor(name, TensorProto.FLOAT, [], [value]))


# Graphs that look like Box-Muller from the op types alone, with u = RandomUniformLike(x).
NEAR_BOX_MULLER = {
    "log_times_cos": [
        helper.make_node("Log", ["u"], ["l"]),
        helper.make_node("Cos", ["u"], ["c"]),
        helper.make_node("Mul", ["l", "c"], ["n"]),
    ],
    "wrong_scale": [
        _scalar("k", -1.0),
        _scalar("tau", 2 * math.pi),
        helper.make_node("Log", ["u"], ["l"]),
        helper.make_node("Mul", ["l", "k"], ["lk"]),
        helper.make_node("Sqrt", ["lk"], ["r"]),
        helper.make_node("Mul", ["tau", "u"], ["a"]),
        helper.make_node("Cos", ["a"], ["c"]),
        helper.make_node("Mul", ["r", "c"], ["n"]),
    ],
}


def _write_export(export_dir: Path, *, graph: str = "normal", pi05_spec: dict | None = None) -> Path:
    """Write a tiny OpenVINO export shaped like an old pi05 one: ``y = x + noise``.

    The noise comes from an ONNX ``RandomNormalLike`` converted by OpenVINO, which
    builds the same single-RandomUniform Box-Muller subgraph as the real pi05 export.
    ``graph`` picks a variant: ``"cast"`` also outputs the noise cast to fp16,
    ``"uniform"`` adds uniform noise (no Box-Muller), ``"none"`` has no noise, and
    the ``NEAR_BOX_MULLER`` names build ``n`` from ``u`` almost like Box-Muller.
    """
    export_dir.mkdir(parents=True)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, SHAPE)
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, SHAPE)]
    noise_op = "RandomUniformLike" if graph == "uniform" else "RandomNormalLike"
    nodes = [helper.make_node(noise_op, ["x"], ["n"]), helper.make_node("Add", ["x", "n"], ["y"])]
    if graph in NEAR_BOX_MULLER:
        nodes = [helper.make_node("RandomUniformLike", ["x"], ["u"]), *NEAR_BOX_MULLER[graph], nodes[1]]
    if graph == "cast":
        nodes.append(helper.make_node("Cast", ["n"], ["z"], to=TensorProto.FLOAT16))
        outputs.append(helper.make_tensor_value_info("z", TensorProto.FLOAT16, SHAPE))
    if graph == "none":
        nodes = [helper.make_node("Identity", ["x"], ["y"])]
    onnx_graph = helper.make_graph(nodes, "g", [x], outputs)
    onnx_path = export_dir.parent / f"{export_dir.name}.onnx"
    onnx.save(helper.make_model(onnx_graph, opset_imports=[helper.make_opsetid("", 17)]), onnx_path)
    ov.save_model(ov.convert_model(str(onnx_path)), export_dir / "pi05.xml")
    manifest = {
        "model": {
            "artifacts": {"openvino": "pi05.xml"},
            "preprocessors": [{"type": "normalize", "mode": "quantiles"}, pi05_spec or dict(PI05_SPEC)],
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
        assert any("RandomUniform" in name for name in result.replaced_ops)
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

    @pytest.mark.parametrize("prefix", ["", "./"])
    def test_refreshes_existing_checksums(self, export_dir: Path, prefix: str) -> None:
        names = ["manifest.json", "pi05.bin", "pi05.xml", "tokenizer.xml"]
        (export_dir / "SHA256SUMS").write_text("".join(f"{'0' * 64}  {prefix}{name}\n" for name in names))

        convert_openvino_export(export_dir)

        rows = [line.split() for line in (export_dir / "SHA256SUMS").read_text().splitlines()]
        listed = {name: digest for digest, name in rows}
        for name in names[:3]:
            assert listed[prefix + name] == hashlib.sha256((export_dir / name).read_bytes()).hexdigest()
        assert listed[prefix + "tokenizer.xml"] == "0" * 64

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
        export_dir = _write_export(tmp_path / "export", graph="none")

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


class TestRestoreOpenVINOExport:
    """Undoing an in-place conversion."""

    def test_restores_original_files(self, export_dir: Path) -> None:
        original = {path.name: path.read_bytes() for path in export_dir.iterdir()}
        convert_openvino_export(export_dir)
        assert has_backup(export_dir)

        backup = restore_openvino_export(export_dir)

        assert {path.name: path.read_bytes() for path in export_dir.iterdir()} == original
        assert not backup.exists()
        assert not has_backup(export_dir)

    def test_restores_from_backup_dir_with_checksums(self, export_dir: Path, tmp_path: Path) -> None:
        checksums = f"{hashlib.sha256((export_dir / 'pi05.xml').read_bytes()).hexdigest()}  pi05.xml\n"
        (export_dir / "SHA256SUMS").write_text(checksums)
        convert_openvino_export(export_dir, backup_dir=tmp_path / "kept")
        assert (export_dir / "SHA256SUMS").read_text() != checksums

        restore_openvino_export(export_dir, tmp_path / "kept")

        assert (export_dir / "SHA256SUMS").read_text() == checksums
        assert "RandomUniform" in _op_types(export_dir / "pi05.xml")
        assert not (tmp_path / "kept").exists()

    def test_restores_into_an_export_missing_files(self, export_dir: Path) -> None:
        original = (export_dir / "pi05.xml").read_bytes()
        convert_openvino_export(export_dir)
        (export_dir / "pi05.xml").unlink()

        restore_openvino_export(export_dir)

        assert (export_dir / "pi05.xml").read_bytes() == original

    def test_relative_export_path(self, export_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(export_dir)

        result = convert_openvino_export(".")
        restore_openvino_export(".")

        assert result.backup_dir == export_dir.with_name(export_dir.name + BACKUP_SUFFIX)
        assert "RandomUniform" in _op_types(export_dir / "pi05.xml")

    def test_without_backup(self, export_dir: Path) -> None:
        convert_openvino_export(export_dir, backup=False)

        with pytest.raises(FileNotFoundError, match="no complete backup"):
            restore_openvino_export(export_dir)

    def test_incomplete_backup_changes_nothing(self, export_dir: Path) -> None:
        convert_openvino_export(export_dir)
        backup = export_dir.with_name(export_dir.name + BACKUP_SUFFIX)
        (backup / "pi05.bin").unlink()

        with pytest.raises(FileNotFoundError, match="no complete backup"):
            restore_openvino_export(export_dir)
        assert "RandomUniform" not in _op_types(export_dir / "pi05.xml")

    def test_main_restore(self, export_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        convert_openvino_export(export_dir)

        main([str(export_dir), "--restore"])

        assert capsys.readouterr().out.startswith("restored:")
        assert "RandomUniform" in _op_types(export_dir / "pi05.xml")


class TestConversionGraphs:
    """Which graphs the converter accepts, and that it keeps the model's own ops."""

    def test_keeps_ops_after_the_noise(self, tmp_path: Path) -> None:
        export_dir = _write_export(tmp_path / "export", graph="cast")

        convert_openvino_export(export_dir)

        compiled = ov.Core().compile_model(export_dir / "pi05.xml", "CPU")
        x = np.ones(SHAPE, np.float32)
        noise = (np.arange(12, dtype=np.float32) * 0.25 - 1.5).reshape(SHAPE)  # exact in fp16
        result = compiled({"x": x, "noise": noise})
        np.testing.assert_array_equal(result["y"], x + noise)
        np.testing.assert_array_equal(result["z"], noise.astype(np.float16))

    @pytest.mark.parametrize("graph", ["uniform", *NEAR_BOX_MULLER])
    def test_rejects_noise_that_is_not_box_muller(self, tmp_path: Path, graph: str) -> None:
        export_dir = _write_export(tmp_path / "export", graph=graph)

        with pytest.raises(ValueError, match="not the Box-Muller pattern"):
            convert_openvino_export(export_dir)
        assert sorted(path.name for path in export_dir.iterdir()) == [
            "manifest.json",
            "pi05.bin",
            "pi05.xml",
            "tokenizer.xml",
        ]


class TestConversionManifests:
    """The runtime reads ``init_args`` for a ``class_path`` spec and the flat fields otherwise."""

    @pytest.mark.parametrize(
        "spec",
        [
            {"class_path": "physicalai.inference.preprocessors.pi05.Pi05Preprocessor", "init_args": {}},
            {"type": "pi05", "class_path": "physicalai.inference.preprocessors.pi05.Pi05Preprocessor"},
        ],
    )
    def test_class_path_spec(self, tmp_path: Path, spec: dict) -> None:
        from physicalai.inference.component_factory import instantiate_component
        from physicalai.inference.manifest import ComponentSpec

        export_dir = _write_export(tmp_path / "export", pi05_spec=spec)

        convert_openvino_export(export_dir)

        written = json.loads((export_dir / "manifest.json").read_text())["model"]["preprocessors"][1]
        assert written["init_args"]["noise_shape"] == [4, 3]
        assert "noise_shape" not in {k for k in written if k != "init_args"}
        assert read_noise_shape(export_dir) == [4, 3]
        preprocessor = instantiate_component(ComponentSpec(**written))
        assert preprocessor._noise_shape == (4, 3)

    def test_rejects_artifact_outside_the_folder(self, tmp_path: Path) -> None:
        export_dir = _write_export(tmp_path / "export")
        manifest = json.loads((export_dir / "manifest.json").read_text())
        manifest["model"]["artifacts"]["openvino"] = "../outside.xml"
        (export_dir / "manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="must be a file name in the folder"):
            convert_openvino_export(export_dir)

    def test_output_inside_the_export(self, export_dir: Path) -> None:
        with pytest.raises(ValueError, match="must be outside the export"):
            convert_openvino_export(export_dir, export_dir / "fixed")
        assert not (export_dir / "fixed").exists()


class TestFailures:
    """A failure, or the process dying, never leaves the export without a complete set of files."""

    @staticmethod
    def _fail_replace(monkeypatch: pytest.MonkeyPatch, source_parent: str, name: str) -> None:
        real_replace = Path.replace

        def replace(self: Path, target: Path) -> Path:
            if self.parent.name == source_parent and self.name == name:
                msg = "disk full"
                raise OSError(msg)
            return real_replace(self, target)

        monkeypatch.setattr(Path, "replace", replace)

    @pytest.mark.parametrize("backup", [True, False])
    def test_conversion_rolls_back(self, export_dir: Path, monkeypatch: pytest.MonkeyPatch, backup: bool) -> None:
        original = {path.name: path.read_bytes() for path in export_dir.iterdir()}
        self._fail_replace(monkeypatch, ".noise-input-new", "pi05.bin")

        with pytest.raises(OSError, match="disk full"):
            convert_openvino_export(export_dir, backup=backup)

        assert {path.name: path.read_bytes() for path in export_dir.iterdir()} == original
        assert not export_dir.with_name(export_dir.name + BACKUP_SUFFIX).exists()

    @pytest.mark.parametrize("backup", [True, False])
    def test_failed_rollback_keeps_the_originals(
        self,
        export_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        backup: bool,
    ) -> None:
        original = {path.name: path.read_bytes() for path in export_dir.iterdir()}
        real_replace = Path.replace

        def replace(self: Path, target: Path) -> Path:
            if (self.parent.name, self.name) in {(".noise-input-new", "pi05.bin"), (".noise-input-swap", "pi05.xml")}:
                msg = "disk full"
                raise OSError(msg)
            return real_replace(self, target)

        monkeypatch.setattr(Path, "replace", replace)

        with pytest.raises(RollbackError, match="copy them back"):
            convert_openvino_export(export_dir, backup=backup)

        kept = export_dir.with_name(export_dir.name + BACKUP_SUFFIX) if backup else export_dir / ".noise-input-old"
        assert {name: (kept / name).read_bytes() for name in ("pi05.xml", "pi05.bin", "manifest.json")} == {
            name: original[name] for name in ("pi05.xml", "pi05.bin", "manifest.json")
        }

    def test_checksum_failure_changes_nothing(self, export_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (export_dir / "SHA256SUMS").write_text(f"{'0' * 64}  pi05.xml\n")
        original = {path.name: path.read_bytes() for path in export_dir.iterdir()}

        def fail(*_args: object) -> None:
            msg = "disk full"
            raise OSError(msg)

        monkeypatch.setattr(noise_input, "_write_checksums", fail)

        with pytest.raises(OSError, match="disk full"):
            convert_openvino_export(export_dir, backup=False)

        assert {path.name: path.read_bytes() for path in export_dir.iterdir()} == original

    def test_restore_failure_keeps_the_backup(self, export_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        convert_openvino_export(export_dir)
        self._fail_replace(monkeypatch, ".noise-input-swap", "pi05.bin")

        with pytest.raises(OSError, match="disk full"):
            restore_openvino_export(export_dir)
        monkeypatch.undo()

        assert has_backup(export_dir)
        restore_openvino_export(export_dir)
        assert "RandomUniform" in _op_types(export_dir / "pi05.xml")

    def test_killed_during_conversion(self, export_dir: Path) -> None:
        """The process dies after the first file is swapped in: the backup is complete and restores."""
        original = {path.name: path.read_bytes() for path in export_dir.iterdir()}
        script = textwrap.dedent(
            f"""
            import os
            from pathlib import Path
            from physicalai.policies.pi05.noise_input import convert_openvino_export

            real_replace = Path.replace

            def replace(self, target):
                result = real_replace(self, target)
                if self.parent.name == ".noise-input-new":
                    os._exit(3)
                return result

            Path.replace = replace
            convert_openvino_export({str(export_dir)!r})
            """,
        )
        killed = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True)  # noqa: S603
        assert killed.returncode == 3, killed.stderr.decode()

        assert sorted(original) == sorted(path.name for path in export_dir.iterdir() if path.is_file())
        assert has_backup(export_dir)
        restore_openvino_export(export_dir)
        assert {path.name: path.read_bytes() for path in export_dir.iterdir() if path.is_file()} == original

    def test_killed_during_restore_then_rerun(self, export_dir: Path) -> None:
        """The process dies after the first file is restored: the first rerun succeeds."""
        original = {path.name: path.read_bytes() for path in export_dir.iterdir()}
        convert_openvino_export(export_dir)
        script = textwrap.dedent(
            f"""
            import os
            from pathlib import Path
            from physicalai.policies.pi05.noise_input import restore_openvino_export

            real_replace = Path.replace

            def replace(self, target):
                result = real_replace(self, target)
                if self.parent.name == ".noise-input-swap":
                    os._exit(3)
                return result

            Path.replace = replace
            restore_openvino_export({str(export_dir)!r})
            """,
        )
        killed = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True)  # noqa: S603
        assert killed.returncode == 3, killed.stderr.decode()

        restore_openvino_export(export_dir)

        assert {path.name: path.read_bytes() for path in export_dir.iterdir() if path.is_file()} == original

    def test_restore_from_the_export_itself(self, export_dir: Path) -> None:
        convert_openvino_export(export_dir)
        converted = {path.name: path.read_bytes() for path in export_dir.iterdir()}

        with pytest.raises(ValueError, match="must be outside the export"):
            restore_openvino_export(export_dir, backup_dir=export_dir)

        assert {path.name: path.read_bytes() for path in export_dir.iterdir()} == converted

    def test_checksums_symlink_target_is_not_written(self, export_dir: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside_sums"
        outside.write_text(f"{'0' * 64}  pi05.xml\n")
        (export_dir / "SHA256SUMS").symlink_to(outside)

        convert_openvino_export(export_dir)
        restore_openvino_export(export_dir)

        assert outside.read_text() == f"{'0' * 64}  pi05.xml\n"

    def test_backup_inside_the_export(self, export_dir: Path) -> None:
        original = {path.name: path.read_bytes() for path in export_dir.iterdir()}

        with pytest.raises(ValueError, match="must be outside the export"):
            convert_openvino_export(export_dir, backup_dir=export_dir / ".noise-input-new")

        assert {path.name: path.read_bytes() for path in export_dir.iterdir()} == original

    def test_backup_with_artifact_outside_the_folder(self, export_dir: Path, tmp_path: Path) -> None:
        convert_openvino_export(export_dir)
        backup = export_dir.with_name(export_dir.name + BACKUP_SUFFIX)
        manifest = json.loads((backup / "manifest.json").read_text())
        manifest["model"]["artifacts"]["openvino"] = "../outside.xml"
        (backup / "manifest.json").write_text(json.dumps(manifest))
        (tmp_path / "outside.xml").write_text("victim")

        assert not has_backup(export_dir)
        with pytest.raises(FileNotFoundError, match="no complete backup"):
            restore_openvino_export(export_dir)
        assert (tmp_path / "outside.xml").read_text() == "victim"
