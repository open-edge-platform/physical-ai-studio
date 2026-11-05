import importlib
import json
from pathlib import Path
from uuid import uuid4

import onnx
import openvino as ov
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper

# cli/__init__.py binds `models` to the click group, so import the module itself.
models_cli = importlib.import_module("cli.models")


def _write_old_pi05_export(export_dir: Path) -> Path:
    """Tiny OpenVINO export that samples its noise in-graph like an old pi05 export: y = x + noise."""
    export_dir.mkdir(parents=True)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 3])
    nodes = [helper.make_node("RandomNormalLike", ["x"], ["n"]), helper.make_node("Add", ["x", "n"], ["y"])]
    onnx_path = export_dir / "tiny.onnx"
    onnx.save(helper.make_model(helper.make_graph(nodes, "g", [x], [y])), onnx_path)
    ov.save_model(ov.convert_model(str(onnx_path)), export_dir / "pi05.xml")
    onnx_path.unlink()
    manifest = {"model": {"artifacts": {"openvino": "pi05.xml"}, "preprocessors": [{"type": "pi05"}]}}
    (export_dir / "manifest.json").write_text(json.dumps(manifest))
    return export_dir


def _input_names(export_dir: Path) -> set[str]:
    return {name for inp in ov.Core().read_model(export_dir / "pi05.xml").inputs for name in inp.get_names()}


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    _write_old_pi05_export(model_dir / "exports" / "openvino")
    return model_dir


def test_converts_studio_model_by_id(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_id = uuid4()
    monkeypatch.setattr(models_cli, "_get_model_dir", lambda mid: model_dir if mid == model_id else None)

    result = CliRunner().invoke(models_cli.models, ["fix-pi05-noise", str(model_id)])

    assert result.exit_code == 0, result.output
    assert "Converted:" in result.output
    export_dir = model_dir / "exports" / "openvino"
    assert "noise" in _input_names(export_dir)
    assert "noise" not in _input_names(model_dir / models_cli.NOISE_BACKUP_DIRNAME)
    # The backup stays out of exports/, where Studio would count it as part of the model.
    assert sorted(path.name for path in (model_dir / "exports").iterdir()) == ["openvino"]


def test_converts_any_export_folder(tmp_path: Path) -> None:
    export_dir = _write_old_pi05_export(tmp_path / "my_export")

    result = CliRunner().invoke(models_cli.models, ["fix-pi05-noise", str(export_dir), "--no-backup"])

    assert result.exit_code == 0, result.output
    assert "noise" in _input_names(export_dir)
    assert not (tmp_path / "my_export.orig").exists()


def test_out_leaves_studio_model_untouched(model_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "converted"

    result = CliRunner().invoke(models_cli.models, ["fix-pi05-noise", str(model_dir), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert "noise" in _input_names(out)
    assert "noise" not in _input_names(model_dir / "exports" / "openvino")


def test_second_run_reports_already_converted(model_dir: Path) -> None:
    runner = CliRunner()
    runner.invoke(models_cli.models, ["fix-pi05-noise", str(model_dir)])

    result = runner.invoke(models_cli.models, ["fix-pi05-noise", str(model_dir)])

    assert result.exit_code == 0, result.output
    assert "Already takes the noise as input" in result.output


def test_rejects_unknown_target(tmp_path: Path) -> None:
    result = CliRunner().invoke(models_cli.models, ["fix-pi05-noise", str(tmp_path / "missing")])

    assert result.exit_code != 0
    assert "Not a folder or a model UUID" in result.output


def test_rejects_folder_without_export(tmp_path: Path) -> None:
    result = CliRunner().invoke(models_cli.models, ["fix-pi05-noise", str(tmp_path)])

    assert result.exit_code != 0
    assert "no manifest.json" in result.output


def test_lists_studio_models_and_converts_the_chosen_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old = tmp_path / "old"
    _write_old_pi05_export(old / "exports" / "openvino")
    fixed = tmp_path / "fixed"
    _write_old_pi05_export(fixed / "exports" / "openvino")
    CliRunner().invoke(models_cli.models, ["fix-pi05-noise", str(fixed)])
    other = tmp_path / "act"
    (other / "exports" / "openvino").mkdir(parents=True)
    (other / "exports" / "openvino" / "manifest.json").write_text(json.dumps({"model": {"preprocessors": []}}))
    no_export = tmp_path / "torch_only"
    no_export.mkdir()
    ids = [uuid4() for _ in range(4)]
    studio = [
        (ids[0], "pi05 old", "v1, 2026-10-28", old),
        (ids[1], "act", "v1", other),
        (ids[2], "pi05 fixed", "v2", fixed),
        (ids[3], "torch only", "v1", no_export),
    ]
    monkeypatch.setattr(models_cli, "_studio_models", lambda: studio)

    result = CliRunner().invoke(models_cli.models, ["fix-pi05-noise"], input="1\n")

    assert result.exit_code == 0, result.output
    assert f"1. pi05 old (v1, 2026-10-28)  [needs fix]  {ids[0]}" in result.output
    assert f"2. pi05 fixed (v2)  [already fixed]  {ids[2]}" in result.output
    assert "act" not in result.output
    assert "torch only" not in result.output
    assert "Converted:" in result.output
    assert "noise" in _input_names(old / "exports" / "openvino")


def test_list_without_pi05_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models_cli, "_studio_models", list)

    result = CliRunner().invoke(models_cli.models, ["fix-pi05-noise"])

    assert result.exit_code != 0
    assert "No Studio pi05 model has an OpenVINO export" in result.output
