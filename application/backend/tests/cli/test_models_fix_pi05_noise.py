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
