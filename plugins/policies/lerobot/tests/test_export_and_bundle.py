# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LeRobot export enablements, bundle packaging, and Tier 3 OpenVINO execution."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib import resources
from pathlib import Path

import numpy as np
import openvino as ov
import pytest
from openvino import opset13 as opset
from physicalai.engine import Run
from safetensors.numpy import save_file as save_safetensors

from physicalai_lerobot_policy_plugin import engine as engine_module
from physicalai_lerobot_policy_plugin.bundle_runtime import (
    LeRobotBundleAdapter,
    OpenVINOBundleExecutor,
    _BundleManifest,
    _to_bundle_frame,
)
from physicalai_lerobot_policy_plugin.engine import LeRobotEngine, LeRobotExportOptions
from physicalai_lerobot_policy_plugin.package import package_lerobot_bundle
from physicalai_lerobot_policy_plugin.policy_support import (
    VERIFIED_EXPORT_POLICIES,
    require_exportable_policy,
)


def _write_minimal_ov_bundle(bundle_dir: Path, *, with_history: bool = False) -> Path:
    """Create a real OpenVINO IR `.lerobot-bundle` (y = 2 * state) with zero PyTorch."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    input_shape = [2, 3] if with_history else [3]
    param = opset.parameter(input_shape, np.float32, name="state")
    scale = opset.constant(np.array(2.0, dtype=np.float32))
    scaled = opset.multiply(param, scale)
    if with_history:
        # Output shape [2, 3] directly (execution_steps=2, action_dim=3).
        output = scaled
    else:
        # Reshape [3] -> [1, 3] (execution_steps=1, action_dim=3).
        output = opset.reshape(scaled, opset.constant(np.array([1, 3], dtype=np.int64)), special_zero=False)
    ov_model = ov.Model([output], [param], "tiny_policy")
    ov.save_model(ov_model, str(bundle_dir / "model.xml"), compress_to_fp16=False)

    if with_history:
        frames = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        expected = frames * 2.0
    else:
        frames = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        expected = np.array([[2.0, 4.0, 6.0]], dtype=np.float32)

    save_safetensors({"observation.state": frames, "action": expected}, str(bundle_dir / "test.safetensors"))
    manifest: dict[str, object] = {
        "format_version": 1,
        "backend": "openvino",
        "runtime": "openvino",
        "runtime_imports": ["openvino"],
        "programs": {"model": {"file": "model.xml", "data": "model.bin"}},
        "runs": [
            {
                "name": "model",
                "program": "model",
                "inputs": ["observation.state"],
                "outputs": ["action"],
            },
        ],
        "action_scale": [1.0, 1.0, 1.0],
        "tolerance": 1e-4,
        "test_case": "test.safetensors",
    }
    if with_history:
        manifest["history"] = {"observation.state": 2}
    (bundle_dir / "export.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return bundle_dir


def test_enablements_advertise_dual_intel_backends() -> None:
    assert {enablement.name for enablement in LeRobotEngine.enablements} == {"openvino", "executorch-openvino"}
    for policy in VERIFIED_EXPORT_POLICIES:
        caps = LeRobotEngine.capabilities(policy)
        assert set(caps.export_formats) == {"native", "openvino", "executorch_openvino"}
        assert set(caps.enablements) == {"openvino", "executorch-openvino"}
    assert LeRobotEngine.environment("openvino").name == "export"
    assert LeRobotEngine.environment("executorch-openvino").name == "export"


def test_require_exportable_policy_enforces_allowlist_with_opt_in() -> None:
    assert require_exportable_policy("act") == "act"
    with pytest.raises(ValueError, match="trust_remote_code=True"):
        require_exportable_policy("groot")
    assert require_exportable_policy("groot", trust_remote_code=True) == "groot"
    with pytest.raises(ValueError, match="must match"):
        require_exportable_policy("../escape", trust_remote_code=True)


def test_export_routes_openvino_and_builds_cli_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ckpt = tmp_path / "ckpt" / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text(json.dumps({"type": "act"}), encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run_worker(env, args, *, device, accelerator, interpreter=None, extra_env=None):  # type: ignore[no-untyped-def]
        captured["env_name"] = env.name
        captured["args"] = list(args)
        captured["device"] = device
        captured["accelerator"] = accelerator
        captured["extra_env"] = dict(extra_env or {})
        out_arg = next(a for a in args if a.startswith("--output_dir=")).split("=", 1)[1]
        _write_minimal_ov_bundle(Path(out_arg))
        return Run(framework="lerobot", output_dir=Path(out_arg), checkpoint=Path(out_arg))

    monkeypatch.setattr(engine_module, "run_worker", fake_run_worker)

    package = LeRobotEngine.export(
        ckpt,
        tmp_path / "exported_act",
        backend="openvino",
        options=LeRobotExportOptions(
            task="pick cube",
            quantize="int8",
            tolerance=0.2,
            n_action_steps=1,
            seed=42,
        ),
    )

    assert captured["env_name"] == "export"
    assert captured["accelerator"] == "cpu"
    assert "PHYSICALAI_LEROBOT_ENABLEMENT_OV_DIR" in captured["extra_env"]  # type: ignore[operator]
    args = captured["args"]
    assert isinstance(args, list)
    assert "--backend.type=openvino" in args
    assert "--task=pick cube" in args
    assert "--quantize=int8" in args
    assert "--tolerance=0.2" in args
    assert "--seed=42" in args
    assert (package / "manifest.json").is_file()
    assert (package / "package.json").is_file()
    assert (package / "act.lerobot-bundle" / "export.json").is_file()


def test_export_accepts_direct_kwargs_via_engine_create(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from physicalai.engine import Engine

    ckpt = tmp_path / "ckpt" / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text(json.dumps({"type": "act"}), encoding="utf-8")

    captured: list[str] = []

    def fake_run_worker(env, args, *, device, accelerator, interpreter=None, extra_env=None):  # type: ignore[no-untyped-def]
        del env, device, accelerator, interpreter, extra_env
        captured.extend(args)
        out_arg = next(a for a in args if a.startswith("--output_dir=")).split("=", 1)[1]
        _write_minimal_ov_bundle(Path(out_arg))
        return Run(framework="lerobot", output_dir=Path(out_arg), checkpoint=Path(out_arg))

    monkeypatch.setattr(engine_module, "run_worker", fake_run_worker)

    engine = Engine.create("lerobot")
    package = engine.export(
        ckpt,
        tmp_path / "exported_via_engine_create",
        backend="openvino",
        task="pick cube",
        quantize="int8",
        tolerance=0.2,
    )
    assert "--task=pick cube" in captured
    assert "--quantize=int8" in captured
    assert "--tolerance=0.2" in captured
    assert (package / "manifest.json").is_file()

    with pytest.raises(ValueError, match="not both"):
        engine.export(
            ckpt,
            tmp_path / "conflict",
            backend="openvino",
            options=LeRobotExportOptions(task="a"),
            task="b",
        )


def test_export_rejects_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="supports export backends"):
        LeRobotEngine.export(tmp_path, tmp_path / "out", backend="tensorrt")


def test_tier3_openvino_bundle_adapter_runs_and_resets(tmp_path: Path) -> None:
    bundle = _write_minimal_ov_bundle(tmp_path / "raw_bundle")
    package = package_lerobot_bundle(bundle, tmp_path / "pkg", policy_name="act", export_backend="openvino")

    adapter = LeRobotBundleAdapter(device="CPU")
    adapter.load(package / "act.lerobot-bundle")
    out = adapter.predict({"state": np.array([3.0, -1.0, 0.5], dtype=np.float32)})
    np.testing.assert_allclose(out["action"], np.array([[[6.0, -2.0, 1.0]]], dtype=np.float32), atol=1e-5)
    adapter.reset()
    adapter.close()


def test_tier3_openvino_bundle_handles_history_and_reset(tmp_path: Path) -> None:
    bundle = _write_minimal_ov_bundle(tmp_path / "raw_hist", with_history=True)
    executor = OpenVINOBundleExecutor(bundle, _BundleManifest.load(bundle), device="CPU")
    assert executor.has_history is True

    first = executor.get_action({"observation.state": np.array([1.0, 2.0, 3.0], dtype=np.float32)})
    np.testing.assert_allclose(first, np.array([2.0, 4.0, 6.0], dtype=np.float32), atol=1e-5)
    executor.reset()
    after_reset = executor.get_action({"observation.state": np.array([10.0, 0.0, -5.0], dtype=np.float32)})
    np.testing.assert_allclose(after_reset, np.array([20.0, 0.0, -10.0], dtype=np.float32), atol=1e-5)


def test_bundle_security_rejects_path_escape_symlinks_and_disallowed_imports(tmp_path: Path) -> None:
    bundle = _write_minimal_ov_bundle(tmp_path / "bad_imports")
    manifest = json.loads((bundle / "export.json").read_text(encoding="utf-8"))
    manifest["runtime_imports"] = ["os"]
    (bundle / "export.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="disallowed runtime_imports"):
        _BundleManifest.load(bundle)

    bundle_escape = _write_minimal_ov_bundle(tmp_path / "bad_path")
    manifest2 = json.loads((bundle_escape / "export.json").read_text(encoding="utf-8"))
    manifest2["programs"]["model"]["file"] = "../escape.xml"
    (bundle_escape / "export.json").write_text(json.dumps(manifest2), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the bundle folder"):
        _BundleManifest.load(bundle_escape)

    bundle_symlink = _write_minimal_ov_bundle(tmp_path / "symlink_bundle")
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    (bundle_symlink / "link.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic links"):
        package_lerobot_bundle(bundle_symlink, tmp_path / "out_sym", policy_name="act", export_backend="openvino")


def test_observation_coercion_handles_chw_float_and_batched_inputs() -> None:
    chw_float = np.ones((1, 3, 4, 5), dtype=np.float32) * 0.5
    frame, task = _to_bundle_frame(
        {"images": {"top": chw_float}, "state": np.array([[1.0, 2.0]], dtype=np.float64), "task": ["pick"]},
        frame_shapes={"observation.images.top": (4, 5, 3), "observation.state": (2,)},
        frame_dtypes={"observation.images.top": np.dtype(np.uint8), "observation.state": np.dtype(np.float32)},
    )
    assert frame["observation.images.top"].shape == (4, 5, 3)
    assert frame["observation.images.top"].dtype == np.uint8
    assert int(frame["observation.images.top"][0, 0, 0]) == 128
    assert frame["observation.state"].shape == (2,)
    assert frame["observation.state"].dtype == np.float32
    assert task == "pick"


def test_layer1_bundle_runtime_imports_without_torch_lerobot_or_executorch() -> None:
    code = (
        "import sys; "
        "import physicalai_lerobot_policy_plugin.bundle_runtime; "
        "import physicalai_lerobot_policy_plugin.environments; "
        "import physicalai_lerobot_policy_plugin.policy_support; "
        "import physicalai_lerobot_policy_plugin.inference; "
        "assert 'torch' not in sys.modules, 'torch loaded!'; "
        "assert 'lerobot' not in sys.modules, 'lerobot loaded!'; "
        "assert 'executorch' not in sys.modules, 'executorch loaded!'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("worker_name", ["export.py", "infer.py"])
def test_workers_compile_and_import_under_safe_path_without_physicalai(worker_name: str) -> None:
    worker = resources.files("physicalai_lerobot_policy_plugin") / "_worker" / worker_name
    with resources.as_file(worker) as worker_path:
        code = (
            "import importlib.util, sys; "
            f"spec = importlib.util.spec_from_file_location('worker_under_test', {str(worker_path)!r}); "
            "mod = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(mod); "
            "assert not any(m == 'physicalai' or m.startswith('physicalai.') for m in sys.modules); "
            "assert 'torch' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-P", "-c", code], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
