# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Temporary POC validation sweep for real pretrained Hugging Face Hub VLAs.

Exports each OpenVINO bundle once and evaluates it across every requested OpenVINO device
(e.g. ``CPU``, ``GPU``, ``NPU`` on Intel Panther Lake), recording export time, bundle size,
load time, warm chunk inference latency, dynamic ``task_tokens`` switching, and Tier 3
torch-free execution.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import openvino as ov
from physicalai.benchmark.performance import InferenceLatencyBenchmark
from physicalai.engine import Engine

CASE_CATALOG: dict[str, tuple[str, str, str, dict[str, object]]] = {
    "smolvla_ov": (
        "smolvla",
        "lerobot/smolvla_base",
        "openvino",
        {"task": "pick up the red cube"},
    ),
    "smolvla_ov_fp16": (
        "smolvla",
        "lerobot/smolvla_base",
        "openvino",
        {"task": "pick up the red cube", "quantize": "fp16", "tolerance": 0.15},
    ),
    "smolvla_ov_int8": (
        "smolvla",
        "lerobot/smolvla_base",
        "openvino",
        {"task": "pick up the red cube", "quantize": "int8", "tolerance": 0.25},
    ),
    "smolvla_et": (
        "smolvla",
        "lerobot/smolvla_base",
        "executorch_openvino",
        {"task": "pick up the red cube"},
    ),
    "xvla_ov": (
        "xvla",
        "lerobot/xvla-libero",
        "openvino",
        {"task": "pick up the red cube", "dtype": "bfloat16", "trust_remote_code": True},
    ),
    "xvla_ov_int8": (
        "xvla",
        "lerobot/xvla-libero",
        "openvino",
        {"task": "pick up the red cube", "dtype": "bfloat16", "quantize": "int8", "tolerance": 0.25, "trust_remote_code": True},
    ),
    "xvla_et": (
        "xvla",
        "lerobot/xvla-libero",
        "executorch_openvino",
        {"task": "pick up the red cube", "dtype": "bfloat16", "trust_remote_code": True},
    ),
    "pi05_ov": (
        "pi05",
        "lerobot/pi05_libero",
        "openvino",
        {"task": "pick up the black bowl", "dtype": "bfloat16"},
    ),
    "pi05_ov_int8": (
        "pi05",
        "lerobot/pi05_libero",
        "openvino",
        {"task": "pick up the black bowl", "dtype": "bfloat16", "quantize": "int8", "tolerance": 0.30},
    ),
    "pi05_ov_int4": (
        "pi05",
        "lerobot/pi05_libero",
        "openvino",
        {"task": "pick up the black bowl", "dtype": "bfloat16", "quantize": "int4", "tolerance": 0.5},
    ),
}

DEFAULT_CASES = ("smolvla_ov", "smolvla_ov_int8", "smolvla_et", "xvla_ov", "xvla_et", "pi05_ov")


def _detect_devices(requested: list[str]) -> list[str]:
    core = ov.Core()
    available = list(core.available_devices)
    print("Detected OpenVINO devices:")
    for dev in available:
        try:
            name = core.get_property(dev, "FULL_DEVICE_NAME")
        except RuntimeError:
            name = dev
        print(f"  - {dev}: {name}")
    if requested == ["auto"]:
        return available or ["CPU"]
    return requested


_PARITY_RE = re.compile(r"max normalized diff ([0-9.eE+-]+)")


class _ParityCaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.last_diff: float | None = None

    def emit(self, record: logging.LogRecord) -> None:
        match = _PARITY_RE.search(record.getMessage())
        if match:
            self.last_diff = float(match.group(1))


def _verify_torch_free_tier3(pkg: Path, device: str, cache_dir: Path, task: str) -> bool:
    code = f"""
import sys, numpy as np
from physicalai.inference import InferenceModel
with InferenceModel({str(pkg)!r}, device={device!r}, cache_dir={str(cache_dir)!r}, verify=False) as m:
    ex = m.adapter._executor
    obs = {{k: np.zeros(s, dtype=ex.frame_dtypes[k]) for k, s in ex.frame_shapes.items()}}
    if {task!r}:
        obs["task"] = {task!r}
    a = m.select_action(obs)
    assert np.isfinite(a).all()
    assert "torch" not in sys.modules
    assert "lerobot" not in sys.modules
    assert "executorch" not in sys.modules
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    return res.returncode == 0


def _run_bench_worker(payload: dict[str, object]) -> dict[str, object]:
    """Run a single (package, device) load + InferenceLatencyBenchmark in an isolated subprocess."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parity_handler = _ParityCaptureHandler()
    logging.getLogger("physicalai_lerobot_policy_plugin.bundle_runtime").addHandler(parity_handler)

    pkg = Path(str(payload["pkg"]))
    backend = str(payload["backend"])
    dev = str(payload["device"])
    cache_dir = Path(str(payload["cache_dir"]))
    task_str = str(payload["task"])
    has_task_tokens = bool(payload["has_task_tokens"])
    verify_tier3 = bool(payload["verify_tier3"])
    warmup_iters = int(payload["warmup_iters"])  # type: ignore[arg-type]
    iters = int(payload["iters"])  # type: ignore[arg-type]
    max_duration_ms = int(payload["max_duration_ms"])  # type: ignore[arg-type]

    dev_entry: dict[str, object] = {
        "status": "FAIL",
        "load_s": None,
        "parity_diff": None,
        "infer_ms": None,
        "latency": None,
        "error": None,
    }
    t1 = time.monotonic()
    try:
        load_kwargs: dict[str, object] = {"device": dev}
        if backend == "openvino":
            load_kwargs["cache_dir"] = cache_dir / dev.lower()
        with Engine.load(pkg, **load_kwargs) as model:
            dev_entry["load_s"] = round(time.monotonic() - t1, 2)
            dev_entry["parity_diff"] = parity_handler.last_diff
            ex = model.adapter._executor
            if hasattr(ex, "torch_loaded"):
                dev_entry["worker_torch_loaded"] = ex.torch_loaded
            obs = {k: np.zeros(s, dtype=ex.frame_dtypes[k]) for k, s in ex.frame_shapes.items()}
            if task_str:
                obs["task"] = task_str
            # Sanity probe + InferenceLatencyBenchmark
            chunk = model.predict_action_chunk(obs)
            assert np.isfinite(chunk).all()
            model.reset()
            benchmark = InferenceLatencyBenchmark(
                max_iters=iters,
                warmup_iters=warmup_iters,
                max_duration=max_duration_ms,
            )
            metrics = benchmark.run(model, inputs=itertools.repeat(obs))
            model.reset()
            latency_ms: dict[str, float | int] = {
                "num_iters": int(metrics["num_iters"]),
                "warmup_ms": round(float(metrics["avg_warmup_iter_time"]) * 1000.0, 2),
                "median_ms": round(float(metrics["median_iter_time"]) * 1000.0, 2),
                "mean_ms": round(float(metrics["mean_iter_time"]) * 1000.0, 2),
                "min_ms": round(float(metrics["min_iter_time"]) * 1000.0, 2),
                "max_ms": round(float(metrics["max_iter_time"]) * 1000.0, 2),
                "std_ms": round(float(metrics["std_iter_time"]) * 1000.0, 2),
            }
            dev_entry["action_shape"] = list(chunk.shape)
            dev_entry["infer_ms"] = latency_ms["median_ms"]
            dev_entry["latency"] = latency_ms
            if has_task_tokens:
                obs_alt = dict(obs, task="place the mug on the plate")
                alt_chunk = model.predict_action_chunk(obs_alt)
                assert np.isfinite(alt_chunk).all()
                dev_entry["dynamic_task"] = True
        if verify_tier3:
            dev_entry["tier3_torch_free"] = _verify_torch_free_tier3(pkg, dev, cache_dir / dev.lower(), task_str)
        dev_entry["status"] = "PASS"
    except Exception as exc:
        dev_entry["load_s"] = round(time.monotonic() - t1, 2)
        dev_entry["parity_diff"] = parity_handler.last_diff
        dev_entry["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
    finally:
        gc.collect()
    return dev_entry


def _run_bench_in_subprocess(payload: dict[str, object]) -> dict[str, object]:
    t0 = time.monotonic()
    with tempfile.NamedTemporaryFile(prefix="hub-bench-", suffix=".json", delete=False) as tmp:
        result_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--_bench-worker", json.dumps(payload), "--_bench-out", str(result_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result_path.is_file() and result_path.stat().st_size > 0:
            return json.loads(result_path.read_text(encoding="utf-8"))
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return {
            "status": "FAIL",
            "load_s": round(time.monotonic() - t0, 2),
            "parity_diff": None,
            "infer_ms": None,
            "latency": None,
            "error": f"Worker crashed (exit={proc.returncode}): {' | '.join(tail)[:240]}",
        }
    finally:
        result_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pretrained HF Hub LeRobot VLAs across Intel devices.")
    parser.add_argument("--_bench-worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_bench-out", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES), choices=sorted(CASE_CATALOG))
    parser.add_argument("--devices", nargs="+", default=["auto"], help="OpenVINO runtime devices (e.g. CPU GPU NPU, or auto).")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/ov_cache_poc"))
    parser.add_argument("--keep-exports", type=Path, default=None, help="Directory to keep exported packages instead of deleting them.")
    parser.add_argument("--warmup-iters", type=int, default=2, help="InferenceLatencyBenchmark warmup iterations.")
    parser.add_argument("--iters", type=int, default=20, help="InferenceLatencyBenchmark max measured iterations.")
    parser.add_argument("--max-duration-ms", type=int, default=60000, help="InferenceLatencyBenchmark max duration (ms).")
    parser.add_argument("--output-json", type=Path, default=Path("tmp_hub_validation_results.json"))
    args = parser.parse_args()

    if args._bench_worker is not None:
        result = _run_bench_worker(json.loads(args._bench_worker))
        assert args._bench_out is not None
        args._bench_out.write_text(json.dumps(result), encoding="utf-8")
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    devices = _detect_devices(args.devices)

    temp_ctx = None if args.keep_exports is not None else tempfile.TemporaryDirectory(prefix="poc-hub-validation-")
    export_root = args.keep_exports.expanduser().resolve() if args.keep_exports is not None else Path(temp_ctx.name)
    export_root.mkdir(parents=True, exist_ok=True)

    engine = Engine.create("lerobot")
    results: list[dict[str, object]] = []
    try:
        for case_name in args.cases:
            policy_name, repo_id, backend, opts = CASE_CATALOG[case_name]
            out_dir = export_root / case_name
            entry: dict[str, object] = {
                "case": case_name,
                "policy": policy_name,
                "repo_id": repo_id,
                "backend": backend,
                "quantize": opts.get("quantize"),
                "dtype": opts.get("dtype"),
                "export_status": "FAIL",
                "export_s": None,
                "parts": [],
                "features": [],
                "tolerance": None,
                "bundle_mb": None,
                "devices": {},
                "error": None,
            }
            print(
                f"\n>>> [{case_name}] {repo_id} ({backend}, dtype={opts.get('dtype')}, quantize={opts.get('quantize')})",
                flush=True,
            )
            t0 = time.monotonic()
            try:
                if out_dir.exists() and (out_dir / "manifest.json").is_file():
                    print(f"Reusing existing export at {out_dir}", flush=True)
                    pkg = out_dir
                    entry["export_s"] = 0.0
                else:
                    pkg = engine.export(repo_id, out_dir, backend=backend, **opts)
                    entry["export_s"] = round(time.monotonic() - t0, 1)
                entry["export_status"] = "PASS"

                bundle_dir = pkg / f"{policy_name}.lerobot-bundle"
                export_json = json.loads((bundle_dir / "export.json").read_text(encoding="utf-8"))
                entry["parts"] = sorted(export_json["programs"].keys())
                entry["tolerance"] = export_json["tolerance"]
                entry["bundle_mb"] = round(
                    sum(p.stat().st_size for p in bundle_dir.rglob("*") if p.is_file()) / (1024 * 1024),
                    1,
                )
                entry["features"] = [
                    k
                    for k in ("history", "carry", "first_runs", "task_tokens", "action_tokens", "constants", "noise_shape")
                    if export_json.get(k)
                ]

                run_devices = devices if backend == "openvino" else ["CPU"]
                dev_results: dict[str, dict[str, object]] = {}
                for dev in run_devices:
                    dev_entry = _run_bench_in_subprocess(
                        {
                            "pkg": str(pkg),
                            "backend": backend,
                            "device": dev,
                            "cache_dir": str(args.cache_dir),
                            "task": str(opts.get("task") or ""),
                            "has_task_tokens": bool(export_json.get("task_tokens")),
                            "verify_tier3": backend == "openvino" and dev == run_devices[0],
                            "warmup_iters": args.warmup_iters,
                            "iters": args.iters,
                            "max_duration_ms": args.max_duration_ms,
                        }
                    )
                    dev_results[dev] = dev_entry
                    print(
                        f"  [{dev_entry['status']:4s}] {case_name} on {dev}: "
                        f"load={dev_entry['load_s']}s parity={dev_entry['parity_diff']} "
                        f"median={dev_entry['infer_ms']}ms size={entry['bundle_mb']}MB "
                        f"err={dev_entry['error'] or '-'}",
                        flush=True,
                    )
                entry["devices"] = dev_results
            except Exception as exc:
                entry["export_s"] = round(time.monotonic() - t0, 1)
                entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                print(f"  [FAIL] {case_name}: {entry['error']}", flush=True)
            results.append(entry)
            args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

    args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote results to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
