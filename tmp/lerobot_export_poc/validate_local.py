# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Temporary POC validation sweep across local LeRobot policy families and Intel OpenVINO devices.

Exports each ``openvino`` bundle once on CPU, then validates loading and inference across every
requested OpenVINO device (e.g. ``CPU``, ``GPU``, ``NPU`` on Intel Panther Lake). Also validates
``executorch_openvino`` (Tier 2 isolated worker) and verifies zero PyTorch/LeRobot imports in Tier 3.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import openvino as ov
from physicalai.benchmark.performance import InferenceLatencyBenchmark
from physicalai.engine import Engine, prepare_environment
from physicalai_lerobot_policy_plugin.environments import LEROBOT_EXPORT_UPSTREAM_REV, export_worker_env

ALL_LOCAL_POLICIES = (
    "act",
    "diffusion",
    "vqbet",
    "tdmpc",
    "gaussian_actor",
    "groot",
    "g05",
    "cosmos3_edge",
    "lingbot_va",
)

POLICY_SPECS: dict[str, tuple[str, bool, bool]] = {
    # policy: (task, trust_remote_code, needs_upstream_test_helpers)
    "act": ("", False, False),
    "diffusion": ("", False, False),
    "vqbet": ("", True, False),
    "tdmpc": ("", True, False),
    "gaussian_actor": ("", True, False),
    "groot": ("Pick up the red cube", True, True),
    "g05": ("pick up the cube", True, True),
    "cosmos3_edge": ("pick up the cube", True, True),
    "lingbot_va": ("pick up the red cube", True, True),
}

_LEROBOT_REPO = "https://github.com/shoumikhin/lerobot.git"


def _ensure_lerobot_test_helpers(cache_dir: Path) -> Path:
    checkout = cache_dir / f"shoumikhin-lerobot-{LEROBOT_EXPORT_UPSTREAM_REV[:8]}"
    if (checkout / "tests" / "export" / "tiny_groot.py").is_file():
        return checkout
    if checkout.exists():
        shutil.rmtree(checkout)
    print(f"Fetching upstream LeRobot export test helpers ({LEROBOT_EXPORT_UPSTREAM_REV[:8]})...", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", "lerobot-export-stack-nov5", _LEROBOT_REPO, str(checkout)],
        check=True,
    )
    return checkout


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


def _benchmark_loaded(
    model: object,
    obs: dict[str, object],
    *,
    warmup: int = 2,
    iters: int = 20,
    max_duration_ms: int = 15000,
) -> tuple[list[int], dict[str, float | int]]:
    has_history = bool(getattr(model.adapter._executor, "has_history", False))
    probe = model.select_action(obs) if has_history else model.predict_action_chunk(obs)
    assert np.isfinite(probe).all()
    model.reset()

    def _inputs() -> Iterator[dict[str, object]]:
        while True:
            if has_history:
                model.reset()
            yield obs

    benchmark = InferenceLatencyBenchmark(max_iters=iters, warmup_iters=warmup, max_duration=max_duration_ms)
    metrics = benchmark.run(model, inputs=_inputs())
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
    return list(probe.shape), latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local LeRobot policy families across Intel backends/devices.")
    parser.add_argument("--policies", nargs="+", default=list(ALL_LOCAL_POLICIES), choices=ALL_LOCAL_POLICIES)
    parser.add_argument("--backends", nargs="+", default=["openvino", "executorch_openvino"], choices=["openvino", "executorch_openvino"])
    parser.add_argument("--devices", nargs="+", default=["auto"], help="OpenVINO runtime devices (e.g. CPU GPU NPU, or auto).")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/ov_cache_poc"))
    parser.add_argument("--warmup-iters", type=int, default=2, help="InferenceLatencyBenchmark warmup iterations.")
    parser.add_argument("--iters", type=int, default=20, help="InferenceLatencyBenchmark max measured iterations.")
    parser.add_argument("--max-duration-ms", type=int, default=15000, help="InferenceLatencyBenchmark max duration (ms).")
    parser.add_argument("--group-norm", action="store_true", help="Enable use_group_norm=True on diffusion/vqbet.")
    parser.add_argument("--output-json", type=Path, default=Path("tmp_local_validation_results.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parity_handler = _ParityCaptureHandler()
    bundle_logger = logging.getLogger("physicalai_lerobot_policy_plugin.bundle_runtime")
    bundle_logger.setLevel(logging.INFO)
    bundle_logger.addHandler(parity_handler)
    devices = _detect_devices(args.devices)

    needs_helpers = any(POLICY_SPECS[p][2] for p in args.policies)
    lerobot_src = _ensure_lerobot_test_helpers(Path("/tmp")) if needs_helpers else None

    vla_env = prepare_environment(export_worker_env("vla"), "cpu")
    builder_script = Path(__file__).resolve().parent / "build_local_checkpoints.py"

    with tempfile.TemporaryDirectory(prefix="poc-local-validation-") as tmp:
        tmp_root = Path(tmp)
        ckpt_root = tmp_root / "checkpoints"
        build_cmd = [str(vla_env.python), str(builder_script), str(ckpt_root), "--policies", *args.policies]
        if lerobot_src is not None:
            build_cmd.extend(["--lerobot-src", str(lerobot_src)])
        if args.group_norm:
            build_cmd.append("--group-norm")
        print(f"Building checkpoints for: {', '.join(args.policies)} ...", flush=True)
        subprocess.run(build_cmd, check=True)

        export_root = tmp_root / "exports"
        export_root.mkdir()
        engine = Engine.create("lerobot")
        results: list[dict[str, object]] = []

        for policy_name in args.policies:
            task, trust_remote, _ = POLICY_SPECS[policy_name]
            if policy_name == "groot":
                ckpt_path = Path((ckpt_root / "groot.path").read_text(encoding="utf-8").strip())
            else:
                ckpt_path = ckpt_root / policy_name

            for backend in args.backends:
                out_dir = export_root / f"{policy_name}_{backend}"
                entry: dict[str, object] = {
                    "policy": policy_name,
                    "backend": backend,
                    "export_status": "FAIL",
                    "export_s": None,
                    "parts": [],
                    "features": [],
                    "tolerance": None,
                    "devices": {},
                    "error": None,
                }
                t0 = time.monotonic()
                try:
                    pkg = engine.export(
                        ckpt_path,
                        out_dir,
                        backend=backend,
                        task=task,
                        trust_remote_code=trust_remote,
                    )
                    entry["export_s"] = round(time.monotonic() - t0, 1)
                    entry["export_status"] = "PASS"

                    bundle_dir = pkg / f"{policy_name}.lerobot-bundle"
                    export_json = json.loads((bundle_dir / "export.json").read_text(encoding="utf-8"))
                    entry["parts"] = sorted(export_json["programs"].keys())
                    entry["tolerance"] = export_json["tolerance"]
                    entry["features"] = [
                        k
                        for k in ("history", "carry", "first_runs", "task_tokens", "action_tokens", "constants", "noise_shape")
                        if export_json.get(k)
                    ]

                    run_devices = devices if backend == "openvino" else ["CPU"]
                    device_results: dict[str, dict[str, object]] = {}
                    for dev in run_devices:
                        dev_entry: dict[str, object] = {
                            "status": "FAIL",
                            "load_s": None,
                            "parity_diff": None,
                            "infer_ms": None,
                            "latency": None,
                            "error": None,
                        }
                        parity_handler.last_diff = None
                        t1 = time.monotonic()
                        try:
                            load_kwargs: dict[str, object] = {"device": dev}
                            if backend == "openvino":
                                load_kwargs["cache_dir"] = args.cache_dir / dev.lower()
                            with Engine.load(pkg, **load_kwargs) as model:
                                dev_entry["load_s"] = round(time.monotonic() - t1, 2)
                                dev_entry["parity_diff"] = parity_handler.last_diff
                                ex = model.adapter._executor
                                if hasattr(ex, "torch_loaded"):
                                    dev_entry["worker_torch_loaded"] = ex.torch_loaded
                                obs = {k: np.zeros(s, dtype=ex.frame_dtypes[k]) for k, s in ex.frame_shapes.items()}
                                if task:
                                    obs["task"] = task
                                shape, latency = _benchmark_loaded(
                                    model,
                                    obs,
                                    warmup=args.warmup_iters,
                                    iters=args.iters,
                                    max_duration_ms=args.max_duration_ms,
                                )
                                dev_entry["action_shape"] = shape
                                dev_entry["infer_ms"] = latency["median_ms"]
                                dev_entry["latency"] = latency
                                if export_json.get("task_tokens"):
                                    obs_alt = dict(obs, task="stop")
                                    alt_out = model.predict_action_chunk(obs_alt)
                                    assert np.isfinite(alt_out).all()
                                    dev_entry["dynamic_task"] = True
                            dev_entry["status"] = "PASS"
                        except Exception as exc:
                            dev_entry["load_s"] = round(time.monotonic() - t1, 2)
                            dev_entry["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
                        device_results[dev] = dev_entry
                    entry["devices"] = device_results
                except Exception as exc:
                    entry["export_s"] = round(time.monotonic() - t0, 1)
                    entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"

                results.append(entry)
                dev_summary = ", ".join(
                    f"{d}:{r['status']}({r['infer_ms']}ms)" if r["status"] == "PASS" else f"{d}:FAIL"
                    for d, r in entry["devices"].items()  # type: ignore[union-attr]
                )
                print(
                    f"[{entry['export_status']:4s}] {policy_name:14s} | {backend:19s} | "
                    f"export={entry['export_s']}s | parts={len(entry['parts'])} | {dev_summary or entry['error']}",
                    flush=True,
                )

    args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote results to {args.output_json}")


if __name__ == "__main__":
    main()
