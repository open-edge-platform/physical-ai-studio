# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001
"""ExecuTorch (OpenVINO delegate) support matrix for every available policy.

For each first-party and LeRobot policy, in an isolated subprocess:
  1. build it with random weights and synthetic dataset stats,
  2. export to ExecuTorch with the OpenVINO delegate, once per target device
     (the device is baked into the delegate at export time), even if the policy
     does not declare ExecuTorch support,
  3. run the ``.pte`` with the ExecuTorch runtime on the export sample, check the
     output is finite and report max |diff| against the eager export module.
     The diff is informational: flow-matching / diffusion policies draw noise.

Results go to ``<out-dir>/executorch_support.csv``; per-policy logs sit next to it.

Run in an environment with CPU torch + ``executorch`` + ``openvino`` + ``nncf``
(torch XPU builds and ExecuTorch do not work together). The ``xpu`` column uses
the OpenVINO GPU plugin, not torch XPU. Running the OpenVINO-delegated ``.pte``
needs an ExecuTorch runtime built with the OpenVINO backend.

Usage:
    python scripts/executorch_support_matrix.py
        [--policies NAME ...] [--sources physicalai lerobot]
        [--devices cpu xpu] [--timeout SEC] [--out-dir DIR]

Examples:
    python scripts/executorch_support_matrix.py
    python scripts/executorch_support_matrix.py --sources lerobot --devices cpu
    python scripts/executorch_support_matrix.py --policies act diffusion
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import signal
import subprocess  # noqa: S404  # nosec B404  # reason: re-invokes this script with sys.executable, no shell
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# --devices name -> OpenVINO device baked into the ExecuTorch delegate.
DEVICES = {"cpu": "CPU", "xpu": "GPU"}

STATE_DIM, ACTION_DIM, IMAGE_SHAPE = 8, 7, (3, 224, 224)


def _stat(name: str, ftype: str, shape: tuple[int, ...], channels: int) -> dict[str, Any]:
    return {
        "name": name,
        "type": ftype,
        "shape": shape,
        "mean": [0.0] * channels,
        "std": [1.0] * channels,
        "min": [-1.0] * channels,
        "max": [1.0] * channels,
        "q01": [-1.0] * channels,
        "q99": [1.0] * channels,
    }


# Synthetic physicalai-format dataset stats shared by all first-party policies.
STATS = {
    "observation.state": _stat("state", "STATE", (STATE_DIM,), STATE_DIM),
    "observation.images.base": _stat("images.base", "VISUAL", IMAGE_SHAPE, IMAGE_SHAPE[0]),
    "action": _stat("action", "ACTION", (ACTION_DIM,), ACTION_DIM),
}

# Per-policy constructor kwargs (random weights: no pretrained checkpoint).
# Configs / tokenizers of VLM backbones may still be fetched from the Hub.
PHYSICALAI_KWARGS: dict[str, dict[str, Any]] = {
    "act": {},
    "molmoact2": {"pretrained_name_or_path": None},
    "pi05": {"dtype": "float32"},
    "rldx1": {"pretrained_name_or_path": None},
    "smolvla": {"dtype": "float32"},
    "xr0": {"dtype": "float32"},
}
LEROBOT_KWARGS: dict[str, dict[str, Any]] = {
    "act": {"pretrained_backbone_weights": None},
    # Empty sub-configs -> Florence-2 defaults; transformers 5 has no default pad_token_id.
    "xvla": {"florence_config": {"vision_config": {}, "text_config": {}, "pad_token_id": 1}},
}


def _build(source: str, name: str) -> Any:  # noqa: ANN401
    import torch  # noqa: PLC0415

    torch.manual_seed(0)
    if source == "physicalai":
        from physicalai.data import Feature, FeatureType, NormalizationParameters  # noqa: PLC0415
        from physicalai.policies import get_policy  # noqa: PLC0415

        if name == "xr0":
            # XR0 wants per-timestep action stats of shape (chunk_size=30, D).
            action = {**STATS["action"], "mean": [[0.0] * ACTION_DIM] * 30, "std": [[1.0] * ACTION_DIM] * 30}
            return get_policy(name, dataset_stats={**STATS, "action": action}, **PHYSICALAI_KWARGS.get(name, {}))
        if name != "molmoact2":
            return get_policy(name, dataset_stats=STATS, **PHYSICALAI_KWARGS.get(name, {}))
        # MolmoAct2 takes Feature lists instead of dataset_stats.
        features = [
            Feature(
                name=s["name"],
                ftype=FeatureType(s["type"]),
                shape=s["shape"],
                normalization_data=NormalizationParameters(
                    **{k: s[k] for k in ("mean", "std", "min", "max", "q01", "q99")},
                ),
            )
            for s in STATS.values()
        ]
        return get_policy(
            name,
            input_features=[f for f in features if f.ftype != FeatureType.ACTION],
            output_features=[f for f in features if f.ftype == FeatureType.ACTION],
            **PHYSICALAI_KWARGS.get(name, {}),
        )

    from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: PLC0415

    from physicalai.policies.lerobot import get_lerobot_policy  # noqa: PLC0415

    shapes = {"observation.state": (STATE_DIM,), "observation.images.base": IMAGE_SHAPE, "action": (ACTION_DIM,)}
    stat_shapes = {**shapes, "observation.images.base": (IMAGE_SHAPE[0], 1, 1)}
    stats = {
        k: {
            "mean": torch.zeros(s),
            "std": torch.ones(s),
            "min": -torch.ones(s),
            "max": torch.ones(s),
            "q01": -torch.ones(s),
            "q99": torch.ones(s),
        }
        for k, s in stat_shapes.items()
    }
    return get_lerobot_policy(
        name,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=shapes["observation.state"]),
            "observation.images.base": PolicyFeature(type=FeatureType.VISUAL, shape=IMAGE_SHAPE),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=shapes["action"])},
        dataset_stats=stats,
        device="cpu",
        **LEROBOT_KWARGS.get(name, {}),
    )


def _error(exc: BaseException) -> str:
    cause = f" <- {type(exc.__cause__).__name__}: {exc.__cause__}" if exc.__cause__ else ""
    return f"{type(exc).__name__}: {exc}{cause}".replace("\n", " ")


def _first_tensor(out: Any) -> Any:  # noqa: ANN401
    import torch  # noqa: PLC0415

    return next(t for t in torch.utils._pytree.tree_leaves(out) if isinstance(t, torch.Tensor))  # noqa: SLF001


def _check_device(
    policy: Any,  # noqa: ANN401
    sample: dict,
    reference: Any,  # noqa: ANN401
    device: str,
    export_dir: Path,
    row: dict[str, Any],
    errors: list[str],
    save: Callable[[], None],
) -> None:
    """Export + run one device, filling ``export_<d>`` / ``run_<d>`` / ``diff_<d>`` in ``row``."""
    import torch  # noqa: PLC0415

    step = f"export_{device}"
    try:
        pte = policy.to_executorch(
            export_dir,
            input_sample=sample,
            delegate="openvino",
            delegate_config={"device": DEVICES[device]},
        )
        row[step] = "PASS"
        save()  # persist before running: a runtime segfault must not erase the export result

        step = f"run_{device}"
        from executorch.runtime import Runtime  # noqa: PLC0415

        method = Runtime.get().load_program(pte).load_method("forward")
        output = _first_tensor(method.execute(list(sample.values()))).float()
    except Exception as exc:  # noqa: BLE001
        row[step] = "FAIL"
        errors.append(f"{step}: {_error(exc)}")
        return

    if not torch.isfinite(output).all():
        row[step] = "FAIL"
        errors.append(f"{step}: non-finite output")
        return
    row[step] = "PASS"
    if reference is not None:
        row[f"diff_{device}"] = (
            f"{(output - reference).abs().max().item():.3g}"
            if output.shape == reference.shape
            else f"shape {tuple(output.shape)} vs {tuple(reference.shape)}"
        )


def _probe(source: str, name: str, devices: list[str], result_path: Path) -> None:
    """Child process: fill one CSV row, saving after every step so crashes keep partial results."""
    row: dict[str, Any] = {"policy": name, "source": source}
    errors: list[str] = []

    def save() -> None:
        row["error"] = " | ".join(errors)
        result_path.write_text(json.dumps(row), encoding="utf-8")

    from physicalai.export import ExportablePolicyMixin, ExportBackend  # noqa: PLC0415

    if source == "physicalai":
        from physicalai.policies import get_physicalai_policy_class  # noqa: PLC0415

        if not issubclass(get_physicalai_policy_class(name), ExportablePolicyMixin):
            row["build"] = "SKIP"
            errors.append("no ExportablePolicyMixin")
            save()
            return

    try:
        policy = _build(source, name).eval()
    except Exception as exc:  # noqa: BLE001
        row["build"] = "FAIL"
        errors.append(f"build: {_error(exc)}")
        save()
        return
    row["build"] = "PASS"
    row["declared_et"] = ExportBackend.EXECUTORCH in policy.get_supported_export_backends()
    save()

    import torch  # noqa: PLC0415

    # Probe ExecuTorch even when the policy does not declare it.
    all_backends = list(ExportBackend)
    type(policy).get_supported_export_backends = staticmethod(lambda: all_backends)

    try:
        # Captured once: sample_input is unseeded randn, export and run must share it.
        sample = policy._get_default_export_input_sample()  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        sample = None
        errors.append(f"sample: {_error(exc)}")
    if sample is None:
        errors.append("sample: policy provides no sample_input")
        save()
        return
    sample = {k: v.cpu() for k, v in sample.items()}

    try:
        with torch.no_grad():
            # Shallow copy: normalizers write normalized values back into the input dict.
            reference = _first_tensor(policy._export_module(dict(sample))).float()  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        reference = None
        errors.append(f"eager: {_error(exc)}")

    with tempfile.TemporaryDirectory() as tmp:
        for device in devices:
            _check_device(policy, sample, reference, device, Path(tmp) / device, row, errors, save)
            save()


def _run_one(source: str, name: str, devices: list[str], out_dir: Path, timeout: int) -> dict[str, Any]:
    """Parent side: run the probe in a subprocess (ExecuTorch / big VLAs can segfault or OOM).

    Returns:
        One CSV row; a crash / timeout is written into the first step that did not finish.
    """
    result_path = out_dir / f"{source}_{name}.json"
    result_path.unlink(missing_ok=True)
    cmd = [sys.executable, __file__, "--one", source, name, "--devices", *devices, "--result", str(result_path)]
    start = time.monotonic()
    with (out_dir / f"{source}_{name}.log").open("w", encoding="utf-8") as log:
        try:
            # Safe: argv list (no shell) re-invoking this script; names validated / taken from registries.
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)  # noqa: S603  # nosec B603
            failure = None if proc.returncode == 0 else f"CRASH rc={proc.returncode}"
            if proc.returncode == -signal.SIGKILL:  # almost always the kernel OOM killer
                failure = "OOM? (SIGKILL)"
        except subprocess.TimeoutExpired:
            failure = f"TIMEOUT {timeout}s"

    row = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    row.update(policy=name, source=source, seconds=round(time.monotonic() - start))
    if failure:
        steps = ["build"] + [f"{s}_{d}" for d in devices for s in ("export", "run")]
        first_missing = next((s for s in steps if s not in row), None)
        if first_missing:
            row[first_missing] = failure
        row["error"] = " | ".join(filter(None, [row.get("error"), failure]))
    return row


def main() -> None:
    """Build the ExecuTorch support CSV (or, with ``--one``, probe a single policy)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policies", nargs="+", help="Policy short names to include (default: all).")
    parser.add_argument("--sources", nargs="+", choices=["physicalai", "lerobot"], default=["physicalai", "lerobot"])
    parser.add_argument("--devices", nargs="+", choices=list(DEVICES), default=list(DEVICES))
    parser.add_argument("--timeout", type=int, default=1800, help="Per-policy timeout in seconds.")
    parser.add_argument("--out-dir", type=Path, default=Path("executorch_report"))
    parser.add_argument("--one", nargs=2, metavar=("SOURCE", "NAME"), help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one:
        _probe(*args.one, args.devices, args.result)
        return

    from physicalai.policies import list_physicalai_policies  # noqa: PLC0415
    from physicalai.policies.lerobot import SUPPORTED_POLICIES  # noqa: PLC0415

    available = {"physicalai": list_physicalai_policies(), "lerobot": list(SUPPORTED_POLICIES)}
    for name in args.policies or []:
        # Names become file names under --out-dir.
        if not re.fullmatch(r"^[a-zA-Z0-9_-]{1,64}$", name):
            parser.error(f"invalid policy name: {name!r}")
    targets = [(s, n) for s in args.sources for n in available[s] if not args.policies or n in args.policies]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["policy", "source", "declared_et", "build"]
    columns += [f"{step}_{d}" for d in args.devices for step in ("export", "run", "diff")]
    columns += ["error", "seconds"]
    csv_path = args.out_dir / "executorch_support.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for source, name in targets:
            logger.info("[%s] %s ...", source, name)
            row = _run_one(source, name, args.devices, args.out_dir, args.timeout)
            writer.writerow(row)
            handle.flush()
            logger.info("    %s", {k: row.get(k) for k in columns if k not in {"policy", "source", "error"}})
    logger.info("Wrote %s", csv_path)


if __name__ == "__main__":
    main()
