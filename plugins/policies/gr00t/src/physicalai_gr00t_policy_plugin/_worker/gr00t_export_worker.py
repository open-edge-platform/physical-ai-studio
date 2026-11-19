# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export a GR00T checkpoint with the Robotics AI Suite OpenVINO patches.

This worker runs inside the patched ``openvino`` environment and calls the
Suite's single-IR exporter (``scripts/deployment/export_ov_n1d7_single_ov.py``)
directly. It records the exporter's own OpenVINO-versus-PyTorch verification
metrics so the Runtime package carries that evidence.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import gr00t_worker_common as common

RESULT_NAME = "physicalai_export.json"


def main(argv: list[str]) -> int:
    """Run one export request passed as a JSON argument.

    Returns:
        Process exit status.
    """
    channel = common.EventChannel()
    channel.send({"type": "hello", "protocol": common.PROTOCOL_VERSION, **common.identity()})
    try:
        result = _export(json.loads(argv[0]))
    except Exception as exc:  # ruff: ignore[blind-except]  # Report any exporter failure over the protocol.
        channel.send({"type": "error", "error": type(exc).__name__, "message": str(exc)})
        traceback.print_exc(file=sys.stderr)
        return 1
    channel.send({"type": "result", **result})
    return 0


def _export(request: dict[str, Any]) -> dict[str, Any]:
    common.disable_remote_code()
    deployment = common.source_dir() / "scripts" / "deployment"
    sys.path.insert(0, str(deployment))
    import export_ov_n1d7_single_ov as exporter
    from gr00t.data.embodiment_tags import EmbodimentTag

    captured: dict[str, Any] = {}
    native_verify = exporter.verify_openvino_single_ov

    def verify(*args: Any, **kwargs: Any) -> Any:  # ruff: ignore[any-type]
        metrics = native_verify(*args, **kwargs)
        captured["verification"] = metrics
        return metrics

    exporter.verify_openvino_single_ov = verify
    config = exporter.ExportConfig(
        model_path=request["model_path"],
        dataset_path=request.get("dataset_path"),
        embodiment_tag=EmbodimentTag.resolve(request["embodiment_tag"]),
        output_dir=request["output_dir"],
        precision=request.get("precision", "fp16"),
        device=request.get("device", "cpu"),
        llm_lang_tokens=request.get("llm_lang_tokens"),
    )
    exporter.main(config)
    output_dir = Path(config.output_dir)
    record = {
        "output_name": config.output_name,
        "precision": config.precision,
        "llm_lang_tokens": config.llm_lang_tokens,
        "verification": _numbers(captured.get("verification")),
    }
    (output_dir / RESULT_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(output_dir), "checkpoint": None}


def _numbers(metrics: Any) -> dict[str, float] | None:  # ruff: ignore[any-type]
    if not isinstance(metrics, dict):
        return None
    return {str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
