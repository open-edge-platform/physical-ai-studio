# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Print the weekly simulation report from downloaded CI artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

POLICIES = ("act", "molmoact2", "pi05", "rldx1", "smolvla", "xr0")


def summarize(root: Path) -> str:
    """Render separate training, wiring, timing, and task-quality verdicts.

    Returns:
        Markdown table for the GitHub Actions summary.
    """
    controls_path = root / "sim-controls" / "controls.json"
    baseline = "unavailable/failed"
    if controls_path.is_file():
        controls = json.loads(controls_path.read_text())
        if "off" in controls and "drive" in controls:
            baseline = (
                f"{controls.get('status', 'incomplete')}: expert {controls['drive']['score']:.1%}, "
                f"no action {controls['off']['score']:.1%}"
            )
    lines = [
        "## Weekly application conveyor regression",
        f"Simulator controls: {baseline}",
        "",
        "| Policy | Training/reload | Export + actions | Timing | Quality | Mean score |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for policy in POLICIES:
        train_path = root / f"sim-train-{policy}" / "training.json"
        evaluate_path = root / f"sim-evaluate-{policy}" / "result.json"
        train = json.loads(train_path.read_text()) if train_path.is_file() else {}
        result = json.loads(evaluate_path.read_text()) if evaluate_path.is_file() else {}
        training = train.get("status", "failed/incomplete").replace("unsupported_hardware", "unsupported hardware")
        if result.get("status") == "quality_baseline_unavailable":
            wiring, timing, quality = "not evaluated", "not evaluated", "baseline unavailable"
        else:
            wiring = result.get("wiring", "failed/incomplete")
            timing = result.get("timing", "failed/incomplete")
            quality = result.get("quality", "failed/incomplete")
        value = f"{result['score']:.1%}" if "score" in result else "—"
        lines.append(f"| {policy} | {training} | {wiring} | {timing} | {quality} | {value} |")
    lines += [
        "",
        "Missing/unqualified checkpoints and unsupported hardware are not passes. "
        "Inspect per-job logs and JUnit for errors.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.stdout.write(summarize(Path(sys.argv[1])))
