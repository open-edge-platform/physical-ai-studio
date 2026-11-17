# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Render simulation evidence as a GitHub Actions job summary, including partial runs."""

import html
import json
import math
import os
from pathlib import Path

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException


def cell(value: object) -> str:
    """Keep evidence text inside its Markdown table cell."""
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", "<br>").replace("\r", "")


def status(value: object) -> str:
    """Distinguish missing evidence from successful checks."""
    if not isinstance(value, str):
        return "⚠️ not reported"
    icon = {"passed": "✅", "success": "✅", "failed": "❌", "failure": "❌", "error": "❌", "skipped": "⏭️"}
    return f"{icon.get(value, '⚠️')} {cell(value)}"


def test_results(output: Path, warnings: list[str]) -> list[str]:
    """Render each pytest result and retain missing-report warnings."""
    lines = ["", "| Test | Status | Duration |", "| --- | --- | --- |"]
    for name in ("controls-junit.xml", "junit.xml"):
        try:
            root = ElementTree.parse(output / name).getroot()
        except (OSError, ElementTree.ParseError, DefusedXmlException) as exc:
            warnings.append(f"{name}: {exc}")
            continue
        for test in root.iter("testcase"):
            if test.get("name") == "test_camera_latest_frame_does_not_backlog":
                continue
            outcome = next((key for key in ("error", "failure", "skipped") if test.find(key) is not None), "passed")
            lines.append(f"| {cell(test.get('name'))} | {status(outcome)} | {float(test.get('time', '0')):.1f}s |")
            detail = test.find(outcome)
            if detail is not None:
                message = detail.get("message") or detail.text or outcome
                lines.append(f"| ↳ {cell(message[:1000])} | | |")
    return lines


def scored_trials(value: object, source: str, warnings: list[str]) -> list[dict]:
    """Keep valid scored episodes without losing them to incomplete neighboring entries."""
    if not isinstance(value, list):
        warnings.append(f"{source}: expected a list of trials")
        return []
    trials = []
    for index, trial in enumerate(value):
        if (
            not isinstance(trial, dict)
            or not isinstance(trial.get("seed"), (int, str))
            or not isinstance(trial.get("episodes"), list)
        ):
            warnings.append(f"{source}: invalid trial {index}")
            continue
        episodes = []
        for episode in trial["episodes"]:
            if not isinstance(episode, dict) or not all(
                type(episode.get(key)) is int and episode[key] >= 0 for key in ("correct", "wrong", "missed")
            ):
                warnings.append(f"{source}: invalid episode in trial {index}")
                continue
            episodes.append(episode)
        trials.append({**trial, "episodes": episodes})
    return trials


def render_summary(output: Path, artifact_url: str, outcomes: dict[str, str]) -> str:
    """Summarize pytest status, conveyor item accounting, training, and video evidence."""
    lines = ["# 🦾 Conveyor simulation — ACT", "", "## Test results", "", "| Phase | Status |", "| --- | --- |"]
    lines.extend(f"| {cell(name)} | {status(outcome)} |" for name, outcome in outcomes.items())
    warnings = []
    lines += test_results(output, warnings)
    reports = {}
    for name in ("controls", "training", "result", "rollouts"):
        try:
            report = json.loads((output / f"{name}.json").read_text())
        except (OSError, ValueError) as exc:
            warnings.append(f"{name}.json: {exc}")
            continue
        if isinstance(report, list if name == "rollouts" else dict):
            reports[name] = report
        else:
            warnings.append(f"{name}.json: unexpected report type")
    controls = reports.get("controls", {})
    training = reports.get("training", {})
    result = reports.get("result", {})
    lines += ["", "## Checks", "", "| Check | Status |", "| --- | --- |"]
    checks = {
        "Expert beats no-action control": controls.get("status", "not reported"),
        "Training and validation": training.get("status", "not reported"),
        "Deployment wiring": result.get("wiring", "not reported"),
        "Control / camera timing": result.get("timing", "not reported"),
        "Overall deployment result": result.get("status", "not reported"),
    }
    lines.extend(f"| {name} | {status(value)} |" for name, value in checks.items())
    if training:
        if all(
            type(training.get(key)) in {int, float} and math.isfinite(training[key]) and training[key] >= 0
            for key in ("training_epochs", "training_seconds", "validation_loss")
        ):
            lines += [
                "",
                f"**Training:** {training['training_epochs']} epochs · "
                f"{training['training_seconds'] / 60:.1f} minutes · "
                f"validation loss **{training['validation_loss']:.4f}**",
            ]
        else:
            warnings.append("training.json: missing or invalid training metrics")

    drive = controls.get("drive", {})
    if not isinstance(drive, dict):
        warnings.append("controls.json: invalid drive report")
        drive = {}
    policy_trials = scored_trials(result.get("rollouts", []), "result.json rollouts", warnings)
    if not any(trial["episodes"] for trial in policy_trials):
        policy_trials = scored_trials(reports.get("rollouts", []), "rollouts.json", warnings)
    groups = [
        ("Scripted expert", scored_trials(drive.get("trials", []), "controls.json trials", warnings)),
        ("ACT policy", policy_trials),
    ]
    lines += [
        "",
        "## Conveyor sorting",
        "",
        "Counts cover completed, scored episodes only. Partial runs can contain fewer seeds.",
        "",
        "| Controller | Episodes | Correct / total | Wrong | Missed | Correctly sorted |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    details = []
    for label, trials in groups:
        episodes = [episode for trial in trials for episode in trial["episodes"]]
        if not episodes:
            lines.append(f"| {label} | 0 | — | — | — | Not reported |")
            continue
        correct, wrong, missed = (sum(episode[key] for episode in episodes) for key in ("correct", "wrong", "missed"))
        total = correct + wrong + missed
        rate = f"{correct / total:.1%}" if total else "—"
        lines.append(f"| {label} | {len(episodes)} | **{correct} / {total}** | {wrong} | {missed} | **{rate}** |")
        for trial in trials:
            for episode in trial["episodes"]:
                details.append(
                    f"| {label} | {cell(trial['seed'])} | Completed | {episode['correct']} | "
                    f"{episode['wrong']} | {episode['missed']} |"
                )
    lines += [
        "",
        "### Per-seed episodes",
        "",
        "| Controller | Seed | Episode status | Correct | Wrong | Missed |",
        "| --- | --- | --- | --- | --- | --- |",
        *details,
        "",
        "## 🎬 Inference video",
        "",
    ]
    videos = sorted(output.glob("*.mp4"))
    if videos:
        for video in videos:
            label = cell(video.name)
            lines.append(f"- [{label}]({artifact_url})" if artifact_url else f"- {label} — artifact upload unavailable")
        lines += [
            "",
            "Download the **sim-full-act** artifact and open the MP4 for the combined camera views. "
            "Failed runs may contain only a partial recording.",
        ]
    else:
        lines.append("No inference video was produced (the run may have stopped before deployment).")
    if artifact_url:
        lines += ["", f"[Download all simulation evidence]({artifact_url}) · retained for 14 days."]
    if warnings:
        lines += ["", "<details><summary>Missing or unreadable evidence</summary>", ""]
        lines.extend(f"- {cell(warning)}" for warning in warnings)
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    summary = render_summary(
        Path(os.environ["SIM_E2E_OUTPUT_DIR"]),
        os.environ.get("SIM_E2E_ARTIFACT_URL", ""),
        {
            "Seeded controls, device registration and camera freshness": os.environ.get(
                "SIM_E2E_CONTROLS_OUTCOME", "not run"
            ),
            "Fresh training, export and deployment": os.environ.get("SIM_E2E_TRAINING_OUTCOME", "not run"),
        },
    )
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
        stream.write(summary)
