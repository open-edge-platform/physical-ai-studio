# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline checks for GitHub simulation summaries."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[4] / ".github/scripts/simulation_summary.py"
render_summary = runpy.run_path(str(SCRIPT))["render_summary"]
ARTIFACT_URL = "https://github.com/open-edge-platform/physical-ai-studio/actions/runs/123/artifacts/456"


def trial(seed, correct, wrong, missed):
    return {"seed": seed, "episodes": [{"correct": correct, "wrong": wrong, "missed": missed}]}


def test_summary_shows_tests_counts_checks_and_video(tmp_path):
    (tmp_path / "controls-junit.xml").write_text(
        '<testsuites><testsuite><testcase name="test_controls" time="2.1" />'
        '<testcase name="test_camera_latest_frame_does_not_backlog" time="2.5" /></testsuite></testsuites>'
    )
    (tmp_path / "junit.xml").write_text(
        '<testsuites><testsuite><testcase name="test_deployment" time="10" /></testsuite></testsuites>'
    )
    (tmp_path / "controls.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "off": {"trials": [trial(42, 0, 0, 10)]},
                "drive": {"trials": [trial(42, 10, 0, 0)]},
            }
        )
    )
    (tmp_path / "training.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "training_epochs": 5,
                "training_seconds": 120,
                "validation_loss": 0.1743,
            }
        )
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "quality": "passed",
                "timing": "passed",
                "wiring": "passed",
                "rollouts": [trial(42, 9, 0, 1), trial(43, 8, 1, 1)],
            }
        )
    )
    (tmp_path / "rollouts.json").write_text("[]")
    (tmp_path / "inference-act-conveyor_sort.mp4").write_bytes(b"video")
    text = render_summary(tmp_path, ARTIFACT_URL, {"Controls": "success", "Deployment": "success"})
    assert [line for line in text.splitlines() if line.startswith("| test_")] == [
        "| test_controls | ✅ passed | 2.1s |",
        "| test_deployment | ✅ passed | 10.0s |",
    ]
    assert "| ACT policy | 2 | **17 / 20** | 1 | 2 | **85.0%** |" in text
    assert "| Scripted expert | 1 | **10 / 10** | 0 | 0 | **100.0%** |" in text
    assert "| ACT policy | 43 | Completed | 8 | 1 | 1 |" in text
    assert "| Control / camera timing | ✅ passed |" in text
    assert "5 epochs · 2.0 minutes · validation loss **0.1743**" in text
    assert f"[inference-act-conveyor_sort.mp4]({ARTIFACT_URL})" in text


@pytest.mark.parametrize("outcome,icon", [("failure", "❌"), ("error", "❌"), ("skipped", "⏭️")])
def test_summary_preserves_unsuccessful_test_status_and_escapes_messages(tmp_path, outcome, icon):
    (tmp_path / "junit.xml").write_text(
        f'<testsuite><testcase name="test_bad" time="1"><{outcome} '
        'message="bad | input&#10;&lt;script&gt;" /></testcase></testsuite>'
    )
    text = render_summary(tmp_path, "", {"Deployment": "failure"})
    assert f"| test_bad | {icon} {outcome} | 1.0s |" in text
    assert "bad &#124; input<br>&lt;script&gt;" in text
    assert "| Deployment | ❌ failure |" in text
    assert "| Overall deployment result | ⚠️ not reported |" in text


def test_summary_uses_partial_rollouts_and_reports_corrupt_evidence(tmp_path):
    (tmp_path / "controls.json").write_text('{"status": "incomplete"}')
    (tmp_path / "result.json").write_text("broken")
    (tmp_path / "junit.xml").write_text("<broken")
    (tmp_path / "rollouts.json").write_text(json.dumps([trial(42, 7, 1, 2)]))
    (tmp_path / "inference-act-conveyor_sort.mp4").write_bytes(b"video")
    text = render_summary(tmp_path, "", {"Controls": "failure", "Deployment": "skipped"})
    assert "| Expert beats no-action control | ⚠️ incomplete |" in text
    assert "| ACT policy | 1 | **7 / 10** | 1 | 2 | **70.0%** |" in text
    assert "Missing or unreadable evidence" in text
    assert "result.json:" in text and "junit.xml:" in text
    assert "artifact upload unavailable" in text


def test_summary_reports_forbidden_xml_entities_and_continues(tmp_path):
    (tmp_path / "controls-junit.xml").write_text(
        '<!DOCTYPE testsuite [<!ENTITY payload "expanded">]><testsuite><testcase name="&payload;" /></testsuite>'
    )
    (tmp_path / "junit.xml").write_text('<testsuite><testcase name="test_deployment" time="1" /></testsuite>')
    text = render_summary(tmp_path, "", {"Deployment": "success"})
    assert "controls-junit.xml: EntitiesForbidden" in text
    assert "| test_deployment | ✅ passed | 1.0s |" in text
    assert "## Conveyor sorting" in text


def test_summary_cli_appends_even_when_setup_produced_no_evidence(tmp_path):
    summary = tmp_path / "summary.md"
    summary.write_text("Existing summary\n")
    subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=True,
        env={
            **os.environ,
            "GITHUB_STEP_SUMMARY": str(summary),
            "SIM_E2E_OUTPUT_DIR": str(tmp_path / "missing"),
            "SIM_E2E_ARTIFACT_URL": "",
            "SIM_E2E_CONTROLS_OUTCOME": "skipped",
            "SIM_E2E_TRAINING_OUTCOME": "skipped",
        },
    )
    text = summary.read_text()
    assert text.startswith("Existing summary\n# 🦾")
    assert "⏭️ skipped" in text
    assert "No inference video was produced" in text
