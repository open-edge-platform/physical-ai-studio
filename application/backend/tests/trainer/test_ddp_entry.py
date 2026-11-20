# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the torchrun entry point of multi-device training jobs."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from trainer import ddp_entry
from trainer.ddp_entry import EventLog, build_command, load_request
from trainer.schemas import SubmitJobRequest
from training import RunOptions, TrainingJobSpec

if TYPE_CHECKING:
    from pathlib import Path

ENTRY = "trainer.ddp_entry"
_JOB_ID = "3f6c1c1e-0e5a-4f10-9d0f-2b8b3a4c5d6e"


def _ddp_request() -> SubmitJobRequest:
    return SubmitJobRequest(spec=TrainingJobSpec(policy="act", device_type="cuda", all_devices=True))


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def request_file(tmp_path: Path) -> Path:
    path = tmp_path / "request.json"
    path.write_text(_ddp_request().model_dump_json())
    return path


@pytest.fixture
def runner():
    with patch(f"{ENTRY}.TrainerRunner") as runner_class:
        yield runner_class.return_value


class TestRankZero:
    def test_progress_and_completion_are_written_as_events(self, tmp_path: Path, runner) -> None:
        def fake_run(_job_id, _request, *, should_stop, report):
            assert should_stop() is False
            report(0, "Dataset ready", None)
            report(50, None, {"train/loss_step": 1.5})
            return tmp_path / "archive.zip"

        runner.run.side_effect = fake_run
        events_path = tmp_path / "events.jsonl"

        ddp_entry.run(_JOB_ID, _ddp_request(), EventLog(events_path), rank=0)

        assert _events(events_path) == [
            {"type": "progress", "progress": 0, "message": "Dataset ready", "extra_info": None},
            {"type": "progress", "progress": 50, "message": None, "extra_info": {"train/loss_step": 1.5}},
            {"type": "completed", "archive": str(tmp_path / "archive.zip")},
        ]
        runner.train_worker_rank.assert_not_called()

    def test_a_failure_is_recorded_before_it_propagates(self, tmp_path: Path, runner) -> None:
        """torchrun only reports an exit code; the events file carries the reason."""
        runner.run.side_effect = RuntimeError("CUDA out of memory")
        events_path = tmp_path / "events.jsonl"

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            ddp_entry.run(_JOB_ID, _ddp_request(), EventLog(events_path), rank=0)

        assert _events(events_path) == [{"type": "failed", "error": "CUDA out of memory"}]


class TestOtherRanks:
    def test_train_without_reporting_or_archiving(self, tmp_path: Path, runner) -> None:
        events_path = tmp_path / "events.jsonl"
        request = _ddp_request()

        ddp_entry.run(_JOB_ID, request, EventLog(events_path), rank=1)

        runner.train_worker_rank.assert_called_once_with(_JOB_ID, request)
        runner.run.assert_not_called()
        assert not events_path.exists()

    def test_a_failure_propagates_without_an_event(self, tmp_path: Path, runner) -> None:
        """Rank 0 owns the events file; torchrun tears the job down on any rank's exit code."""
        runner.train_worker_rank.side_effect = RuntimeError("NCCL timeout")
        events_path = tmp_path / "events.jsonl"

        with pytest.raises(RuntimeError):
            ddp_entry.run(_JOB_ID, _ddp_request(), EventLog(events_path), rank=1)

        assert not events_path.exists()


class TestLoadRequest:
    def test_the_hf_token_comes_from_the_environment(self, request_file: Path, monkeypatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf-secret")

        request = load_request(request_file)

        assert request.spec.run_options.hf_token.get_secret_value() == "hf-secret"
        assert request.spec.all_devices is True

    def test_a_masked_token_in_the_file_never_reaches_training(self, tmp_path: Path, monkeypatch) -> None:
        """Dumping a request with run_options set writes the token as '**********'."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        request = _ddp_request()
        request.spec.run_options = RunOptions(hf_token=SecretStr("hf-secret"))
        path = tmp_path / "request.json"
        path.write_text(request.model_dump_json())
        assert "hf-secret" not in path.read_text()

        assert load_request(path).spec.run_options.hf_token is None


class TestMain:
    def test_the_rank_comes_from_torchrun(self, request_file: Path, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("RANK", "1")
        with patch(f"{ENTRY}.run") as run, patch(f"{ENTRY}.setup_logging"):
            ddp_entry.main(["--job-id", _JOB_ID, "--request", str(request_file), "--events", str(tmp_path / "e.jsonl")])

        assert run.call_args.args[0] == _JOB_ID
        assert run.call_args.kwargs["rank"] == 1


def test_build_command_launches_this_module_under_torchrun(tmp_path: Path) -> None:
    command = build_command(_JOB_ID, tmp_path / "request.json", tmp_path / "events.jsonl", nproc=4)

    assert command == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "-m",
        "trainer.ddp_entry",
        "--job-id",
        _JOB_ID,
        "--request",
        str(tmp_path / "request.json"),
        "--events",
        str(tmp_path / "events.jsonl"),
    ]
