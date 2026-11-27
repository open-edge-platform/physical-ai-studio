# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The model worker writes its records, with their tracebacks, to the file the Logs page reads for it."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

import pytest

from runtime_session.bus import Bus
from runtime_session.store import FeatureSpec, ImageSpec, Store
from services.log_service import STATIC_SOURCES
from workers.base import WorkerStartupError

from .fake_model_named import SENTINEL, ModelWorker

if TYPE_CHECKING:
    from pathlib import Path

SPEC = FeatureSpec(("shoulder_pan.pos", "gripper.pos"), {"cam 0": ImageSpec(120, 160, 3)})


@pytest.fixture
def log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # The spawned child resolves its settings, and with them its log folder, from the environment.
    storage = tmp_path / "storage"
    monkeypatch.setenv("STORAGE_DIR", str(storage))
    return storage / "logs"


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _failure(path: Path) -> dict:
    (failure,) = [entry for entry in _records(path) if entry["record"]["exception"] is not None]
    return failure


async def test_a_model_that_fails_to_load_is_logged_with_its_traceback(log_dir: Path, tmp_path: Path) -> None:
    export = tmp_path / "model" / "exports" / "torch"
    export.mkdir(parents=True)
    (export / "broken").touch()
    session_id = uuid.uuid4().hex
    store = Store.create(SPEC, uuid.uuid4().hex)
    try:
        with Bus.host(session_id):
            worker = ModelWorker(
                export,
                backend="torch",
                device="cpu",
                session_id=session_id,
                store_name=store.name,
                spec=SPEC,
                images={"images.cam 0": "cam 0"},
            )
            with pytest.raises(WorkerStartupError, match="does not load"):
                async with worker:
                    pass
    finally:
        store.unlink()

    failure = _failure(log_dir / STATIC_SOURCES["inference"].filename)
    assert failure["record"]["message"] == f"Worker {worker.name} failed"
    assert failure["record"]["extra"]["worker"] == "ModelWorker"
    assert failure["record"]["exception"]["type"] == "RuntimeError"
    assert "Traceback (most recent call last):" in failure["text"]
    assert "in open_model" in failure["text"]
    # The file keeps the traceback but not the values of its local variables.
    assert SENTINEL not in failure["text"]
    assert "RuntimeError: the export does not load" in failure["text"]
