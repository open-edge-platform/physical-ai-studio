"""A model worker whose model is an oracle instead of an export, for the model tests.

Module-level so the spawned child can import it.

Files in the export steer the oracle from the test process: ``broken`` fails the load, ``slow`` makes each
inference take 0.5 s, ``fail`` makes each inference raise, ``record`` appends what each inference received and
returned to ``received.jsonl``, and ``gate_<n>`` holds inference ``n`` (counted from 1) until ``release_<n>``
exists, after writing ``entered_<n>``.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from workers.model_worker import (
    EXECUTION_HORIZON,
    INFERENCE_DELAY,
    MAX_GUIDANCE_WEIGHT,
    PREV_CHUNK_LEFT_OVER,
    RTC_INPUTS,
    ModelWorker,
)

if TYPE_CHECKING:
    from pathlib import Path

CHUNK = 10
NO_FRAME = "Camera 'cam 0' has no frame yet"


class Oracle:
    """Commands ``state + 1 .. state + CHUNK``, so a follower that follows it keeps climbing.

    While recording, inference ``n`` adds ``n / 1000`` to every action, so a test can tell whose chunk was played.
    """

    def __init__(
        self, images: list[str], *, rtc: bool = False, delay_s: float = 0.0, export_dir: Path | None = None
    ) -> None:
        self.images = images
        self.rtc = rtc
        self.delay_s = delay_s
        self.export_dir = export_dir
        self.calls = 0

    def _flag(self, name: str) -> bool:
        return self.export_dir is not None and (self.export_dir / name).exists()

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        self.calls += 1
        n = self.calls
        for name in self.images:
            if observation[name].shape[0] != 1:
                raise ValueError(f"{name} is not one frame")
        if self.rtc and not all(key in observation for key in RTC_INPUTS):
            # A real RTC export fails the same way without them.
            raise KeyError(f"Expected inputs {RTC_INPUTS}, got {sorted(observation)}")
        if self._flag("fail"):
            raise ValueError(NO_FRAME)
        if self._flag(f"gate_{n}"):
            assert self.export_dir is not None
            (self.export_dir / f"entered_{n}").touch()
            deadline = time.monotonic() + 30
            while not self._flag(f"release_{n}") and time.monotonic() < deadline:
                time.sleep(0.01)
        time.sleep(self.delay_s)
        state = observation["state"][0]
        chunk = np.stack([state + step for step in range(1, CHUNK + 1)]).astype(np.float32)
        if self._flag("record"):
            assert self.export_dir is not None
            chunk += np.float32(n / 1000)
            rtc = None
            if self.rtc:
                rtc = {
                    "prev_chunk_left_over": np.asarray(observation[PREV_CHUNK_LEFT_OVER])[0].tolist(),
                    "inference_delay": int(observation[INFERENCE_DELAY]),
                    "max_guidance_weight": float(observation[MAX_GUIDANCE_WEIGHT]),
                    "execution_horizon": int(observation[EXECUTION_HORIZON]),
                }
            row = {"n": n, "state": state.tolist(), "rtc": rtc, "chunk": chunk.tolist()}
            with (self.export_dir / "received.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
        return chunk

    def reset(self) -> None:
        pass


class FakeModelWorker(ModelWorker):
    def open_model(self) -> Oracle:  # type: ignore[override]
        if (self.export_dir / "broken").exists():
            raise RuntimeError("the export does not load")
        # A "slow" file in the export makes each inference take 0.5 s, as a real model's does.
        delay_s = 0.5 if (self.export_dir / "slow").exists() else 0.0
        if (self.export_dir / "plain_graph").exists():
            # A plain graph under a stale manifest that lists RTC inputs: its adapter says what it takes.
            oracle = Oracle(list(self.images), delay_s=delay_s, export_dir=self.export_dir)
            oracle.adapter = SimpleNamespace(input_names=["state", *self.images])  # type: ignore[attr-defined]
            return oracle
        return Oracle(list(self.images), rtc=self.rtc_chunk is not None, delay_s=delay_s, export_dir=self.export_dir)
