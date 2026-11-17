"""A model worker whose model is an oracle instead of an export, for the model tests.

Module-level so the spawned child can import it.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from workers.model_worker import RTC_INPUTS, ModelWorker

CHUNK = 10


class Oracle:
    """Commands ``state + 1 .. state + CHUNK``, so a follower that follows it keeps climbing."""

    def __init__(self, images: list[str], *, rtc: bool = False, delay_s: float = 0.0) -> None:
        self.images = images
        self.rtc = rtc
        self.delay_s = delay_s

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        for name in self.images:
            if observation[name].shape[0] != 1:
                raise ValueError(f"{name} is not one frame")
        if self.rtc and not all(key in observation for key in RTC_INPUTS):
            # A real RTC export fails the same way without them.
            raise KeyError(f"Expected inputs {RTC_INPUTS}, got {sorted(observation)}")
        time.sleep(self.delay_s)
        state = observation["state"][0]
        return np.stack([state + step for step in range(1, CHUNK + 1)])

    def reset(self) -> None:
        pass


class FakeModelWorker(ModelWorker):
    def open_model(self) -> Oracle:  # type: ignore[override]
        if (self.export_dir / "broken").exists():
            raise RuntimeError("the export does not load")
        # A "slow" file in the export makes each inference take 0.5 s, as a real model's does.
        delay_s = 0.5 if (self.export_dir / "slow").exists() else 0.0
        return Oracle(list(self.images), rtc=self.rtc_chunk is not None, delay_s=delay_s)
