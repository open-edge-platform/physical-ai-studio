"""A model worker whose model is an oracle instead of an export, for the model tests.

Module-level so the spawned child can import it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from physicalai.inference.preprocessors.observation_history import StackObservationHistory

from workers.model_worker import ModelWorker

CHUNK = 10


class Oracle:
    """Commands ``state + 1 .. state + CHUNK``, so a follower that follows it keeps climbing."""

    def __init__(self, images: list[str], steps: int | None = None) -> None:
        self.images = images
        self.steps = steps

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        if self.steps is not None:
            if not all(isinstance(observation[name], tuple) for name in ["state", *self.images]):
                raise ValueError("History was not deferred")
            observation = StackObservationHistory()(observation)
        for name in self.images:
            if observation[name].shape[0] != 1:
                raise ValueError(f"{name} is not one frame")
            if self.steps is not None and observation[name].shape[1] != self.steps:
                raise ValueError("Wrong history length")
        state = observation["state"][0]
        if self.steps is not None:
            if state.shape != (self.steps, 2):
                raise ValueError("Wrong state history shape")
            state = state[-1]
        return np.stack([state + step for step in range(1, CHUNK + 1)])

    def reset(self) -> None:
        pass


class FakeModelWorker(ModelWorker):
    def open_model(self) -> Oracle:  # type: ignore[override]
        if (self.export_dir / "broken").exists():
            raise RuntimeError("the export does not load")
        return Oracle(list(self.images), self.observation_history.n_obs_steps if self.observation_history else None)
