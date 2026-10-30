"""Run a model's inference in the model worker's own process.

The worker loads one of a model's exports and serves ``rpc/model/infer``: the
request carries the state and the task, the worker adds each camera's latest
frame from the store, and replies with the whole action chunk. What to do with
the chunk (queueing, smoothing, when to ask again) is up to the executor in the
session's ``ModelControl``, which keeps writing ``action`` itself.

Its own process, because preprocessing and the runtimes' Python side hold the
GIL for long stretches, which would stall the API process's event loop.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from pydantic import BaseModel

from runtime_session.bus import Bus
from runtime_session.store import FeatureSpec, Store
from schemas.runtime_session import CommandDone
from workers.base import ProcessWorker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

INFER = "rpc/model/infer"
RESET = "rpc/model/reset"
READY = "rpc/model/ready"

# The model input names physicalai.inference expects.
STATE = "state"
TASK = "task"


class Empty(BaseModel):
    pass


class InferRequest(BaseModel):
    state: list[float]
    task: str | None = None


class ActionChunk(BaseModel):
    actions: list[list[float]]
    """``(chunk_size, action_dim)``."""


class Predictor(Protocol):
    """What the worker needs of a loaded model; ``physicalai.inference.InferenceModel`` is one."""

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray: ...
    def reset(self) -> None: ...


@dataclass
class Inference:
    """The worker's loaded model; only touched on the worker's event loop."""

    model: Predictor
    store: Store
    images: Mapping[str, str]
    """Model input name to the store camera it reads, such as ``images.gripper`` to ``gripper``."""

    def infer(self, request: InferRequest) -> ActionChunk:
        observation: dict[str, Any] = {STATE: np.array([request.state], dtype=np.float32)}
        for name, camera in self.images.items():
            frame = self.store.read_observation(camera)
            if frame is None:
                raise RuntimeError(f"Camera {camera!r} has no frame yet")
            observation[name] = frame.value[np.newaxis]
        if request.task is not None:
            observation[TASK] = [request.task]
        return ActionChunk(actions=np.asarray(self.model.predict_action_chunk(observation)).tolist())

    def reset(self, _request: Empty) -> CommandDone:
        self.model.reset()
        return CommandDone()


class ModelWorker(ProcessWorker[Inference]):
    ready_timeout_s = 300.0
    """Loading compiles or moves the model onto its device, which can take minutes."""

    def __init__(
        self,
        export_dir: Path,
        *,
        backend: str,
        device: str,
        session_id: str,
        store_name: str,
        spec: FeatureSpec,
        images: Mapping[str, str],
    ) -> None:
        super().__init__(name=f"ModelWorker-{export_dir.parent.parent.name}")
        self.export_dir = export_dir
        self.backend = backend
        self.device = device
        self.session_id = session_id
        self.store_name = store_name
        self.spec = spec
        self.images = dict(images)

    def open_model(self) -> Predictor:
        """Runs in the child. Tests replace this to infer without a real export."""
        from physicalai.inference import InferenceModel

        return InferenceModel(self.export_dir, backend=self.backend, device=self.device)

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[Inference]:
        store = Store.attach(self.spec, self.store_name)
        try:
            with Bus.join(self.session_id) as bus:
                inference = Inference(self.open_model(), store, self.images)
                bus.serve(INFER, InferRequest, inference.infer)
                bus.serve(RESET, Empty, inference.reset)
                bus.serve(READY, Empty, lambda _request: CommandDone())
                yield inference
        finally:
            store.close()

    async def run_loop(self, context: Inference) -> None:  # noqa: ARG002
        # Requests are served on this loop; nothing else to do between them.
        while not self.should_stop():
            await self.sleep(0.1)
