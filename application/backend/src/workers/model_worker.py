"""Run a model's inference in the model worker's own process.

The worker loads one of a model's exports and serves ``rpc/model/infer``: the
request carries the state and the task, the worker adds each camera's latest
frame from the store (or copies the request's selected history), and replies
with the whole action chunk. What to do with
the chunk (queueing, smoothing, when to ask again) is up to the executor in the
session's ``ModelControl``, which keeps writing ``action`` itself.

Its own process, because preprocessing and the runtimes' Python side hold the
GIL for long stretches, which would stall the API process's event loop.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from physicalai.inference.preprocessors.observation_history import ObservationWindow
from pydantic import BaseModel

from runtime_session.bus import Bus
from runtime_session.store import FeatureSpec, Store
from schemas.runtime_session import CommandDone
from workers.base import ProcessWorker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

    from physicalai.inference.manifest import ObservationHistorySpec

INFER = "rpc/model/infer"
RESET = "rpc/model/reset"
READY = "rpc/model/ready"

# The model input names physicalai.inference expects.
STATE = "state"
TASK = "task"


class Empty(BaseModel):
    pass


class InferRequest(BaseModel):
    state: list[float] | list[list[float]]
    task: str | None = None
    history_images: str | None = None


def history_image_layout(
    spec: FeatureSpec, images: Mapping[str, str], steps: int
) -> tuple[dict[str, tuple[tuple[int, ...], int]], int]:
    """Derive uint8 window shapes and offsets from trusted worker configuration."""
    layout: dict[str, tuple[tuple[int, ...], int]] = {}
    size = 0
    for name, camera in sorted(images.items()):
        shape = (steps, 1, *spec.observations[camera].shape)
        layout[name] = (shape, size)
        size += int(np.prod(shape))
    return layout, size


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
    observation_history: ObservationHistorySpec | None = None
    history_prefix: str = ""

    def infer(self, request: InferRequest) -> ActionChunk:
        if self.observation_history is not None:
            observation: dict[str, Any] = self._history_input(request)
            return ActionChunk(actions=np.asarray(self.model.predict_action_chunk(observation)).tolist())
        if request.history_images is not None or np.asarray(request.state).ndim != 1:
            raise ValueError("Unexpected observation history request")
        observation = {STATE: np.array([request.state], dtype=np.float32)}
        for name, camera in self.images.items():
            frame = self.store.read_observation(camera)
            if frame is None:
                raise RuntimeError(f"Camera {camera!r} has no frame yet")
            observation[name] = frame.value[np.newaxis]
        if request.task is not None:
            observation[TASK] = [request.task]
        return ActionChunk(actions=np.asarray(self.model.predict_action_chunk(observation)).tolist())

    def _history_input(self, request: InferRequest) -> dict[str, Any]:
        history = self.observation_history
        if history is None:
            raise ValueError("Observation history is not enabled")
        steps = history.n_obs_steps
        state = np.asarray(request.state, dtype=np.float32)
        if state.shape != (steps, len(self.store.spec.joint_names)):
            raise ValueError("Invalid observation history state shape")
        observation: dict[str, Any] = {STATE: ObservationWindow(row[np.newaxis] for row in state)}
        layout, size = history_image_layout(self.store.spec, self.images, steps)
        if size:
            name = request.history_images
            suffix = name.removeprefix(self.history_prefix) if name is not None else ""
            if (
                not self.history_prefix
                or name is None
                or not name.startswith(self.history_prefix)
                or len(suffix) != 8
                or any(char not in "0123456789abcdef" for char in suffix)
            ):
                raise ValueError("Invalid observation history shared memory namespace")
            block = shared_memory.SharedMemory(name=name, track=False)
            try:
                if block.size != size:
                    raise ValueError("Invalid observation history shared memory size")
                for key, (shape, offset) in layout.items():
                    # No shared views survive attachment: inference owns the selected window.
                    owned: np.ndarray = np.ndarray(shape, np.uint8, block.buf, offset).copy()
                    observation[key] = ObservationWindow(owned)
            finally:
                block.close()
        elif request.history_images is not None:
            raise ValueError("Unexpected observation history images")
        for window in observation.values():
            for frame in window:
                frame.setflags(write=False)
        if request.task is not None:
            observation[TASK] = [request.task]
        return observation

    def reset(self, _request: Empty) -> CommandDone:
        self.model.reset()
        return CommandDone()


class ModelWorker(ProcessWorker[Inference]):
    ready_timeout_s = 300.0
    """Loading compiles or moves the model onto its device, which can take minutes."""

    def __init__(  # noqa: PLR0913
        self,
        export_dir: Path,
        *,
        backend: str,
        device: str,
        session_id: str,
        store_name: str,
        spec: FeatureSpec,
        images: Mapping[str, str],
        observation_history: ObservationHistorySpec | None = None,
        history_prefix: str = "",
    ) -> None:
        super().__init__(name=f"ModelWorker-{export_dir.parent.parent.name}")
        self.export_dir = export_dir
        self.backend = backend
        self.device = device
        self.session_id = session_id
        self.store_name = store_name
        self.spec = spec
        self.images = dict(images)
        self.observation_history = observation_history
        self.history_prefix = history_prefix

    def open_model(self) -> Predictor:
        """Runs in the child. Tests replace this to infer without a real export."""
        from physicalai.inference import InferenceModel

        return InferenceModel(self.export_dir, backend=self.backend, device=self.device)

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[Inference]:
        store = Store.attach(self.spec, self.store_name)
        try:
            with Bus.join(self.session_id) as bus:
                inference = Inference(
                    self.open_model(), store, self.images, self.observation_history, self.history_prefix
                )
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
