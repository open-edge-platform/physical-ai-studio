"""Run a model's inference in the model worker's own process.

The worker loads one of a model's exports and serves ``rpc/model/infer``: the
request carries the state and the task, the worker adds each camera's latest
frame from the store, and replies with the whole action chunk. What to do with
the chunk (queueing, smoothing, when to ask again) is up to the executor in the
session's ``ModelControl``, which keeps writing ``action`` itself.

An export made with RTC (real-time chunking) inputs also takes the unplayed
tail of the previous chunk and the guidance settings. RTC execution sends them
with the request; sync and async execution do not, and the worker then fills
in the neutral values RTC itself sends for a first chunk (no tail, guidance
weight 1e-8), so such an export runs in every mode.

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
PREV_CHUNK_LEFT_OVER = "prev_chunk_left_over"
INFERENCE_DELAY = "inference_delay"
MAX_GUIDANCE_WEIGHT = "max_guidance_weight"
EXECUTION_HORIZON = "execution_horizon"
RTC_INPUTS = (PREV_CHUNK_LEFT_OVER, INFERENCE_DELAY, MAX_GUIDANCE_WEIGHT, EXECUTION_HORIZON)
"""The inputs an RTC export takes besides the observation."""


class Empty(BaseModel):
    pass


class RtcInputs(BaseModel):
    """What ``RTCExecution`` adds to an observation, as ``physicalai.runtime`` names them."""

    prev_chunk_left_over: list[list[float]]
    """``(chunk_size, action_dim)``: the queued actions not yet played, padded with zeros."""
    inference_delay: int
    max_guidance_weight: float
    execution_horizon: int


class InferRequest(BaseModel):
    state: list[float]
    task: str | None = None
    rtc: RtcInputs | None = None


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
    rtc_chunk: tuple[int, int] | None = None
    """``prev_chunk_left_over``'s shape when the export takes RTC inputs."""

    def infer(self, request: InferRequest) -> ActionChunk:
        observation: dict[str, Any] = {STATE: np.array([request.state], dtype=np.float32)}
        observation.update(self._rtc_inputs(request.rtc))
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

    def _rtc_inputs(self, rtc: RtcInputs | None) -> dict[str, Any]:
        if self.rtc_chunk is None:
            if rtc is not None:
                # The export would drop them without a word and run unguided.
                raise ValueError("This export takes no RTC inputs; it cannot run with RTC execution")
            return {}
        if rtc is None:  # sync or async: what RTC sends for a first chunk, which leaves the chunk unguided
            return {
                PREV_CHUNK_LEFT_OVER: np.zeros((1, *self.rtc_chunk), dtype=np.float32),
                INFERENCE_DELAY: np.int64(0),
                MAX_GUIDANCE_WEIGHT: np.float32(1e-8),
                EXECUTION_HORIZON: np.int64(1),
            }
        left_over = np.asarray(rtc.prev_chunk_left_over, dtype=np.float32)
        if left_over.shape != self.rtc_chunk:
            raise ValueError(f"prev_chunk_left_over is {left_over.shape}; the export takes {self.rtc_chunk}")
        return {
            PREV_CHUNK_LEFT_OVER: left_over[np.newaxis],
            INFERENCE_DELAY: np.int64(rtc.inference_delay),
            MAX_GUIDANCE_WEIGHT: np.float32(rtc.max_guidance_weight),
            EXECUTION_HORIZON: np.int64(rtc.execution_horizon),
        }


class ModelWorker(ProcessWorker[Inference]):
    ready_timeout_s = 300.0
    """Loading compiles or moves the model onto its device, which can take minutes."""

    def __init__(  # noqa: PLR0913 - one flat option per thing the child needs, as it is pickled across
        self,
        export_dir: Path,
        *,
        backend: str,
        device: str,
        session_id: str,
        store_name: str,
        spec: FeatureSpec,
        images: Mapping[str, str],
        rtc_chunk: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(name=f"ModelWorker-{export_dir.parent.parent.name}")
        self.export_dir = export_dir
        self.backend = backend
        self.device = device
        self.session_id = session_id
        self.store_name = store_name
        self.spec = spec
        self.images = dict(images)
        self.rtc_chunk = rtc_chunk

    def open_model(self) -> Predictor:
        """Runs in the child. Tests replace this to infer without a real export."""
        from physicalai.inference import InferenceModel

        return InferenceModel(self.export_dir, backend=self.backend, device=self.device)

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[Inference]:
        store = Store.attach(self.spec, self.store_name)
        try:
            with Bus.join(self.session_id) as bus:
                inference = Inference(self.open_model(), store, self.images, self.rtc_chunk)
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
