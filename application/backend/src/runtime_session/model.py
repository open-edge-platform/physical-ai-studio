"""A model loaded into a session: inference in its own process, called like a local model.

A :class:`ProcessModel` starts a ``ModelWorker`` that loads one of the model's
exports, and reports as ``model`` on the session's bus. To ``ModelControl``'s
executor it is the model itself: ``predict_action_chunk`` sends the state and
the task to the worker, which adds the frames from the store, and returns the
whole chunk. Queueing it, smoothing and deciding when to ask again stays with
the executor, as ``physicalai.runtime`` runs it.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from runtime_session.bus import Bus, publish_status, until_serving
from schemas.runtime_session import CommandDone, SessionModel
from workers import model_worker as model_rpc
from workers.model_worker import ActionChunk, InferRequest, ModelWorker

if TYPE_CHECKING:
    from runtime_session.store import FeatureSpec, Store
    from schemas.model import Model

COMPONENT = "model"
"""What a loaded model reports as."""

_READY_S = 10.0
"""How long a loaded model's worker has to answer on the bus."""

_INFER_S = 30.0
"""A backstop for one inference; a slow first one compiles."""

_IMAGES = "images"


@dataclass(frozen=True)
class ModelInputs:
    """What an export takes, from its manifest: the state's size and the cameras it reads."""

    state_size: int
    images: dict[str, str]
    """Model input name to the store camera it reads."""


def model_inputs(export_dir: Path, spec: FeatureSpec) -> ModelInputs:
    """What the export at ``export_dir`` reads from a store with ``spec``.

    Raises:
        ValueError: The export needs joints or cameras ``spec`` lacks.
    """
    manifest = json.loads((export_dir / "manifest.json").read_text())
    features = [feature["init_args"] for feature in manifest["model"]["input_features"]]
    reasons: list[str] = []
    state_size = 0
    images: dict[str, str] = {}
    for feature in features:
        name: str = feature["name"]
        if feature["ftype"] == "STATE":
            state_size = int(feature["shape"][0])
            if state_size != len(spec.joint_names):
                reasons.append(f"it takes {state_size} joints, the follower has {len(spec.joint_names)}")
        elif feature["ftype"] == "VISUAL":
            camera = name.removeprefix(f"{_IMAGES}.") if name != _IMAGES else next(iter(spec.observations), "")
            # shortcut: checks the camera is there, not its resolution; the export may resize. Compare once manifests
            # say what they resize from.
            if camera not in spec.observations:
                reasons.append(f"it reads camera {camera or name!r}, which the environment lacks")
            images[name] = camera
    if reasons:
        raise ValueError("The model does not fit this environment: " + "; ".join(reasons))
    return ModelInputs(state_size, images)


class ProcessModel:
    """A model's export loaded in a ``ModelWorker``.

    ``async with`` starts the worker, waits until the model is loaded and answers,
    and reports ``running``; leaving stops it and reports ``stopped``. A worker
    that dies is reported ``failed``.
    """

    def __init__(self, model: Model, session_id: str, store: Store, *, backend: str, device: str) -> None:
        if backend not in model.available_backends:
            raise ValueError(f"Model {model.name!r} has no {backend} export; it has {model.available_backends}")
        export_dir = Path(model.path) / "exports" / backend
        self.model = model
        self.backend = backend
        self.device = device
        self.inputs = model_inputs(export_dir, store.spec)
        self._worker = ModelWorker(
            export_dir,
            backend=backend,
            device=device,
            session_id=session_id,
            store_name=store.name,
            spec=store.spec,
            images=self.inputs.images,
        )
        self._session_id = session_id
        self._bus: Bus | None = None
        self._stack: AsyncExitStack | None = None
        self._started = False

    def state(self) -> SessionModel:
        return SessionModel(model_id=self.model.id, name=self.model.name, backend=self.backend, device=self.device)

    async def __aenter__(self) -> Self:
        """Start the worker and wait until the model is loaded.

        Raises:
            WorkerStartupError: The model could not be loaded.
        """
        async with AsyncExitStack() as stack:
            bus = stack.enter_context(Bus.join(self._session_id))
            publish_status(bus, COMPONENT, "starting")
            stack.callback(self._report_stopped, bus)
            try:
                await stack.enter_async_context(self._worker)
                await until_serving(bus, model_rpc.READY, model_rpc.Empty(), CommandDone, within=_READY_S)
            except Exception as error:
                publish_status(bus, COMPONENT, "failed", str(error))
                raise
            watch = asyncio.create_task(self._report_death(bus))
            stack.callback(watch.cancel)
            publish_status(bus, COMPONENT, "running")
            self._started = True
            self._bus = bus
            self._stack = stack.pop_all()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        stack, self._stack, self._bus = self._stack, None, None
        if stack is not None:
            await stack.aclose()

    # What an executor calls, from its own thread.

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        """The action chunk for ``observation``'s state and task, with the store's latest frames.

        Raises:
            BusRequestError: The model failed on it.
            NoResponderError: The model is no longer loaded.
        """
        task = observation.get(model_rpc.TASK)
        request = InferRequest(
            state=np.asarray(observation[model_rpc.STATE])[0].tolist(), task=task[0] if task else None
        )
        chunk = self._loaded_bus().request_sync(model_rpc.INFER, request, ActionChunk, timeout=_INFER_S)
        return np.asarray(chunk.actions, dtype=np.float32)

    def reset(self) -> None:
        self._loaded_bus().request_sync(model_rpc.RESET, model_rpc.Empty(), CommandDone, timeout=_INFER_S)

    def _loaded_bus(self) -> Bus:
        if self._bus is None:
            raise RuntimeError(f"Model {self.model.name!r} is not loaded")
        return self._bus

    async def _report_death(self, bus: Bus) -> None:
        await self._worker.wait_stopped()
        publish_status(bus, COMPONENT, "failed", self._worker.failure or "stopped unexpectedly")

    def _report_stopped(self, bus: Bus) -> None:
        # A worker that failed stays reported failed.
        if self._started and self._worker.failure is None:
            publish_status(bus, COMPONENT, "stopped")
