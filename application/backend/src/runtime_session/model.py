"""A model loaded into a session: inference in its own process, called like a local model.

A :class:`ProcessModel` starts a ``ModelWorker`` that loads one of the model's
exports, and reports as ``model`` on the session's bus. To ``ModelControl``'s
executor it is the model itself: ``predict_action_chunk`` sends the state and
the task to the worker, which adds the frames from the store, and returns the
whole chunk. Queueing it, smoothing and deciding when to ask again stays with
the executor, as ``physicalai.runtime`` runs it. ``RTCExecution`` calls the
model itself instead, with its RTC inputs added; those cross to the worker too.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from runtime_session.bus import Bus, publish_status, until_serving
from schemas.runtime_session import CommandDone, SessionModel
from workers import model_worker as model_rpc
from workers.model_worker import RTC_INPUTS, ActionChunk, InferRequest, ModelWorker, RtcInputs

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
    """What an export takes, from its manifest: the state's size, the cameras it reads and the tasks it knows."""

    state_size: int
    images: dict[str, str]
    """Model input name to the store camera it reads."""
    tasks: tuple[str, ...] = ()
    """The only tasks a policy trained on a fixed set takes, such as a multi-task ACT; empty if any text goes."""
    rtc_chunk: tuple[int, int] | None = None
    """``prev_chunk_left_over``'s shape, ``(chunk_size, action_dim)``, when the export takes RTC inputs."""
    rtc_dtypes: dict[str, str] = field(default_factory=dict)
    """Each RTC input's dtype, as the export declares it, when it takes RTC inputs."""


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
    rtc_chunk = _rtc_chunk(features, len(spec.joint_names))
    rtc_dtypes = _rtc_dtypes(features) if rtc_chunk is not None else {}
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
    tasks = next(
        (spec["tasks"] for spec in manifest["model"].get("preprocessors", []) if spec["type"] == "task_index"), ()
    )
    return ModelInputs(state_size, images, tuple(tasks), rtc_chunk, rtc_dtypes)


_REEXPORT = "re-export the policy with RTC enabled, or export it without RTC inputs to run it synchronously"


def _shape(feature: dict[str, Any]) -> list[int] | None:
    """A feature's shape as a list of whole, non-negative sizes, or ``None`` if it is anything else."""
    shape = feature.get("shape")
    if isinstance(shape, list) and all(type(size) is int and size >= 0 for size in shape):
        return shape
    return None


def _rtc_chunk(features: list[dict[str, Any]], joints: int) -> tuple[int, int] | None:
    """``(chunk_size, action_dim)`` of an export that takes all four RTC inputs, or ``None`` if it takes none.

    Raises:
        ValueError: The export lists only some RTC inputs, or shapes RTC cannot fill for this follower.
    """
    shapes = {feature["name"]: _shape(feature) for feature in features if feature["name"] in RTC_INPUTS}
    if not shapes:
        return None
    if missing := [name for name in RTC_INPUTS if name not in shapes]:
        raise ValueError(f"The export takes some RTC inputs but is missing {', '.join(missing)}: {_REEXPORT}")
    tail = shapes[model_rpc.PREV_CHUNK_LEFT_OVER]
    if tail is None or len(tail) != 2:  # whole sizes (chunk_size, action_dim)
        raise ValueError(
            f"The export's {model_rpc.PREV_CHUNK_LEFT_OVER} has shape {tail}, not (chunk_size, action_dim): {_REEXPORT}"
        )
    chunk_size, action_dim = tail
    if action_dim != joints:
        raise ValueError(
            f"The export's RTC tail has {action_dim} actions per step, but the follower has {joints} joints: "
            f"{_REEXPORT} for this robot"
        )
    if chunk_size < 2:
        raise ValueError(f"The export's RTC tail has a chunk of {chunk_size}, too short to guide: {_REEXPORT}")
    if scalars := [name for name in RTC_INPUTS[1:] if shapes[name] != []]:
        raise ValueError(
            f"The export takes {', '.join(scalars)} with shape {shapes[scalars[0]]}, not a scalar: {_REEXPORT}"
        )
    return chunk_size, action_dim


def _rtc_dtypes(features: list[dict[str, Any]]) -> dict[str, str]:
    """Each RTC input's dtype as the export declares it, or the usual one where it does not say.

    Raises:
        ValueError: A declared dtype is not one RTC can fill: whole numbers for the delay and horizon, floats else.
    """
    dtypes = dict(model_rpc.RTC_DTYPES)
    for feature in features:
        name = feature["name"]
        if name not in dtypes or "dtype" not in feature:
            continue
        declared = feature["dtype"]
        kind = "iu" if name in {model_rpc.INFERENCE_DELAY, model_rpc.EXECUTION_HORIZON} else "f"
        try:
            valid = isinstance(declared, str) and np.dtype(declared).kind in kind
        except TypeError:
            valid = False
        if not valid:
            raise ValueError(f"The export takes {name} as {declared!r}, which RTC cannot fill: {_REEXPORT}")
        dtypes[name] = declared
    return dtypes


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
            rtc_chunk=self.inputs.rtc_chunk,
            rtc_dtypes=self.inputs.rtc_dtypes or None,
        )
        self._session_id = session_id
        self._bus: Bus | None = None
        self._stack: AsyncExitStack | None = None
        self._started = False
        self.last_error: Exception | None = None
        """Why the last inference failed, or ``None`` once one succeeds. ``RTCExecution`` reports a failed warmup
        or too many failed refills in its own words; ``ModelControl`` reports this instead, as Sync and Async do."""

    def state(self) -> SessionModel:
        return SessionModel(
            model_id=self.model.id,
            name=self.model.name,
            backend=self.backend,
            device=self.device,
            tasks=list(self.inputs.tasks),
            rtc=self.inputs.rtc_chunk is not None,
            rtc_max_horizon=self.inputs.rtc_chunk[0] // 2 if self.inputs.rtc_chunk is not None else None,
        )

    @property
    def chunk_size(self) -> int:
        """The export's chunk size when it says (an RTC export does), else 1 as ``InferenceModel`` reports."""
        return self.inputs.rtc_chunk[0] if self.inputs.rtc_chunk is not None else 1

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
        rtc = None
        if model_rpc.PREV_CHUNK_LEFT_OVER in observation:
            rtc = RtcInputs(
                prev_chunk_left_over=np.asarray(observation[model_rpc.PREV_CHUNK_LEFT_OVER])[0].tolist(),
                inference_delay=int(observation[model_rpc.INFERENCE_DELAY]),
                max_guidance_weight=float(observation[model_rpc.MAX_GUIDANCE_WEIGHT]),
                execution_horizon=int(observation[model_rpc.EXECUTION_HORIZON]),
            )
        request = InferRequest(
            state=np.asarray(observation[model_rpc.STATE])[0].tolist(), task=task[0] if task else None, rtc=rtc
        )
        try:
            chunk = self._loaded_bus().request_sync(model_rpc.INFER, request, ActionChunk, timeout=_INFER_S)
        except Exception as error:
            self.last_error = error
            raise
        self.last_error = None
        return np.asarray(chunk.actions, dtype=np.float32)

    def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """What ``RTCExecution`` calls: the chunk for ``inputs``, RTC inputs included, as ``{"action": chunk}``.

        Raises:
            BusRequestError: The model failed on it.
            NoResponderError: The model is no longer loaded.
        """
        return {"action": self.predict_action_chunk(inputs)}

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
