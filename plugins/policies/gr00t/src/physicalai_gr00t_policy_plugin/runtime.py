# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime adapter for GR00T packages, backed by an isolated worker process.

GR00T cannot be imported in the Runtime process because its pinned
dependencies conflict with Runtime's. The adapter converts Runtime observations
into GR00T modality arrays, sends them over pickle-free pipes to a worker in
the package's pinned environment, and flattens the returned action chunk.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Fixed worker argv; shell is disabled.
import threading
import weakref
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from physicalai.inference.adapters.base import RuntimeAdapter
from physicalai.inference.adapters.registry import adapter_registry
from physicalai.inference.callbacks.base import Callback

from ._worker import gr00t_worker_protocol as protocol
from .package import ARTIFACT_SUFFIX, BACKEND, FORMAT_OPENVINO, IoSpec, load_descriptor

if TYPE_CHECKING:
    from physicalai.inference.model import InferenceModel

    from .environments import WorkerEnvironmentSpec

logger = logging.getLogger(__name__)

_CHANNELS = {1, 3, 4}
_IMAGE_RANK = 3
_BATCHED_IMAGE_RANK = 4
_CLOSE_TIMEOUT_SECONDS = 30.0
_PROJECT_ENV = "PHYSICALAI_WORKER_PROJECT"
_CACHE_ENV = "PHYSICALAI_GR00T_CACHE"


@adapter_registry.register(BACKEND, extensions=(ARTIFACT_SUFFIX,))
class Gr00tRuntimeAdapter(RuntimeAdapter):
    """Run native or OpenVINO GR00T packages through a pinned worker process."""

    def __init__(
        self,
        device: str = "auto",
        *,
        interpreter: str | Path | None = None,
        project: str | Path | None = None,
        seed: int | None = None,
        static_shape: bool = False,
        cache_dir: str | Path | None = None,
        **kwargs: Any,  # ruff: ignore[any-type]  # Runtime forwards backend-specific options.
    ) -> None:
        """Configure the adapter.

        Args:
            device: ``auto``, a PyTorch device for native packages, or an
                OpenVINO device for OpenVINO packages.
            interpreter: User-managed worker interpreter instead of a managed environment.
            project: Isaac-GR00T checkout used by *interpreter*.
            seed: Draw flow-matching noise from a seeded CPU generator. Calls use
                ``seed + n`` for the n-th prediction since the last reset, so two
                adapters with the same seed see identical noise.
            static_shape: Compile OpenVINO models with static input shapes.
            cache_dir: Writable directory for OpenVINO's compiled-model cache.
                Defaults to a per-package directory in the GR00T cache; the
                package itself is never written.
            **kwargs: Additional Runtime options.
        """
        super().__init__(device=device, **kwargs)
        self._interpreter = interpreter
        self._project = project
        self._seed = seed
        self._static_shape = static_shape
        self._cache_dir = cache_dir
        self._calls = 0
        self._worker: _Worker | None = None
        self._io: IoSpec | None = None
        self.descriptor: dict[str, Any] | None = None
        self.model_info: dict[str, Any] | None = None

    def default_device(self) -> str:  # ruff: ignore[no-self-use]
        """Let the worker choose CUDA, CPU, or the OpenVINO CPU plugin.

        Returns:
            ``auto``.
        """
        return "auto"

    def load(self, model_path: Path | str) -> None:
        """Start the worker for the package's environment and load the model.

        The worker is stopped and ``ValueError`` raised if the model's modalities
        do not match the package's I/O mapping.
        """
        artifact = Path(model_path).expanduser().resolve(strict=True)
        descriptor = load_descriptor(artifact)
        if descriptor.get("backbone_override"):
            logger.warning(
                "GR00T package uses development backbone source %r instead of the upstream default.",
                descriptor.get("backbone_source"),
            )
        io = IoSpec.from_dict(descriptor["io"])
        python, project, env = self._environment(descriptor)
        worker = _Worker(python, project)
        try:
            _check_worker(worker.request({"op": "hello"}), env, project)
            info = worker.request(
                {
                    "op": "load",
                    "format": descriptor["format"],
                    "path": str(artifact / descriptor["model_dir"]),
                    "embodiment_tag": descriptor["embodiment_tag"],
                    "device": self.device,
                    "static_shape": self._static_shape,
                    "cache_dir": str(self._openvino_cache(artifact))
                    if descriptor["format"] == FORMAT_OPENVINO
                    else None,
                },
            )
            _check_modalities(io, info)
        except BaseException:
            worker.close()
            raise
        if self._worker is not None:
            self._worker.close()
        self._worker, self._io, self.descriptor, self.model_info = worker, io, descriptor, info
        self._calls = 0
        logger.info(
            "Loaded %s GR00T package on %s in %.1fs.",
            descriptor["format"],
            info.get("device"),
            info.get("load_seconds", 0.0),
        )

    def predict(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Predict one action chunk.

        Returns:
            ``{"action": array}`` with shape ``(1, horizon, action_dim)``.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._worker is None or self._io is None:
            msg = "GR00T model not loaded. Call load() first."
            raise RuntimeError(msg)
        request = {"op": "predict", **to_gr00t_observation(inputs, self._io)}
        if self._seed is not None:
            request["seed"] = self._seed + self._calls
        response = self._worker.request(request)
        self._calls += 1
        return {"action": flatten_actions(response["actions"], self._io)}

    def reset(self) -> None:
        """Reset worker-side policy state and the deterministic noise sequence."""
        self._calls = 0
        if self._worker is not None:
            self._worker.request({"op": "reset"})

    def close(self) -> None:
        """Stop the worker process."""
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    @property
    def input_names(self) -> list[str]:
        """Empty, so Runtime passes the structured observation unchanged."""
        return []

    @property
    def output_names(self) -> list[str]:
        """The flat action output."""
        return ["action"]

    @property
    def is_openvino(self) -> bool:
        """Whether the loaded package runs the OpenVINO single IR."""
        return bool(self.descriptor and self.descriptor["format"] == FORMAT_OPENVINO)

    def _environment(self, descriptor: Mapping[str, Any]) -> tuple[Path, Path, WorkerEnvironmentSpec | None]:
        """Resolve the worker environment recorded by the package.

        The package's stable environment identity must match the installed plugin's
        pins. Runtime prepares managed environments through ``physicalai.isolate``;
        user-managed interpreters bypass preparation and lock verification.

        Returns:
            Interpreter, checkout, and the managed environment (``None`` when unverified).

        Raises:
            ValueError: If a user-managed interpreter has no checkout or the package was built from different pins.
        """
        recorded = descriptor["environment"]
        from . import environments  # ruff: ignore[import-outside-top-level]  # Load plugin-owned isolate declarations lazily.

        env = environments.ENVIRONMENTS.get(recorded["name"])
        if env is None or recorded.get("key") != env.key():
            msg = (
                "This GR00T package was built in a different environment than the installed plugin provides; "
                "install the plugin version that produced it."
            )
            raise ValueError(msg)
        python, checkout, managed = environments.resolve(env, interpreter=self._interpreter, project=self._project)
        if not managed:
            logger.warning("Using a user-managed GR00T environment; the package's lock is not verified.")
        return python, checkout, env if managed else None

    def _openvino_cache(self, artifact: Path) -> Path:
        if self._cache_dir is not None:
            return Path(self._cache_dir).expanduser()
        key = hashlib.sha256(str(artifact).encode("utf-8")).hexdigest()[:16]
        configured = os.environ.get(_CACHE_ENV)
        cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
        root = Path(configured).expanduser() if configured else cache_home / "physicalai" / "gr00t"
        return root / "openvino-cache" / key


class Gr00tResetCallback(Callback):
    """Forward Runtime episode resets to the GR00T worker."""

    def __init__(self) -> None:
        """Create an unbound callback."""
        self._adapter: Gr00tRuntimeAdapter | None = None

    def on_load(self, model: InferenceModel) -> None:
        """Bind to the loaded GR00T adapter.

        Raises:
            TypeError: If the package was loaded by another adapter.
        """
        if not isinstance(model.adapter, Gr00tRuntimeAdapter):
            msg = "The GR00T reset callback requires the gr00t Runtime adapter."
            raise TypeError(msg)
        self._adapter = model.adapter

    def on_reset(self) -> None:
        """Reset the worker when Runtime starts a new episode."""
        if self._adapter is not None:
            self._adapter.reset()


def to_gr00t_observation(inputs: Mapping[str, Any], io: IoSpec) -> dict[str, Any]:
    """Convert a Runtime observation into GR00T modality arrays.

    Accepted image keys are ``images`` (a camera mapping), ``images.<camera>``,
    and ``observation.images.<camera>``. The state vector is ``state`` or
    ``observation.state``; the instruction is ``task``.

    Returns:
        ``video`` (``(1, H, W, 3)`` uint8 per key), ``state`` (``(1, D)`` float32
        per key), and ``task``.

    Raises:
        KeyError: If a mapped camera, the state, or the task is missing.
        ValueError: If the state vector is shorter than the mapping requires.
    """
    images = _images(inputs)
    video = {}
    for key, camera in io.cameras.items():
        if camera not in images:
            msg = f"Observation is missing camera {camera!r} for GR00T video key {key!r}."
            raise KeyError(msg)
        video[key] = _hwc_uint8(np.asarray(images[camera]))[np.newaxis]
    raw_state = inputs.get("state", inputs.get("observation.state"))
    if raw_state is None:
        msg = "Observation is missing 'state'."
        raise KeyError(msg)
    vector = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    if vector.size < io.state_dim:
        msg = f"State has {vector.size} values; the GR00T mapping needs {io.state_dim}."
        raise ValueError(msg)
    state = {key: vector[start:end][np.newaxis].copy() for key, (start, end) in io.state.items()}
    return {"video": video, "state": state, "task": _task(inputs)}


def flatten_actions(actions: Mapping[str, np.ndarray], io: IoSpec) -> np.ndarray:
    """Place per-key GR00T actions into one flat Runtime action chunk.

    Returns:
        Float32 array with shape ``(1, horizon, action_dim)``.

    Raises:
        KeyError: If an action key is missing.
        ValueError: If an action does not match its mapped width.
    """
    horizon = None
    chunk = None
    for key, (start, end) in io.action.items():
        if key not in actions:
            msg = f"GR00T worker did not return action {key!r}."
            raise KeyError(msg)
        value = np.asarray(actions[key], dtype=np.float32)
        if horizon is None:
            horizon = value.shape[0]
            chunk = np.zeros((horizon, io.action_dim), dtype=np.float32)
        if value.shape != (horizon, end - start):
            msg = f"Action {key!r} has shape {value.shape}, expected {(horizon, end - start)}."
            raise ValueError(msg)
        chunk[:, start:end] = value  # type: ignore[index]
    if chunk is None:
        msg = "The GR00T action mapping is empty."
        raise ValueError(msg)
    return chunk[np.newaxis]


def _images(inputs: Mapping[str, Any]) -> dict[str, Any]:
    images: dict[str, Any] = {}
    nested = inputs.get("images")
    if isinstance(nested, Mapping):
        images.update({str(name): value for name, value in nested.items()})
    for key, value in inputs.items():
        for prefix in ("observation.images.", "images."):
            if key.startswith(prefix):
                images[key.removeprefix(prefix)] = value
    return images


def _hwc_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == _BATCHED_IMAGE_RANK and image.shape[0] == 1:
        image = image[0]
    if image.ndim != _IMAGE_RANK:
        msg = f"Expected an image with three dimensions, got shape {image.shape}."
        raise ValueError(msg)
    if image.shape[0] in _CHANNELS and image.shape[-1] not in _CHANNELS:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(np.rint(image.astype(np.float32) * scale), 0, 255).astype(np.uint8)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return np.ascontiguousarray(image[..., :3])


def _task(inputs: Mapping[str, Any]) -> str:
    task = inputs.get("task")
    if isinstance(task, np.ndarray):
        task = task.tolist()
    if isinstance(task, Sequence) and not isinstance(task, str) and len(task) == 1:
        task = task[0]
    if not isinstance(task, str):
        msg = "Observation needs a 'task' instruction string for GR00T."
        raise KeyError(msg)
    return task


def _check_worker(hello: Mapping[str, Any], env: WorkerEnvironmentSpec | None, project: Path) -> None:
    """Check that the worker runs GR00T, at the locked version for managed environments.

    Raises:
        RuntimeError: If the worker reports another framework or version.
    """
    if hello.get("framework") != BACKEND or hello.get("package") != "gr00t":
        msg = f"Worker reported framework {hello.get('framework')!r} and package {hello.get('package')!r}."
        raise RuntimeError(msg)
    if env is None:
        return
    from .environments import locked_version  # ruff: ignore[import-outside-top-level]  # Read the isolate project's lock.

    expected = locked_version(env, project)
    if hello.get("version") != expected:
        msg = f"Worker reported gr00t {hello.get('version')}, but the environment lock pins {expected}."
        raise RuntimeError(msg)


def _check_modalities(io: IoSpec, info: Mapping[str, Any]) -> None:
    for label, expected, actual in (
        ("video", set(io.cameras), set(info.get("video_keys", []))),
        ("state", set(io.state), set(info.get("state_keys", []))),
        ("action", set(io.action), set(info.get("action_keys", []))),
    ):
        if expected != actual:
            msg = f"Package {label} keys {sorted(expected)} do not match the model's {sorted(actual)}."
            raise ValueError(msg)
    for horizon in ("video_horizon", "state_horizon"):
        if info.get(horizon) != 1:
            msg = f"GR00T models with {horizon} {info.get(horizon)} need observation history, which is not supported."
            raise ValueError(msg)


class _Worker:
    """One inference worker process and its request/response pipes."""

    def __init__(self, python: Path, project: Path) -> None:
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()
        env = {
            **os.environ,
            _PROJECT_ENV: str(project),
            "PHYSICALAI_WORKER_FRAMEWORK": BACKEND,
            "PHYSICALAI_GR00T_REQUEST_FD": str(request_read),
            "PHYSICALAI_GR00T_RESPONSE_FD": str(response_write),
        }
        try:
            self.process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]  # argv list; shell disabled.
                [str(python), str(_infer_worker())],
                stdin=subprocess.DEVNULL,
                env=env,
                pass_fds=(request_read, response_write),
                shell=False,
            )
        finally:
            os.close(request_read)
            os.close(response_write)
        self._requests = os.fdopen(request_write, "wb")
        self._responses = os.fdopen(response_read, "rb")
        self._lock = threading.Lock()
        self._finalizer = weakref.finalize(self, _stop, self.process, self._requests, self._responses)

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            protocol.write_frame(self._requests, message)
            response = protocol.read_frame(self._responses)
        if response is None:
            msg = f"GR00T worker exited with status {self.process.poll()}."
            raise RuntimeError(msg)
        if not response.get("ok"):
            msg = f"GR00T worker raised {response.get('error')}: {response.get('message')}"
            raise RuntimeError(msg)
        return response

    def close(self) -> None:
        self._finalizer()


def _infer_worker() -> Path:
    with resources.as_file(resources.files(__package__) / "_worker" / "gr00t_infer_worker.py") as path:
        return Path(path)


def _stop(process: subprocess.Popen[bytes], requests: Any, responses: Any) -> None:  # ruff: ignore[any-type]
    try:
        if process.poll() is None:
            protocol.write_frame(requests, {"op": "close"})
    except (BrokenPipeError, OSError, ValueError):
        pass
    for stream in (requests, responses):
        with contextlib.suppress(OSError):
            stream.close()
    try:
        process.wait(timeout=_CLOSE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
