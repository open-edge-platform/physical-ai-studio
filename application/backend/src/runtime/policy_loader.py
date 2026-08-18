from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger
from physicalai.inference.constants import IMAGES

from exceptions import BaseException as AppBaseException
from exceptions import ModelCameraMismatchError
from runtime.config_builder import policy_source_fragment, policy_source_from_fragment
from runtime.contract import ErrorEvent, LoadModelCommand
from settings import get_settings

if TYPE_CHECKING:
    from pathlib import Path

    from physicalai.inference import InferenceModel
    from physicalai.runtime import PolicySource

    from runtime.contract import EventSink


type ObservationSnapshot = tuple[Any, Mapping[str, Any]] | None
type ObservationProvider = Callable[[], ObservationSnapshot]
type PolicyReady = Callable[["PolicySource", int], None]


def check_camera_keys(model: InferenceModel, camera_keys: Sequence[str]) -> None:
    """Reject a model whose named image inputs are not in this session's cameras.

    Single-camera models emit a bare ``images`` key and discard the camera name,
    so they are not checked. Adapters that declare no image inputs are skipped.
    """
    expected_images = {name for name in model.adapter.input_names if name == IMAGES or name.startswith(f"{IMAGES}.")}
    if not expected_images or expected_images == {IMAGES}:
        return
    provided = {f"{IMAGES}.{key}" for key in camera_keys}
    missing = expected_images - provided
    if missing:
        raise ModelCameraMismatchError(expected=sorted(expected_images), provided=sorted(provided))


class PolicyLoader:
    """Build a PolicySource off the control thread and hand it over when ready."""

    def __init__(
        self,
        *,
        event_sink: EventSink,
        on_ready: PolicyReady,
        camera_keys: Sequence[str] = (),
        models_dir: Path | None = None,
    ) -> None:
        self._event_sink = event_sink
        self._on_ready = on_ready
        self._camera_keys = tuple(camera_keys)
        self._models_dir = models_dir
        self._lock = threading.Lock()
        self._generation = 0

    def request(
        self,
        command: LoadModelCommand,
        observation_provider: ObservationProvider,
        *,
        bus: object,
        session_id: str,
    ) -> None:
        """Start a daemon load. A newer request invalidates an in-flight one."""
        with self._lock:
            self._generation += 1
            generation = self._generation
        thread = threading.Thread(
            target=self._load,
            name=f"policy-loader-{generation}",
            args=(generation, command, observation_provider, bus, session_id),
            daemon=True,
        )
        thread.start()

    def shutdown(self) -> None:
        """Invalidate in-flight loads so they cannot attach after teardown."""
        with self._lock:
            self._generation += 1

    def _load(
        self,
        generation: int,
        command: LoadModelCommand,
        observation_provider: ObservationProvider,
        bus: object,
        session_id: str,
    ) -> None:
        source: PolicySource | None = None
        try:
            source = self._build_source(command)
            source.connect(bus=bus, session_id=session_id)  # type: ignore[arg-type]
            snapshot = observation_provider()
            if snapshot is not None:
                robot_state, camera_frames = snapshot
                source.warmup(source.to_model_input(robot_state, camera_frames))
            self._handover(generation, source)
        except Exception as exc:
            if source is not None:
                try:
                    source.disconnect()
                except Exception:
                    logger.exception("Failed to disconnect a policy that did not load")
            if self._is_stale(generation):
                return
            self._emit_error(exc)

    def _handover(self, generation: int, source: PolicySource) -> None:
        with self._lock:
            if generation != self._generation:
                source.disconnect()
                return
        self._on_ready(source, generation)

    def _is_stale(self, generation: int) -> bool:
        with self._lock:
            return generation != self._generation

    def _build_source(self, command: LoadModelCommand) -> PolicySource:
        from physicalai.runtime import PolicySource

        export_dir = self._export_dir(command)
        if not export_dir.exists():
            raise FileNotFoundError(export_dir)
        source = policy_source_from_fragment(
            policy_source_fragment(
                export_dir=str(export_dir),
                backend=command.inference_device.backend.value,
                device=command.inference_device.device,
            )
        )
        if not isinstance(source, PolicySource):
            raise TypeError(f"Expected PolicySource, got {type(source).__name__}")
        check_camera_keys(source._model, self._camera_keys)
        return source

    def _export_dir(self, command: LoadModelCommand) -> Path:
        models_dir = self._models_dir if self._models_dir is not None else get_settings().models_dir
        return models_dir / str(command.model_id) / "exports" / command.inference_device.backend.value

    def _emit_error(self, exc: Exception) -> None:
        if isinstance(exc, FileNotFoundError):
            path = exc.filename or (str(exc.args[0]) if exc.args else str(exc))
            self._event_sink.emit(
                ErrorEvent(
                    message=f"Model export not found at {path}.",
                    error_code="model_not_found",
                )
            )
            return
        if isinstance(exc, AppBaseException):
            self._event_sink.emit(ErrorEvent(message=exc.message, error_code=exc.error_code))
            return
        logger.exception("Policy load failed")
        self._event_sink.emit(
            ErrorEvent(
                message=str(exc) or "Failed to load the model.",
                error_code="model_load_failed",
            )
        )
