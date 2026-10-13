# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot v0.6 native training integration."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

    from lerobot.configs.train import TrainPipelineConfig
    from physicalai.engine import FrameworkRegistry, Run, TrainHooks

_NATIVE_TRAIN_LOCK = RLock()
_NON_METRIC_FIELDS = frozenset({"steps", "samples", "episodes", "epochs"})
_CONFIG_CLASS_ATTRIBUTE = "TrainPipelineConfig"
_VALIDATE_ATTRIBUTE = "validate"
_UPDATE_POLICY_ATTRIBUTE = "update_policy"


@contextmanager
def _native_argv(arguments: Sequence[str]) -> Iterator[None]:
    """Temporarily give LeRobot only its own command-line arguments.

    Yields:
        Control while ``sys.argv`` contains LeRobot's program name and flags.
    """
    with _NATIVE_TRAIN_LOCK:
        original_argv = sys.argv
        sys.argv = ["lerobot-train", *arguments]
        try:
            yield
        finally:
            sys.argv = original_argv


@contextmanager
def _capture_native_config(lerobot_train: ModuleType) -> Iterator[list[object]]:
    """Capture the validated native config during LeRobot CLI execution.

    Args:
        lerobot_train: LeRobot's native training script module.

    Yields:
        List populated after LeRobot validates the parsed config.

    Raises:
        TypeError: If the native train module has no TrainPipelineConfig validator.
    """
    config_class = getattr(lerobot_train, _CONFIG_CLASS_ATTRIBUTE, None)
    validate = getattr(config_class, _VALIDATE_ATTRIBUTE, None)
    if not isinstance(config_class, type) or not callable(validate):
        msg = "LeRobot train module does not expose TrainPipelineConfig.validate()."
        raise TypeError(msg)

    original_validate = validate
    configs: list[object] = []

    def capture(config: object) -> None:
        original_validate(config)
        configs.append(config)

    setattr(config_class, _VALIDATE_ATTRIBUTE, capture)
    try:
        yield configs
    finally:
        setattr(config_class, _VALIDATE_ATTRIBUTE, original_validate)


@contextmanager
def _report_steps(
    lerobot_train: ModuleType,
    hooks: TrainHooks,
    total_steps: int | None,
) -> Iterator[None]:
    """Report LeRobot update steps and translate hook cancellation to an exception.

    Args:
        lerobot_train: LeRobot's native training script module.
        hooks: Engine progress and cancellation hooks.
        total_steps: Expected training step count when the native config is known.

    Yields:
        Control while LeRobot's update function reports progress.
    """
    original_update_policy = cast("Callable[..., object]", getattr(lerobot_train, _UPDATE_POLICY_ATTRIBUTE))

    def report_update(*args: object, **kwargs: object) -> object:
        tracker = args[0] if args else None
        completed_steps = getattr(tracker, "steps", None)
        if hooks.should_stop():
            _raise_training_cancelled(completed_steps if isinstance(completed_steps, int) else None)

        result = original_update_policy(*args, **kwargs)
        if not isinstance(result, tuple) or not result:
            msg = "LeRobot update_policy must return a non-empty tuple."
            raise TypeError(msg)
        updated_tracker = result[0]
        current_steps = getattr(updated_tracker, "steps", None)
        if not isinstance(current_steps, int):
            msg = "LeRobot update_policy returned a tracker without an integer 'steps' counter."
            raise TypeError(msg)
        step = current_steps + 1
        hooks.on_step(step, total_steps, _read_metrics(updated_tracker))
        if hooks.should_stop():
            _raise_training_cancelled(step)
        return result

    setattr(lerobot_train, _UPDATE_POLICY_ATTRIBUTE, report_update)
    try:
        yield
    finally:
        setattr(lerobot_train, _UPDATE_POLICY_ATTRIBUTE, original_update_policy)


def _raise_training_cancelled(step: int | None) -> None:
    from physicalai.engine import TrainingCancelledError  # ruff: ignore[import-outside-top-level]

    raise TrainingCancelledError(step)


def _read_metrics(tracker: object) -> Mapping[str, float]:
    """Extract scalar metrics from LeRobot's native tracker.

    Returns:
        Per-step metric values, excluding progress counters.

    Raises:
        TypeError: If LeRobot's tracker does not return a metric dictionary.
    """
    to_dict = getattr(tracker, "to_dict", None)
    if not callable(to_dict):
        msg = "LeRobot update tracker does not expose to_dict()."
        raise TypeError(msg)
    values = to_dict(use_avg=False)
    if not isinstance(values, dict):
        msg = "LeRobot tracker to_dict() must return a dictionary."
        raise TypeError(msg)
    return {
        name: float(value)
        for name, value in values.items()
        if name not in _NON_METRIC_FIELDS and isinstance(value, (int, float)) and not isinstance(value, bool)
    }


class LeRobotFramework:
    """Delegate training to LeRobot's v0.6 native entry point."""

    name = "lerobot"

    def accepts(self, config: object) -> bool:
        """Recognize LeRobot's native TrainPipelineConfig without importing it.

        Args:
            config: Candidate native training configuration.

        Returns:
            Whether *config* is LeRobot's TrainPipelineConfig.
        """
        config_type = type(config)
        return config_type.__name__ == "TrainPipelineConfig" and config_type.__module__.startswith(f"{self.name}.")

    def train(self, config: object | Sequence[str], hooks: TrainHooks) -> Run:
        """Run LeRobot's native trainer with scoped arguments and registration.

        Args:
            config: Native LeRobot config object or LeRobot CLI argument tokens.
            hooks: Engine progress and cancellation hooks.

        Returns:
            Native LeRobot run and checkpoint paths when available.

        Raises:
            TypeError: If *config* is neither a LeRobot config object nor a
                sequence of CLI argument strings, or the native config is invalid.
        """
        from lerobot.scripts import lerobot_train  # ruff: ignore[import-outside-top-level]

        if isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
            arguments = list(config)
            if not all(isinstance(argument, str) for argument in arguments):
                msg = "LeRobot CLI arguments must be strings."
                raise TypeError(msg)
            with (
                _native_argv(arguments),
                _capture_native_config(lerobot_train) as native_configs,
                _report_steps(lerobot_train, hooks, total_steps=None),
            ):
                lerobot_train.main()
            if native_configs:
                return self._run_from_config(native_configs[-1])
            return self._run_from_config(None)

        if not self.accepts(config):
            msg = f"LeRobot framework requires a native TrainPipelineConfig, got {type(config).__name__}."
            raise TypeError(msg)

        from lerobot.utils.import_utils import register_third_party_plugins  # ruff: ignore[import-outside-top-level]

        total_steps = self._configured_steps(config)
        with _native_argv([]):
            register_third_party_plugins()
            with _report_steps(lerobot_train, hooks, total_steps=total_steps):
                lerobot_train.train(cast("TrainPipelineConfig", config))
        return self._run_from_config(config)

    def export(
        self,
        source: Run | str | Path,
        output_dir: Path,
        format: str | None = None,  # ruff: ignore[builtin-argument-shadowing]
    ) -> Path:
        """Package a native checkpoint for Runtime inference.

        Args:
            source: LeRobot run or local native checkpoint directory.
            output_dir: New directory where the Runtime package is written.
            format: ``None`` or ``"native"`` to preserve LeRobot's artifact format.

        Returns:
            The completed Runtime package directory.

        Raises:
            ValueError: If an unsupported export format is requested.
        """
        if format not in {None, "native"}:
            msg = f"{self.name} supports only the native export format, got {format!r}."
            raise ValueError(msg)
        from .package import package_lerobot_checkpoint  # ruff: ignore[import-outside-top-level]

        return package_lerobot_checkpoint(source, output_dir)

    @staticmethod
    def _configured_steps(config: object) -> int | None:
        configured_steps = getattr(config, "steps", None)
        if isinstance(configured_steps, int) and not isinstance(configured_steps, bool):
            return configured_steps
        return None

    @staticmethod
    def _run_from_config(config: object | None) -> Run:
        """Extract native output and final checkpoint paths after training.

        Args:
            config: LeRobot config after its native training function returns,
                or ``None`` if a remote run did not resolve local output paths.

        Returns:
            Run metadata populated with available output and checkpoint paths.
        """
        from physicalai.engine import Run  # ruff: ignore[import-outside-top-level]

        if config is None:
            return Run(framework="lerobot", output_dir=None, checkpoint=None)
        configured_output = getattr(config, "output_dir", None)
        output_dir = Path(configured_output) if isinstance(configured_output, (str, Path)) else None
        checkpoint = None
        if output_dir is not None:
            candidate = output_dir / "checkpoints" / "last" / "pretrained_model"
            if candidate.is_dir():
                checkpoint = candidate
        return Run(framework="lerobot", output_dir=output_dir, checkpoint=checkpoint)


def register(registry: FrameworkRegistry) -> None:
    """Register the built-in LeRobot framework structurally.

    Args:
        registry: Framework registry to populate.
    """
    registry.register(LeRobotFramework())
