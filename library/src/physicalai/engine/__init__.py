# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Framework-level training API.

Framework implementations are discovered from the
``physicalai.engine.frameworks`` entry-point group when an :class:`Engine` is
created. Importing this module does not import LeRobot, Lightning, or PyTorch.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._registry import DuplicateFrameworkError, FrameworkRegistry
from ._types import Framework, Run, TrainHooks, TrainingCancelledError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from physicalai.inference.model import InferenceModel


class _NullTrainHooks:
    """No-op progress hook used when the caller does not supply one."""

    @staticmethod
    def on_step(step: int, total: int | None, metrics: Mapping[str, float]) -> None:
        """Ignore a progress update."""

    @staticmethod
    def should_stop() -> bool:
        """Never request cancellation.

        Returns:
            Always ``False``.
        """
        return False


def _run_output_directory(path: Path | None) -> Path:
    """Find a sensible package root from a native run or checkpoint path.

    Args:
        path: Run output directory or native model checkpoint path.

    Returns:
        Directory under which ``exports/<framework>`` should be written.

    Raises:
        ValueError: If no run output or checkpoint path is available.
    """
    if path is None:
        msg = "Cannot determine an export directory without a run output or checkpoint path."
        raise ValueError(msg)
    candidate = Path(path).expanduser()
    if candidate.is_file():
        candidate = candidate.parent
    if candidate.name == "pretrained_model":
        if candidate.parent.name == "last" and candidate.parent.parent.name == "checkpoints":
            return candidate.parent.parent.parent
        return candidate.parent
    if (candidate / "checkpoints" / "last" / "pretrained_model").is_dir() or (candidate / "pretrained_model").is_dir():
        return candidate
    return candidate.parent


class Engine:
    """Route native training requests to installed frameworks."""

    def __init__(self, *, registry: FrameworkRegistry | None = None) -> None:
        """Create an engine and discover installed frameworks when needed.

        Args:
            registry: Optional registry, useful for embedding the engine with a
                controlled framework set. Defaults to installed entry points.
        """
        self._registry = registry if registry is not None else FrameworkRegistry.discover()

    def frameworks(self) -> list[str]:
        """Return the identifiers of discovered frameworks.

        Returns:
            Framework names in registry discovery order.
        """
        return self._registry.names()

    def export(
        self,
        source: Run | str | Path,
        output_dir: str | Path | None = None,
        *,
        framework: str | None = None,
        format: str | None = None,  # noqa: A002
    ) -> Path:
        """Package a native training checkpoint for Runtime inference.

        Args:
            source: A training :class:`Run` or a framework-native checkpoint.
            output_dir: Optional destination. Defaults to ``<run>/exports/<framework>``.
            framework: Framework name for a path source. A :class:`Run` carries
                its framework name and does not require this argument.
            format: Optional framework-specific export format.

        Returns:
            The completed Runtime package directory.

        Raises:
            ValueError: If the framework is ambiguous or conflicts with the Run.
        """
        if isinstance(source, Run):
            if framework is not None and framework != source.framework:
                msg = f"Run belongs to framework {source.framework!r}, not {framework!r}."
                raise ValueError(msg)
            framework_name = source.framework
            default_base = source.output_dir or _run_output_directory(source.checkpoint)
        else:
            names = self._registry.names()
            if framework is None:
                if len(names) != 1:
                    msg = "Specify framework= when exporting a native checkpoint path."
                    raise ValueError(msg)
                framework_name = names[0]
            else:
                framework_name = framework
            source_path = Path(source).expanduser()
            default_base = _run_output_directory(source_path)

        selected = self._registry.get(framework_name)
        destination = Path(output_dir) if output_dir is not None else default_base / "exports" / framework_name
        return selected.export(source, destination, format=format)

    @staticmethod
    def load(package: str | Path, **kwargs: Any) -> InferenceModel:  # noqa: ANN401
        """Load a Runtime policy package through the installed Runtime API.

        Args:
            package: Directory containing a Runtime-compatible ``manifest.json``.
            **kwargs: Arguments forwarded to :class:`InferenceModel`.

        Returns:
            Initialized Runtime inference model.
        """
        from physicalai.inference.model import InferenceModel as RuntimeInferenceModel  # noqa: PLC0415

        return RuntimeInferenceModel(package, **kwargs)

    def train(
        self,
        config: object | Sequence[str],
        *,
        framework: str | None = None,
        hooks: TrainHooks | None = None,
    ) -> Run:
        """Train using a framework's native config object or CLI arguments.

        Args:
            config: Native config object, or an ordered sequence of native CLI
                argument tokens.
            framework: Optional framework name. Required for CLI argument
                sequences and when more than one framework accepts a config.
            hooks: Optional progress and cancellation hooks.

        Returns:
            Framework-produced run metadata.

        Raises:
            TypeError: If CLI arguments contain non-string values.
            ValueError: If the framework is unknown, no framework accepts the
                config, or config dispatch is ambiguous.
        """
        if isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
            if not all(isinstance(argument, str) for argument in config):
                msg = "Framework CLI arguments must be strings."
                raise TypeError(msg)
            if framework is None:
                msg = "A framework name is required when training from CLI arguments."
                raise ValueError(msg)
            native_config: object | Sequence[str] = list(config)
            selected = self._registry.get(framework)
        elif framework is not None:
            selected = self._registry.get(framework)
            native_config = config
        else:
            selected = self._registry.for_config(config)
            native_config = config

        return selected.train(native_config, hooks if hooks is not None else _NullTrainHooks())


__all__ = [
    "DuplicateFrameworkError",
    "Engine",
    "Framework",
    "FrameworkRegistry",
    "Run",
    "TrainHooks",
    "TrainingCancelledError",
]
