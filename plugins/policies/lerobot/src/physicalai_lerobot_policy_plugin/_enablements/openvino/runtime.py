# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch-free OpenVINO runtime plugin for ``lerobot.bundle``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_DEVICE = "CPU"
_CACHE_DIR: str | None = None
_CORE: Any = None


def _get_core() -> Any:
    global _CORE  # noqa: PLW0603
    if _CORE is None:
        import openvino as ov

        _CORE = ov.Core()
    if _CACHE_DIR:
        _CORE.set_property({"CACHE_DIR": _CACHE_DIR})
    return _CORE


def configure(options: Mapping[str, Mapping[str, bool | int | str]]) -> None:
    """Apply OpenVINO runtime options (``device``, ``cache_dir``) before loading programs."""
    global _DEVICE, _CACHE_DIR  # noqa: PLW0603
    ov_options = options.get("openvino")
    if not ov_options:
        return
    device = ov_options.get("device")
    if isinstance(device, str) and device.strip():
        _DEVICE = device.strip()
    cache_dir = ov_options.get("cache_dir")
    if isinstance(cache_dir, str) and cache_dir.strip():
        _CACHE_DIR = cache_dir.strip()


class OpenVINOProgram:
    """One OpenVINO IR program (``.xml`` + ``.bin``) implementing ``lerobot.bundle.runtime.Program``."""

    def __init__(self, path: Path, data: Path | None = None) -> None:
        del data
        core = _get_core()
        self._compiled = core.compile_model(str(path), _DEVICE)
        self._input_dtypes = [port.get_element_type().to_dtype() for port in self._compiled.inputs]
        self._outputs = list(self._compiled.outputs)

    def __call__(self, *inputs: Any) -> tuple[np.ndarray, ...]:
        prepared = [self._coerce(index, value) for index, value in enumerate(inputs)]
        results = self._compiled(prepared)
        return tuple(np.array(results[output], copy=True) for output in self._outputs)

    def place(self, index: int, value: np.ndarray) -> np.ndarray:
        """Prepare *value* as a C-contiguous NumPy array matching input *index*'s dtype."""
        return self._coerce(index, value)

    def to_numpy(self, value: Any) -> np.ndarray:
        """Return *value* as a host NumPy array."""
        return np.asarray(value)

    def _coerce(self, index: int, value: Any) -> np.ndarray:
        array = np.asarray(value)
        expected = self._input_dtypes[index]
        if array.dtype != expected:
            array = array.astype(expected, copy=False)
        return np.require(array, requirements="C")


def _register() -> None:
    from lerobot.bundle.runtime import RUNTIMES, Runtime, register_runtime

    if "openvino" not in RUNTIMES:
        register_runtime("openvino", Runtime(load=OpenVINOProgram, configure=configure))


_register()
