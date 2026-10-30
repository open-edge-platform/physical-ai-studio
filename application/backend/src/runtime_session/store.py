"""The session's latest state, action and camera frames, in shared memory.

One block per session holds a buffer and a timestamp per feature. Every
process in the session attaches to it by name and reads or writes without
locking: a read that overlaps a write can mix two consecutive samples, which
is acceptable because they are one tick apart. Writers store the values first
and the timestamp last, and readers read the timestamp first, so a reader
never mistakes an unwritten feature (NaN timestamp) for zeros.

See the "Store design" section of docs/feature_plans/runtime-inhousing-plan.md.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike

STATE = "state"
ACTION = "action"
VECTOR_KEYS = (STATE, ACTION)
"""Joint vectors the store always holds; they share the spec's joint names."""

BLOCK_PREFIX = "pai-"
_SHM_DIR = Path("/dev/shm")  # noqa: S108 - where Linux exposes POSIX shared memory
_WORD = 8


@dataclass(frozen=True, slots=True)
class ImageSpec:
    height: int
    width: int
    channels: int = 3

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    def __str__(self) -> str:
        return f"{self.height}x{self.width}x{self.channels}"


@dataclass(frozen=True, slots=True)
class Mismatch:
    feature: str
    expected: str
    actual: str

    def __str__(self) -> str:
        return f"{self.feature}: expected {self.expected}, got {self.actual}"


class FeatureSpecMismatchError(ValueError):
    """A component needs features the loaded store does not have."""

    def __init__(self, mismatches: list[Mismatch]) -> None:
        super().__init__("Features do not match the loaded session: " + "; ".join(map(str, mismatches)))
        self.mismatches = mismatches


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """The shape of a session: one robot's joints and the cameras observing it."""

    joint_names: tuple[str, ...]
    observations: Mapping[str, ImageSpec] = field(default_factory=dict)

    def check_compatible(self, other: FeatureSpec) -> list[Mismatch]:
        """Everything ``other`` needs that this spec does not provide.

        Joint names must match exactly, in order. Cameras ``other`` needs must
        exist here with the same shape; extra cameras here are fine.
        """
        mismatches: list[Mismatch] = []
        if self.joint_names != other.joint_names:
            mismatches.append(Mismatch("joint names", ", ".join(self.joint_names), ", ".join(other.joint_names)))
        for key, image in other.observations.items():
            ours = self.observations.get(key)
            if ours is None:
                mismatches.append(Mismatch(f"camera {key!r}", "no such camera", str(image)))
            elif ours != image:
                mismatches.append(Mismatch(f"camera {key!r}", str(ours), str(image)))
        return mismatches

    def require_compatible(self, other: FeatureSpec) -> None:
        """Raise :class:`FeatureSpecMismatchError` unless ``other`` fits this spec."""
        mismatches = self.check_compatible(other)
        if mismatches:
            raise FeatureSpecMismatchError(mismatches)


@dataclass(frozen=True, slots=True)
class Sample:
    value: np.ndarray
    names: tuple[str, ...]
    timestamp: float
    """``time.monotonic()`` when the value was measured or commanded."""


@dataclass(frozen=True, slots=True)
class _Slot:
    stamp_index: int
    offset: int
    shape: tuple[int, ...]
    dtype: type[np.generic]

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * np.dtype(self.dtype).itemsize


def _layout(spec: FeatureSpec) -> tuple[dict[str, _Slot], dict[str, _Slot], int]:
    """Where each feature lives. Deterministic, so attaching needs only the spec."""
    stamps = len(VECTOR_KEYS) + len(spec.observations)
    offset = stamps * _WORD
    vectors: dict[str, _Slot] = {}
    for index, key in enumerate(VECTOR_KEYS):
        vectors[key] = _Slot(index, offset, (len(spec.joint_names),), np.float64)
        offset += vectors[key].nbytes
    images: dict[str, _Slot] = {}
    for index, key in enumerate(sorted(spec.observations), start=len(VECTOR_KEYS)):
        images[key] = _Slot(index, offset, spec.observations[key].shape, np.uint8)
        offset += -(-images[key].nbytes // _WORD) * _WORD  # keep the next slot 8-byte aligned
    return vectors, images, max(offset, _WORD)


class Store:
    """Latest sample of every feature of one session, shared across processes.

    The session :meth:`create`\\ s it and :meth:`unlink`\\ s it when done; workers
    :meth:`attach` by name with the same spec and :meth:`close` their mapping.
    """

    def __init__(self, spec: FeatureSpec, block: shared_memory.SharedMemory) -> None:
        self.spec = spec
        self._block = block
        self._vectors, self._images, _size = _layout(spec)
        self._stamps: np.ndarray = np.ndarray((len(self._vectors) + len(self._images),), np.float64, block.buf)
        self._views: dict[str, np.ndarray] = {
            key: np.ndarray(slot.shape, slot.dtype, block.buf, slot.offset)
            for key, slot in (*self._vectors.items(), *(("camera:" + k, s) for k, s in self._images.items()))
        }

    @classmethod
    def create(cls, spec: FeatureSpec, session_id: str) -> Store:
        """Allocate the block for a session, with every feature unwritten."""
        _vectors, _images, size = _layout(spec)
        # The creator's pid in the name lets a sweep tell a dead backend's block from a live one's.
        name = f"{BLOCK_PREFIX}{os.getpid()}-{session_id}"
        block = shared_memory.SharedMemory(name=name, create=True, size=size)
        store = cls(spec, block)
        store._stamps[:] = math.nan
        return store

    @classmethod
    def attach(cls, spec: FeatureSpec, name: str) -> Store:
        """Map a block another process created for the same spec.

        Raises:
            ValueError: The block is smaller than this spec needs.
        """
        # Not tracked: the creator owns the block, an attaching process must not unlink it on exit.
        block = shared_memory.SharedMemory(name=name, track=False)
        _vectors, _images, size = _layout(spec)
        if block.size < size:
            block.close()
            raise ValueError(f"Store {name} holds {block.size} bytes, this spec needs {size}")
        return cls(spec, block)

    @property
    def name(self) -> str:
        """The block's name, for another process to attach by."""
        return self._block.name

    def write(self, key: str, values: ArrayLike, *, timestamp: float) -> None:
        """Set the joint vector ``key`` (``state`` or ``action``)."""
        self._write(self._vectors[key], self._views[key], values, timestamp)

    def read(self, key: str) -> Sample | None:
        """The latest joint vector ``key``, or ``None`` if nothing has written it."""
        return self._read(self._vectors[key], self._views[key], self.spec.joint_names)

    def write_observation(self, camera: str, frame: ArrayLike, *, timestamp: float) -> None:
        self._write(self._images[camera], self._views["camera:" + camera], frame, timestamp)

    def read_observation(self, camera: str) -> Sample | None:
        return self._read(self._images[camera], self._views["camera:" + camera], (camera,))

    def _write(self, slot: _Slot, view: np.ndarray, values: ArrayLike, timestamp: float) -> None:
        array = np.asarray(values)
        if array.shape != slot.shape:
            raise ValueError(f"Expected shape {slot.shape}, got {array.shape}")
        view[...] = array
        self._stamps[slot.stamp_index] = timestamp  # last: marks the values as written

    def _read(self, slot: _Slot, view: np.ndarray, names: tuple[str, ...]) -> Sample | None:
        timestamp = float(self._stamps[slot.stamp_index])  # first: see _write
        if math.isnan(timestamp):
            return None
        return Sample(view.copy(), names, timestamp)

    def close(self) -> None:
        """Release this process's mapping."""
        self._views.clear()
        del self._stamps
        self._block.close()

    def unlink(self) -> None:
        """Close and delete the block. Only the creating session calls this."""
        self.close()
        self._block.unlink()


def sweep_stale_blocks() -> list[str]:
    """Delete session blocks whose creating process is gone, as a killed backend leaves them.

    Blocks of a live process are left alone, so a backend starting (or a test run) never
    removes another backend's sessions. Returns the names removed.
    """
    removed = []
    for path in _SHM_DIR.glob(f"{BLOCK_PREFIX}*") if _SHM_DIR.is_dir() else ():
        if not _owner_alive(path.name):
            path.unlink(missing_ok=True)
            removed.append(path.name)
    if removed:
        logger.warning("Removed {} stale runtime session block(s): {}", len(removed), ", ".join(removed))
    return removed


def _owner_alive(name: str) -> bool:
    """Whether the process named in ``pai-<pid>-<session>`` still runs. A name without one is stale."""
    pid = name.removeprefix(BLOCK_PREFIX).partition("-")[0]
    if not pid.isdigit():
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # another user's process: alive, and not ours to judge
    # shortcut: a reused pid keeps a stale block until that process exits too; harmless, it is only memory.
    return True
