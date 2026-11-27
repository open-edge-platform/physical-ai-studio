# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch-free Physical AI Runtime adapter for exported ``.lerobot-bundle`` packages.

This module implements:
- **Tier 3 (in-process)** execution of direct OpenVINO bundles (``.xml`` + ``.bin``) using only
  ``openvino``, ``numpy``, and ``safetensors`` (zero ``torch``, zero ``executorch``, zero
  ``lerobot``, zero ``isolate``).
- **Tier 2 (isolated worker)** execution of ExecuTorch bundles (``.pte``) via ``_worker/infer.py``
  launched with ``physicalai.isolate.start()`` inside the cached ``envs/export/`` environment.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import math
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # nosec B404
import sys
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Protocol

import numpy as np
from physicalai import isolate
from physicalai.inference.adapters.base import RuntimeAdapter
from physicalai.inference.adapters.registry import adapter_registry
from physicalai.inference.callbacks.base import Callback
from safetensors.numpy import load_file as load_safetensors

from .environments import export_isolate_environment, export_worker_env, infer_worker_resource

if TYPE_CHECKING:
    from physicalai.inference.model import InferenceModel

logger = logging.getLogger(__name__)

_FORMAT_VERSION = 1
_MANIFEST_FILE = "export.json"
_ACTION_KEY = "action"
_NOISE_KEY = "noise"
_TASK_EMBEDDINGS_KEY = "task_embeddings"
_TASK_LENGTH_KEY = "task_length"
_GENERATED_TOKENS_KEY = "generated_tokens"
_DECODED_ACTIONS_KEY = "decoded_actions"

_ALLOWED_RUNTIME_IMPORTS: dict[str, frozenset[str]] = {
    "openvino": frozenset({"openvino", "lerobot_openvino_enablement.runtime"}),
    "executorch": frozenset({"openvino", "executorch"}),
}

_PROMPT_OPS: dict[str, Callable[..., str]] = {
    "strip": lambda text: text.strip(),
    "lower": lambda text: text.lower(),
    "replace": lambda text, old, new: text.replace(old, new),
    "keep": lambda text, *kinds: "".join(c for c in text if any(_is_char_kind(c, kind) for kind in kinds)),
    "default": lambda text, value: text or value,
    "template": lambda text, template: template.replace("{task}", text),
}

_TOKENIZE_SCRIPT = """
import json, sys
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(sys.argv[1])
tokenizer.no_padding()
tokenizer.no_truncation()
print(json.dumps([tokenizer.encode(text).ids for text in json.load(sys.stdin)]))
"""

_CLEAN_UP_SPACES = (
    (" .", "."),
    (" ?", "?"),
    (" !", "!"),
    (" ,", ","),
    (" ' ", "'"),
    (" n't", "n't"),
    (" 'm", "'m"),
    (" 's", "'s"),
    (" 've", "'ve"),
    (" 're", "'re"),
)


def _is_char_kind(character: str, kind: str) -> bool:
    if kind == "alnum":
        return character.isalnum()
    if kind == "space":
        return character.isspace()
    return character == kind


def _validate_relative_bundle_path(bundle_dir: Path, name: str | None) -> Path | None:
    if name is None:
        return None
    relative = Path(name)
    if relative.anchor or ".." in relative.parts or not relative.parts:
        msg = f"Bundle manifest references a file outside the bundle folder: {name!r}."
        raise ValueError(msg)
    candidate = bundle_dir / relative
    if candidate.is_symlink():
        msg = f"Bundle file cannot be a symbolic link: {name!r}."
        raise ValueError(msg)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(bundle_dir):
        msg = f"Bundle file escapes the bundle folder: {name!r}."
        raise ValueError(msg)
    return resolved


@dataclass(frozen=True)
class _ProgramFiles:
    file: str
    data: str | None = None


@dataclass(frozen=True)
class _ProgramRun:
    name: str
    program: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class _TaskTokensSpec:
    prompt_ops: tuple[tuple[str, ...], ...]
    tokenizer: str
    embeddings: str
    known_tasks: str
    max_tokens: int


@dataclass(frozen=True)
class _ActionTokensSpec:
    tokenizer: str
    clean_up_spaces: bool
    id_offset: int
    end: int
    prefix: tuple[int, ...]
    start: tuple[int, ...]
    min_token: int
    scale: float
    horizon: int
    action_dim: int


@dataclass(frozen=True)
class _BundleManifest:
    backend: str
    runtime: str
    runtime_imports: tuple[str, ...]
    programs: dict[str, _ProgramFiles]
    runs: tuple[_ProgramRun, ...]
    first_runs: tuple[tuple[_ProgramRun, ...], ...] = ()
    constants: str | None = None
    task_tokens: _TaskTokensSpec | None = None
    action_tokens: _ActionTokensSpec | None = None
    noise_shape: tuple[int, ...] | None = None
    history: dict[str, int] = field(default_factory=dict)
    carry: dict[str, tuple[int, ...] | None] = field(default_factory=dict)
    task: str | None = None
    action_names: tuple[str, ...] | None = None
    action_scale: tuple[float, ...] = ()
    tolerance: float = 0.0
    test_case: str = "test.safetensors"

    @classmethod
    def load(cls, bundle_dir: Path) -> _BundleManifest:
        manifest_path = _validate_relative_bundle_path(bundle_dir, _MANIFEST_FILE)
        if manifest_path is None or not manifest_path.is_file():
            msg = f"LeRobot bundle is missing {_MANIFEST_FILE}: {bundle_dir}"
            raise FileNotFoundError(msg)
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            msg = f"Expected a JSON object in {_MANIFEST_FILE}."
            raise TypeError(msg)
        version = raw.get("format_version")
        if version != _FORMAT_VERSION:
            msg = f"Unsupported LeRobot bundle format_version {version!r}; expected {_FORMAT_VERSION}."
            raise ValueError(msg)

        runtime = str(raw["runtime"])
        runtime_imports = tuple(str(item) for item in raw.get("runtime_imports", ()))
        allowed_imports = _ALLOWED_RUNTIME_IMPORTS.get(runtime)
        if allowed_imports is None:
            msg = f"Unsupported LeRobot bundle runtime {runtime!r}; expected one of {sorted(_ALLOWED_RUNTIME_IMPORTS)}."
            raise ValueError(msg)
        disallowed = sorted(set(runtime_imports) - allowed_imports)
        if disallowed:
            msg = f"LeRobot bundle requests disallowed runtime_imports for {runtime!r}: {disallowed}."
            raise ValueError(msg)

        programs = {
            str(name): _ProgramFiles(file=str(spec["file"]), data=spec.get("data"))
            for name, spec in raw["programs"].items()
        }
        runs = tuple(
            _ProgramRun(
                name=str(item["name"]),
                program=str(item["program"]),
                inputs=tuple(str(v) for v in item["inputs"]),
                outputs=tuple(str(v) for v in item["outputs"]),
            )
            for item in raw["runs"]
        )
        first_runs = tuple(
            tuple(
                _ProgramRun(
                    name=str(item["name"]),
                    program=str(item["program"]),
                    inputs=tuple(str(v) for v in item["inputs"]),
                    outputs=tuple(str(v) for v in item["outputs"]),
                )
                for item in chunk
            )
            for chunk in raw.get("first_runs", ())
        )
        raw_task_tokens = raw.get("task_tokens")
        task_tokens = (
            None
            if raw_task_tokens is None
            else _TaskTokensSpec(
                prompt_ops=tuple(tuple(str(part) for part in op) for op in raw_task_tokens["prompt_ops"]),
                tokenizer=str(raw_task_tokens["tokenizer"]),
                embeddings=str(raw_task_tokens["embeddings"]),
                known_tasks=str(raw_task_tokens["known_tasks"]),
                max_tokens=int(raw_task_tokens["max_tokens"]),
            )
        )
        raw_action_tokens = raw.get("action_tokens")
        action_tokens = (
            None
            if raw_action_tokens is None
            else _ActionTokensSpec(
                tokenizer=str(raw_action_tokens["tokenizer"]),
                clean_up_spaces=bool(raw_action_tokens["clean_up_spaces"]),
                id_offset=int(raw_action_tokens["id_offset"]),
                end=int(raw_action_tokens["end"]),
                prefix=tuple(int(i) for i in raw_action_tokens["prefix"]),
                start=tuple(int(i) for i in raw_action_tokens["start"]),
                min_token=int(raw_action_tokens["min_token"]),
                scale=float(raw_action_tokens["scale"]),
                horizon=int(raw_action_tokens["horizon"]),
                action_dim=int(raw_action_tokens["action_dim"]),
            )
        )
        noise_shape = None if raw.get("noise_shape") is None else tuple(int(d) for d in raw["noise_shape"])
        history = {str(k): int(v) for k, v in (raw.get("history") or {}).items()}
        carry = {
            str(k): None if v is None else tuple(int(d) for d in v) for k, v in (raw.get("carry") or {}).items()
        }
        action_scale = tuple(float(v) for v in raw["action_scale"])
        tolerance = float(raw["tolerance"])
        if not math.isfinite(tolerance) or tolerance < 0:
            msg = f"Invalid bundle tolerance: {tolerance!r}."
            raise ValueError(msg)
        if not action_scale or not all(math.isfinite(s) and s > 0 for s in action_scale):
            msg = f"Invalid bundle action_scale: {action_scale!r}."
            raise ValueError(msg)

        manifest = cls(
            backend=str(raw["backend"]),
            runtime=runtime,
            runtime_imports=runtime_imports,
            programs=programs,
            runs=runs,
            first_runs=first_runs,
            constants=raw.get("constants"),
            task_tokens=task_tokens,
            action_tokens=action_tokens,
            noise_shape=noise_shape,
            history=history,
            carry=carry,
            task=raw.get("task"),
            action_names=None if raw.get("action_names") is None else tuple(str(n) for n in raw["action_names"]),
            action_scale=action_scale,
            tolerance=tolerance,
            test_case=str(raw.get("test_case", "test.safetensors")),
        )
        manifest.validate_files(bundle_dir)
        return manifest

    def validate_files(self, bundle_dir: Path) -> None:
        _validate_relative_bundle_path(bundle_dir, self.test_case)
        _validate_relative_bundle_path(bundle_dir, self.constants)
        for files in self.programs.values():
            _validate_relative_bundle_path(bundle_dir, files.file)
            _validate_relative_bundle_path(bundle_dir, files.data)
        if self.task_tokens is not None:
            _validate_relative_bundle_path(bundle_dir, self.task_tokens.tokenizer)
            _validate_relative_bundle_path(bundle_dir, self.task_tokens.embeddings)
            _validate_relative_bundle_path(bundle_dir, self.task_tokens.known_tasks)
        if self.action_tokens is not None:
            _validate_relative_bundle_path(bundle_dir, self.action_tokens.tokenizer)


class _TaskEncoder:
    """Pure NumPy + optional ``tokenizers`` encoder for ``export.json`` ``task_tokens``."""

    def __init__(self, bundle_dir: Path, spec: _TaskTokensSpec) -> None:
        self._tokenizer_file = _validate_relative_bundle_path(bundle_dir, spec.tokenizer)
        embeddings_path = _validate_relative_bundle_path(bundle_dir, spec.embeddings)
        known_path = _validate_relative_bundle_path(bundle_dir, spec.known_tasks)
        if self._tokenizer_file is None or embeddings_path is None or known_path is None:
            msg = "Missing task_tokens files in bundle."
            raise FileNotFoundError(msg)
        self._table = _read_embedding_table(embeddings_path)
        self._prompt_ops = spec.prompt_ops
        self._max_tokens = spec.max_tokens
        self._ids: dict[str, list[int]] = json.loads(known_path.read_text(encoding="utf-8"))
        if not self._ids:
            msg = "Bundle known_tasks.json must contain at least one task."
            raise ValueError(msg)

    def _clean(self, task: str) -> str:
        text = task
        for name, *args in self._prompt_ops:
            op = _PROMPT_OPS.get(name)
            if op is None:
                msg = f"Unsupported prompt op in bundle manifest: {name!r}."
                raise ValueError(msg)
            text = op(text, *args)
        return text

    def ids(self, task: str) -> list[int]:
        if task not in self._ids:
            if importlib.util.find_spec("tokenizers") is None:
                msg = (
                    f"Task {task!r} was not pre-tokenized in known_tasks.json, and the optional "
                    "'tokenizers' package is not installed."
                )
                raise ValueError(msg)
            known, known_ids = next(iter(self._ids.items()))
            result = subprocess.run(  # nosec B603
                [sys.executable, "-P", "-c", _TOKENIZE_SCRIPT, str(self._tokenizer_file)],
                input=json.dumps([self._clean(known), self._clean(task)]),
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )
            if result.returncode != 0:
                msg = f"Tokenizing task failed: {result.stderr.strip()}"
                raise RuntimeError(msg)
            checked, new_ids = json.loads(result.stdout)
            if checked != known_ids:
                msg = "Installed tokenizers version does not match the bundle's known_tasks.json."
                raise ValueError(msg)
            self._ids[task] = new_ids
        return self._ids[task]

    def encode(self, task: str) -> dict[str, np.ndarray]:
        token_ids = self.ids(task)
        if len(token_ids) > self._max_tokens:
            msg = f"Task {task!r} is {len(token_ids)} tokens long, exceeding max_tokens={self._max_tokens}."
            raise ValueError(msg)
        rows = np.zeros((self._max_tokens, self._table.shape[1]), dtype=np.float32)
        looked_up = np.asarray(self._table[token_ids])
        if looked_up.dtype == np.uint16:
            looked_up = (looked_up.astype(np.uint32) << 16).view(np.float32)
        rows[: len(token_ids)] = looked_up
        return {_TASK_EMBEDDINGS_KEY: rows, _TASK_LENGTH_KEY: np.array([len(token_ids)], dtype=np.int64)}


def _read_embedding_table(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        header_size = int.from_bytes(file.read(8), "little")
        header = json.loads(file.read(header_size))
    header.pop("__metadata__", None)
    ((_, tensor),) = header.items()
    dtypes = {"BF16": np.uint16, "F32": np.float32}
    if tensor["dtype"] not in dtypes:
        msg = f"{path.name} holds unsupported dtype {tensor['dtype']!r}; expected BF16 or F32."
        raise ValueError(msg)
    start, _ = tensor["data_offsets"]
    return np.memmap(path, dtypes[tensor["dtype"]], "r", 8 + header_size + start, tuple(tensor["shape"]))


class _ActionDecoder:
    """Pure NumPy FAST action token decoder (byte-level BPE + orthonormal DCT-III)."""

    def __init__(self, bundle_dir: Path, spec: _ActionTokensSpec) -> None:
        tokenizer_path = _validate_relative_bundle_path(bundle_dir, spec.tokenizer)
        if tokenizer_path is None:
            msg = "Missing action_tokens tokenizer in bundle."
            raise FileNotFoundError(msg)
        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        model, decoder = tokenizer.get("model") or {}, tokenizer.get("decoder") or {}
        if model.get("type") != "BPE" or decoder.get("type") != "ByteLevel" or tokenizer.get("added_tokens"):
            msg = "Bundle FAST tokenizer must be a byte-level BPE without added tokens."
            raise ValueError(msg)
        self._spec = spec
        self._text = {token_id: token for token, token_id in model["vocab"].items()}
        printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
        others = [byte for byte in range(256) if byte not in printable]
        self._bytes = {chr(b): b for b in printable} | {chr(256 + i): b for i, b in enumerate(others)}
        n = spec.horizon
        time_idx, freq_idx = np.arange(n)[:, None], np.arange(n)[None, :]
        self._inverse_dct = np.sqrt(2 / n) * np.cos(np.pi * (2 * time_idx + 1) * freq_idx / (2 * n))
        self._inverse_dct[:, 0] = np.sqrt(1 / n)

    def decode(self, token_ids: np.ndarray) -> np.ndarray:
        spec = self._spec
        ids = [int(i) for i in np.asarray(token_ids).reshape(-1)]
        if spec.start and ids[: len(spec.start)] != list(spec.start):
            msg = "Exported policy generated tokens without required action prefix."
            raise RuntimeError(msg)
        if spec.end in ids:
            ids = ids[: ids.index(spec.end)]
        prefix = list(spec.prefix)
        index = 0
        while index <= len(ids) - len(prefix):
            if ids[index : index + len(prefix)] == prefix:
                ids = ids[:index] + ids[index + len(prefix) :]
            else:
                index += 1
        fast_ids = [spec.id_offset - i for i in ids]
        size = spec.horizon * spec.action_dim
        if any(i < 0 for i in fast_ids):
            coefficients = np.zeros((spec.horizon, spec.action_dim))
        else:
            data = bytearray()
            for token in (self._text[i] for i in fast_ids if i in self._text):
                if all(c in self._bytes for c in token):
                    data.extend(self._bytes[c] for c in token)
                else:
                    data.extend(token.encode())
            text = data.decode("utf-8", errors="replace")
            if spec.clean_up_spaces:
                for old, new in _CLEAN_UP_SPACES:
                    text = text.replace(old, new)
            values = np.array([ord(c) for c in text], dtype=np.int64)[:size] + spec.min_token
            coefficients = np.pad(values, (0, size - len(values))).reshape(spec.horizon, spec.action_dim)
        return (self._inverse_dct @ (coefficients / spec.scale)).astype(np.float32)


class _CompiledOVProgram:
    """One compiled OpenVINO IR part for Tier 3 bundle execution."""

    def __init__(self, core: Any, path: Path, device: str) -> None:
        self._compiled = core.compile_model(str(path), device)
        self._input_dtypes = [port.get_element_type().to_dtype() for port in self._compiled.inputs]
        self._outputs = list(self._compiled.outputs)

    def __call__(self, *inputs: np.ndarray) -> tuple[np.ndarray, ...]:
        prepared = []
        for index, value in enumerate(inputs):
            array = np.asarray(value)
            expected = self._input_dtypes[index]
            if array.dtype != expected:
                array = array.astype(expected, copy=False)
            prepared.append(np.require(array, requirements="C"))
        results = self._compiled(prepared)
        return tuple(np.array(results[out], copy=True) for out in self._outputs)


class _BundleExecutor(Protocol):
    frame_shapes: dict[str, tuple[int, ...]]
    frame_dtypes: dict[str, np.dtype[Any]]
    action_dim: int
    chunk_length: int
    has_history: bool

    def predict_chunk(self, frame: Mapping[str, np.ndarray], *, task: str | None = None) -> np.ndarray: ...
    def get_action(self, frame: Mapping[str, np.ndarray], *, task: str | None = None) -> np.ndarray: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...


class OpenVINOBundleExecutor:
    """Tier 3 pure ``openvino`` + ``numpy`` + ``safetensors`` executor for ``export.json`` v1."""

    def __init__(
        self,
        bundle_dir: Path,
        manifest: _BundleManifest,
        *,
        device: str = "CPU",
        cache_dir: str | Path | None = None,
        task: str = "",
        verify: bool = True,
    ) -> None:
        import openvino as ov

        self._bundle_dir = bundle_dir
        self._manifest = manifest
        self._fixed_task = manifest.task
        self._task_encoder = (
            None if manifest.task_tokens is None else _TaskEncoder(bundle_dir, manifest.task_tokens)
        )
        self._action_decoder = (
            None if manifest.action_tokens is None else _ActionDecoder(bundle_dir, manifest.action_tokens)
        )
        self._task = task or self._fixed_task or ""
        self._task_values = self._encode_task(self._task)
        self._constants: dict[str, np.ndarray] = (
            {}
            if manifest.constants is None
            else dict(load_safetensors(str(bundle_dir / manifest.constants)))
        )
        self._scale = np.asarray(manifest.action_scale, dtype=np.float32)
        self.action_dim = len(self._scale)
        self._chunks_spec = (*manifest.first_runs, manifest.runs)
        self._host_noise_shape = (
            None
            if any(_NOISE_KEY in run.outputs for runs in self._chunks_spec for run in runs)
            else manifest.noise_shape
        )

        test_case = dict(load_safetensors(str(bundle_dir / manifest.test_case)))
        expected = test_case.pop(_ACTION_KEY, None)
        noise = test_case.pop(_NOISE_KEY, None)
        if expected is None or expected.ndim != 2 or expected.shape[1] != self.action_dim:
            msg = f"Bundle test case has invalid expected actions in {bundle_dir / manifest.test_case}."
            raise ValueError(msg)

        frames: list[dict[str, np.ndarray]] = [test_case]
        if manifest.history or manifest.carry or manifest.first_runs:
            ticks = {value.shape[0] if value.ndim else 0 for value in test_case.values()}
            if len(ticks) != 1 or 0 in ticks:
                msg = "Bundle test case frames do not share a valid leading tick axis."
                raise ValueError(msg)
            frames = [{name: value[tick] for name, value in test_case.items()} for tick in range(ticks.pop())]

        self.frame_shapes = {name: tuple(value.shape) for name, value in frames[-1].items()}
        self.frame_dtypes = {name: value.dtype for name, value in frames[-1].items()}
        self.has_history = bool(manifest.history)
        self._history: dict[str, deque[np.ndarray]] = {
            name: deque(maxlen=size) for name, size in manifest.history.items()
        }
        self._carry: dict[str, np.ndarray] = self._reset_carry()
        self._actions: deque[np.ndarray] = deque()
        self._chunks = 0

        core = ov.Core()
        if cache_dir is not None:
            resolved_cache = Path(cache_dir).expanduser().resolve()
            resolved_cache.mkdir(parents=True, exist_ok=True)
            core.set_property({"CACHE_DIR": str(resolved_cache)})

        self._programs = {
            name: _CompiledOVProgram(core, bundle_dir / files.file, device)
            for name, files in manifest.programs.items()
        }

        if verify:
            self.chunk_length = self._verify_test_case(frames, noise, expected)
        else:
            self.chunk_length = int(expected.shape[0])

    def _encode_task(self, task: str) -> dict[str, np.ndarray]:
        if self._task_encoder is not None:
            return self._task_encoder.encode(task)
        if self._fixed_task is not None and task != self._fixed_task:
            msg = f"This exported policy runs only task {self._fixed_task!r}, not {task!r}."
            raise ValueError(msg)
        return {}

    def _reset_carry(self) -> dict[str, np.ndarray]:
        return {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in self._manifest.carry.items()
            if shape is not None
        }

    def _observe(self, frame: Mapping[str, np.ndarray]) -> None:
        for name, queue in self._history.items():
            value = np.array(frame[name], copy=True)
            if queue:
                queue.append(value)
            else:
                queue.extend([value] * (queue.maxlen or 1))

    def _run_chunk(
        self,
        frame: Mapping[str, np.ndarray],
        task_values: Mapping[str, np.ndarray],
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        values: dict[str, np.ndarray] = {
            name: np.stack(self._history[name]) if name in self._history else frame[name]
            for name in self.frame_shapes
        }
        values.update(self._constants)
        values.update(task_values)
        values.update(self._carry)
        if noise is None and self._host_noise_shape is not None:
            noise = np.random.default_rng().standard_normal(self._host_noise_shape).astype(np.float32)
        if noise is not None:
            values[_NOISE_KEY] = noise

        chunk_index = min(self._chunks, len(self._chunks_spec) - 1)
        runs = self._chunks_spec[chunk_index]
        for run in runs:
            if _NOISE_KEY in values and _NOISE_KEY in run.outputs:
                continue
            if (
                self._action_decoder is not None
                and _DECODED_ACTIONS_KEY in run.inputs
                and _DECODED_ACTIONS_KEY not in values
            ):
                values[_DECODED_ACTIONS_KEY] = self._action_decoder.decode(values[_GENERATED_TOKENS_KEY])
            program = self._programs[run.program]
            outputs = program(*(values[name] for name in run.inputs))
            values.update(zip(run.outputs, outputs, strict=True))

        self._carry = {name: values[name] for name in self._manifest.carry if name in values}
        self._chunks += 1
        actions = np.asarray(values[_ACTION_KEY])
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != self.action_dim:
            msg = f"Exported OpenVINO bundle returned invalid action shape {actions.shape}."
            raise RuntimeError(msg)
        if not np.isfinite(actions).all():
            msg = "Exported OpenVINO bundle returned non-finite actions."
            raise RuntimeError(msg)
        return actions

    def _verify_test_case(
        self,
        frames: list[dict[str, np.ndarray]],
        noise: np.ndarray | None,
        expected: np.ndarray,
    ) -> int:
        task_values = self._encode_task(self._fixed_task or "")
        has_first_runs = len(self._chunks_spec) > 1
        if has_first_runs:
            played: list[np.ndarray] = []
            chunks = 0
            last_chunk = np.empty((0, self.action_dim), dtype=np.float32)
            for frame in frames:
                self._observe(frame)
                if not self._actions:
                    chunk_noise = None if noise is None else noise[chunks]
                    last_chunk = self._run_chunk(frame, task_values, chunk_noise)
                    self._actions.extend(last_chunk)
                    chunks += 1
                played.append(self._actions.popleft())
            actual = np.stack(played)
            chunk_length = len(last_chunk)
        elif self._manifest.carry:
            actual = np.empty((0, self.action_dim), dtype=np.float32)
            for index, frame in enumerate(frames):
                self._observe(frame)
                actual = self._run_chunk(frame, task_values, None if noise is None else noise[index])
            chunk_length = len(actual)
        else:
            for frame in frames:
                self._observe(frame)
            actual = self._run_chunk(frames[-1], task_values, noise)
            chunk_length = len(actual)

        self.reset()
        if actual.shape != expected.shape:
            msg = f"OpenVINO bundle returned actions shaped {actual.shape}, expected {expected.shape}."
            raise RuntimeError(msg)
        error = float((np.abs(actual - expected) / self._scale).max())
        logger.info("OpenVINO bundle test case: max normalized diff %.2e (tolerance %g)", error, self._manifest.tolerance)
        if not np.isfinite(error) or error > self._manifest.tolerance:
            msg = f"OpenVINO bundle failed parity check ({error:.2e} > {self._manifest.tolerance})."
            raise RuntimeError(msg)
        return chunk_length

    def _set_task_if_needed(self, task: str | None) -> None:
        if task is not None and task and task != self._task:
            self._task_values = self._encode_task(task)
            self._task = task

    def predict_chunk(self, frame: Mapping[str, np.ndarray], *, task: str | None = None) -> np.ndarray:
        self._set_task_if_needed(task)
        self._observe(frame)
        return self._run_chunk(frame, self._task_values)

    def get_action(self, frame: Mapping[str, np.ndarray], *, task: str | None = None) -> np.ndarray:
        self._set_task_if_needed(task)
        self._observe(frame)
        if not self._actions:
            self._actions.extend(self._run_chunk(frame, self._task_values))
        return self._actions.popleft()

    def reset(self) -> None:
        self._actions.clear()
        for queue in self._history.values():
            queue.clear()
        self._carry = self._reset_carry()
        self._chunks = 0

    def close(self) -> None:
        self.reset()


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        part = stream.read(count - len(chunks))
        if not part:
            msg = f"Unexpected EOF from isolated bundle worker (expected {count} bytes, got {len(chunks)})."
            raise EOFError(msg)
        chunks.extend(part)
    return bytes(chunks)


def _prepare_export_isolate_env(expected_key: str | None) -> isolate.PreparedEnvironment:
    """Prepare the cached ``envs/export/`` environment, attaching Engine markers when available."""
    try:
        from physicalai.engine import prepare_environment
    except ImportError:
        env = export_isolate_environment()
        return isolate.prepare(env, provider=isolate.UvProvider())

    worker_env = export_worker_env()
    if expected_key is not None and worker_env.key() != expected_key:
        msg = (
            f"Exported bundle was built for environment key {expected_key[:12]}..., "
            f"which does not match this plugin's export environment ({worker_env.key()[:12]}...). "
            "Re-export the policy or pass an explicit `interpreter=`."
        )
        raise RuntimeError(msg)
    return prepare_environment(worker_env, "cpu").isolate


def _read_package_metadata(bundle_dir: Path) -> dict[str, Any]:
    package_meta = bundle_dir.parent / "package.json"
    if package_meta.is_file() and not package_meta.is_symlink():
        try:
            data = json.loads(package_meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}
    return {}


class IsolatedBundleExecutor:
    """Tier 2 isolated ``BundleEngine`` executor (used for ExecuTorch ``.pte`` bundles)."""

    def __init__(
        self,
        bundle_dir: Path,
        *,
        task: str = "",
        interpreter: str | Path | None = None,
    ) -> None:
        self._process: isolate.Session | subprocess.Popen[bytes] | None = None
        self._closed = False
        self.torch_loaded: bool | None = None

        metadata = _read_package_metadata(bundle_dir)
        expected_key = metadata.get("worker_env_key") if isinstance(metadata.get("worker_env_key"), str) else None
        entrypoint = infer_worker_resource()
        args = [str(bundle_dir), task] if task else [str(bundle_dir)]

        if interpreter is not None:
            python = Path(interpreter).expanduser()
            with resources.as_file(entrypoint) as worker_path:
                self._process = subprocess.Popen(  # nosec B603  # User-managed interpreter escape hatch.
                    [str(python), "-P", str(worker_path), *args],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=None,
                    shell=False,
                )
        else:
            prepared = _prepare_export_isolate_env(expected_key)
            self._process = isolate.start(
                prepared,
                entrypoint=entrypoint,
                args=args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
            )

        ready = self._read_header()
        if ready.get("type") != "ready":
            error_msg = ready.get("message", f"Unexpected worker handshake: {ready!r}")
            self.close()
            raise RuntimeError(f"LeRobot bundle inference worker failed to start: {error_msg}")

        self.frame_shapes = {name: tuple(int(d) for d in shape) for name, shape in ready["frame_shapes"].items()}
        self.frame_dtypes = {name: np.dtype(dtype) for name, dtype in ready["frame_dtypes"].items()}
        self.action_dim = int(ready["action_dim"])
        self.chunk_length = int(ready["chunk_length"])
        self.has_history = bool(ready["has_history"])
        self.torch_loaded = bool(ready.get("torch_loaded", False))

    def _streams(self) -> tuple[BinaryIO, BinaryIO]:
        if self._closed or self._process is None or self._process.stdin is None or self._process.stdout is None:
            msg = "Isolated LeRobot bundle inference worker is closed."
            raise RuntimeError(msg)
        return self._process.stdin, self._process.stdout

    def _read_header(self) -> dict[str, Any]:
        _, stdout = self._streams()
        line = stdout.readline()
        if not line:
            msg = "Isolated LeRobot bundle inference worker exited unexpectedly."
            raise RuntimeError(msg)
        return json.loads(line.decode("utf-8"))

    def _send_frame_cmd(self, cmd: str, frame: Mapping[str, np.ndarray], task: str | None) -> np.ndarray:
        stdin, stdout = self._streams()
        specs = []
        buffers = []
        for name in self.frame_shapes:
            arr = np.ascontiguousarray(frame[name], dtype=self.frame_dtypes[name])
            raw = arr.tobytes()
            specs.append({"name": name, "shape": list(arr.shape), "dtype": str(arr.dtype), "nbytes": len(raw)})
            buffers.append(raw)
        header: dict[str, object] = {"cmd": cmd, "arrays": specs}
        if task:
            header["task"] = task
        stdin.write(json.dumps(header, separators=(",", ":")).encode("utf-8") + b"\n")
        for buf in buffers:
            stdin.write(buf)
        stdin.flush()

        response = self._read_header()
        if not response.get("ok"):
            msg = f"Isolated bundle worker error ({response.get('error')}): {response.get('message')}"
            raise RuntimeError(msg)
        payload = _read_exact(stdout, int(response["nbytes"]))
        shape = tuple(int(d) for d in response["shape"])
        dtype = np.dtype(response["dtype"])
        return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()

    def predict_chunk(self, frame: Mapping[str, np.ndarray], *, task: str | None = None) -> np.ndarray:
        return self._send_frame_cmd("predict_chunk", frame, task)

    def get_action(self, frame: Mapping[str, np.ndarray], *, task: str | None = None) -> np.ndarray:
        return self._send_frame_cmd("get_action", frame, task)[0]

    def reset(self) -> None:
        stdin, _ = self._streams()
        stdin.write(b'{"cmd":"reset"}\n')
        stdin.flush()
        response = self._read_header()
        if not response.get("ok"):
            msg = f"Isolated bundle worker reset failed: {response.get('message')}"
            raise RuntimeError(msg)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None and process.poll() is None:
                with contextlib.suppress(OSError):
                    process.stdin.write(b'{"cmd":"close"}\n')
                    process.stdin.flush()
                with contextlib.suppress(OSError):
                    process.stdin.close()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            if isinstance(process, isolate.Session):
                process.close()
            elif process.stdout is not None:
                process.stdout.close()


@adapter_registry.register("lerobot_bundle", extensions=(".lerobot-bundle",))
class LeRobotBundleAdapter(RuntimeAdapter):
    """Load and execute exported LeRobot bundles (``.lerobot-bundle``) in Physical AI Runtime.

    - Direct OpenVINO bundles (``runtime == "openvino"``) run in-process via
      :class:`OpenVINOBundleExecutor` (Tier 3: zero PyTorch, zero ExecuTorch, zero LeRobot).
    - ExecuTorch bundles (``runtime == "executorch"``) run in an isolated ``envs/export/``
      worker via :class:`IsolatedBundleExecutor` (Tier 2: zero PyTorch in RAM).
    """

    def __init__(
        self,
        device: str = "CPU",
        *,
        cache_dir: str | Path | None = None,
        verify: bool = True,
        task: str = "",
        interpreter: str | Path | None = None,
    ) -> None:
        super().__init__(device=device)
        self._cache_dir = cache_dir
        self._verify = verify
        self._task = task
        self._interpreter = interpreter
        self._executor: _BundleExecutor | None = None
        self._bundle_runtime: str | None = None

    def default_device(self) -> str:  # noqa: PLR6301
        """Default to portable OpenVINO CPU; callers may select ``GPU``, ``NPU``, or ``AUTO``."""
        return "CPU"

    def load(self, model_path: Path | str) -> None:
        bundle_dir = _resolve_bundle_directory(Path(model_path).expanduser())
        manifest = _BundleManifest.load(bundle_dir)
        self._bundle_runtime = manifest.runtime
        if manifest.runtime == "openvino":
            self._executor = OpenVINOBundleExecutor(
                bundle_dir,
                manifest,
                device=_normalize_ov_device(self.device),
                cache_dir=self._cache_dir,
                task=self._task,
                verify=self._verify,
            )
        elif manifest.runtime == "executorch":
            self._executor = IsolatedBundleExecutor(
                bundle_dir,
                task=self._task,
                interpreter=self._interpreter,
            )
        else:
            msg = f"Unsupported LeRobot bundle runtime {manifest.runtime!r}."
            raise ValueError(msg)

    def predict(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        if self._executor is None:
            msg = "LeRobot bundle not loaded. Call load() first."
            raise RuntimeError(msg)

        frame, task = _to_bundle_frame(inputs, self._executor.frame_shapes, self._executor.frame_dtypes)
        if self._executor.has_history:
            action = self._executor.get_action(frame, task=task)
            return {"action": np.ascontiguousarray(action)[None, None, :]}
        chunk = self._executor.predict_chunk(frame, task=task)
        return {"action": np.ascontiguousarray(chunk)[None, :, :]}

    def reset(self) -> None:
        if self._executor is not None:
            self._executor.reset()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.close()
            self._executor = None

    @property
    def input_names(self) -> list[str]:
        if self._executor is None:
            msg = "LeRobot bundle not loaded. Call load() first."
            raise RuntimeError(msg)
        return []

    @property
    def output_names(self) -> list[str]:
        if self._executor is None:
            msg = "LeRobot bundle not loaded. Call load() first."
            raise RuntimeError(msg)
        return ["action"]


class LeRobotBundleResetCallback(Callback):
    """Forward Runtime episode resets to the loaded :class:`LeRobotBundleAdapter`."""

    def __init__(self) -> None:
        self._adapter: LeRobotBundleAdapter | None = None

    def on_load(self, model: InferenceModel) -> None:
        if model.backend != "lerobot_bundle" or not isinstance(model.adapter, LeRobotBundleAdapter):
            msg = "LeRobot bundle reset callback requires the lerobot_bundle Runtime adapter."
            raise TypeError(msg)
        self._adapter = model.adapter

    def on_reset(self) -> None:
        if self._adapter is not None:
            self._adapter.reset()


def _normalize_ov_device(device: str) -> str:
    stripped = device.strip()
    if stripped.lower() in {"cpu", "gpu", "npu", "auto"}:
        return stripped.upper()
    return stripped


def _resolve_bundle_directory(artifact_path: Path) -> Path:
    if not artifact_path.is_dir():
        msg = f"LeRobot bundle directory not found: {artifact_path}"
        raise FileNotFoundError(msg)
    artifact_parent = artifact_path.parent.resolve(strict=True)
    bundle_root = artifact_path.resolve(strict=True)
    if not bundle_root.is_relative_to(artifact_parent):
        msg = "LeRobot bundle directory escapes its Runtime export directory."
        raise ValueError(msg)
    if any(path.is_symlink() for path in bundle_root.rglob("*")):
        msg = "LeRobot bundles containing symbolic links are not supported."
        raise ValueError(msg)
    if not (bundle_root / _MANIFEST_FILE).is_file():
        msg = f"LeRobot bundle is missing {_MANIFEST_FILE}: {bundle_root}"
        raise FileNotFoundError(msg)
    return bundle_root


def _extract_task(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
        first = value[0]
        return str(first) if isinstance(first, str) else None
    return None


def _to_bundle_frame(
    inputs: Mapping[str, object],
    frame_shapes: Mapping[str, tuple[int, ...]],
    frame_dtypes: Mapping[str, np.dtype[Any]],
) -> tuple[dict[str, np.ndarray], str | None]:
    """Map Runtime observations to the exact keys, shapes, and dtypes expected by the bundle."""
    raw: dict[str, object] = {}
    task: str | None = None
    for key, value in inputs.items():
        if key == "images" and isinstance(value, Mapping):
            for camera_name, image in value.items():
                if not isinstance(camera_name, str):
                    msg = "LeRobot camera names must be strings."
                    raise TypeError(msg)
                raw[f"observation.images.{camera_name}"] = image
        elif key.startswith("images."):
            raw[f"observation.images.{key.removeprefix('images.')}"] = value
        elif key == "state":
            raw["observation.state"] = value
        elif key == "image":
            raw["observation.image"] = value
        elif key == "task":
            task = _extract_task(value)
        elif key.startswith("observation."):
            raw[key] = value

    frame: dict[str, np.ndarray] = {}
    for name, expected_shape in frame_shapes.items():
        if name not in raw:
            msg = f"Expected bundle observation {name!r} not found in inputs (available: {sorted(raw)})."
            raise KeyError(msg)
        frame[name] = _coerce_observation(name, raw[name], expected_shape, frame_dtypes[name])
    return frame, task


def _coerce_observation(
    name: str,
    value: object,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype[Any],
) -> np.ndarray:
    arr = np.asarray(value)
    # Strip a single leading batch axis if Runtime/caller passed batched (1, ...).
    if arr.ndim == len(expected_shape) + 1 and arr.shape[0] == 1:
        arr = arr[0]

    # Convert CHW -> HWC automatically when the bundle expects HWC uint8/float images.
    if (
        arr.ndim == 3
        and len(expected_shape) == 3
        and arr.shape != expected_shape
        and (arr.shape[1], arr.shape[2], arr.shape[0]) == expected_shape
    ):
        arr = np.transpose(arr, (1, 2, 0))
    elif (
        arr.ndim == 3
        and len(expected_shape) == 3
        and arr.shape != expected_shape
        and (arr.shape[2], arr.shape[0], arr.shape[1]) == expected_shape
    ):
        arr = np.transpose(arr, (2, 0, 1))

    if arr.shape != expected_shape:
        msg = f"Bundle observation {name!r} expects shape {expected_shape}, got {arr.shape}."
        raise ValueError(msg)

    if expected_dtype == np.uint8 and np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    elif expected_dtype != arr.dtype:
        arr = arr.astype(expected_dtype, copy=False)

    return np.ascontiguousarray(arr)


__all__ = [
    "IsolatedBundleExecutor",
    "LeRobotBundleAdapter",
    "LeRobotBundleResetCallback",
    "OpenVINOBundleExecutor",
]
