# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Isolated Tier 2 inference worker running ``lerobot.bundle.BundleEngine`` without PyTorch."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import BinaryIO

import numpy as np


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        part = stream.read(count - len(chunks))
        if not part:
            msg = f"Unexpected EOF while reading {count} bytes from host (got {len(chunks)})."
            raise EOFError(msg)
        chunks.extend(part)
    return bytes(chunks)


def _write_message(stream: BinaryIO, header: dict[str, object], payload: bytes = b"") -> None:
    encoded = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    stream.write(encoded)
    if payload:
        stream.write(payload)
    stream.flush()


def _load_bundle(bundle_dir: Path, task: str) -> object:
    from lerobot.bundle.engine import BundleEngine
    from lerobot.bundle.manifest import Bundle

    bundle = Bundle.read(bundle_dir)
    return BundleEngine(bundle, task=task)


def _serve(bundle_dir: Path, task: str, protocol_out: BinaryIO) -> int:
    try:
        engine = _load_bundle(bundle_dir, task)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        _write_message(protocol_out, {"type": "error", "error": type(exc).__name__, "message": str(exc)})
        return 1

    has_history = bool(getattr(engine, "_history", None))
    has_task_tokens = getattr(engine, "_task_encoder", None) is not None
    _write_message(
        protocol_out,
        {
            "type": "ready",
            "frame_shapes": {name: list(shape) for name, shape in engine.frame_shapes.items()},
            "frame_dtypes": dict(engine.frame_dtypes),
            "action_dim": int(engine.action_dim),
            "chunk_length": int(engine.chunk_length),
            "has_history": has_history,
            "has_task_tokens": has_task_tokens,
            "task": str(engine.task),
            "torch_loaded": "torch" in sys.modules,
        },
    )

    stdin = sys.stdin.buffer
    while True:
        raw_line = stdin.readline()
        if not raw_line:
            return 0
        try:
            request = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _write_message(protocol_out, {"ok": False, "error": type(exc).__name__, "message": str(exc)})
            return 1

        cmd = request.get("cmd")
        if cmd == "close":
            _write_message(protocol_out, {"ok": True})
            return 0
        if cmd == "reset":
            try:
                engine.reset()
            except Exception as exc:  # noqa: BLE001
                _write_message(protocol_out, {"ok": False, "error": type(exc).__name__, "message": str(exc)})
            else:
                _write_message(protocol_out, {"ok": True})
            continue
        if cmd in {"predict_chunk", "get_action"}:
            try:
                frame: dict[str, np.ndarray] = {}
                for spec in request.get("arrays", []):
                    raw_bytes = _read_exact(stdin, int(spec["nbytes"]))
                    dtype = np.dtype(spec["dtype"])
                    shape = tuple(int(dim) for dim in spec["shape"])
                    array = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape).copy()
                    frame[str(spec["name"])] = array
                requested_task = request.get("task")
                if isinstance(requested_task, str) and requested_task and requested_task != engine.task:
                    engine.set_task(requested_task)
                if cmd == "predict_chunk":
                    result = np.ascontiguousarray(engine.predict_chunk(frame))
                else:
                    action = engine.get_action(frame)
                    if action is None:
                        msg = "BundleEngine.get_action returned None."
                        raise RuntimeError(msg)
                    result = np.ascontiguousarray(np.atleast_2d(action))
                payload = result.tobytes()
                _write_message(
                    protocol_out,
                    {
                        "ok": True,
                        "shape": list(result.shape),
                        "dtype": str(result.dtype),
                        "nbytes": len(payload),
                    },
                    payload,
                )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                _write_message(protocol_out, {"ok": False, "error": type(exc).__name__, "message": str(exc)})
            continue

        _write_message(protocol_out, {"ok": False, "error": "ValueError", "message": f"Unknown command: {cmd!r}"})


def main() -> int:
    """Serve BundleEngine inference over a pickle-free stdin/stdout pipe."""
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: infer.py <bundle_dir> [task]\n")
        return 2
    bundle_dir = Path(sys.argv[1]).expanduser().resolve()
    task = sys.argv[2] if len(sys.argv) > 2 else ""

    # Reserve original stdout exclusively for the binary protocol; route library output to stderr.
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    protocol_out = os.fdopen(protocol_fd, "wb", buffering=0)
    try:
        return _serve(bundle_dir, task, protocol_out)
    finally:
        protocol_out.close()


if __name__ == "__main__":
    raise SystemExit(main())
