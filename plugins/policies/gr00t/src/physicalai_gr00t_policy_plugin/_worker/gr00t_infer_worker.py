# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serve GR00T inference requests from inside a pinned worker environment.

The host Runtime adapter starts this script and exchanges pickle-free frames
(see ``gr00t_worker_protocol``) over two inherited pipes. Standard output and
standard error stay free for framework logs.

Two package formats are supported:

- ``native``: upstream ``Gr00tPolicy`` on PyTorch (native environment);
- ``openvino``: the Robotics AI Suite single-IR OpenVINO path (patched environment).
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gr00t_worker_common as common
import gr00t_worker_protocol as protocol
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

_REQUEST_FD_ENV = "PHYSICALAI_GR00T_REQUEST_FD"
_RESPONSE_FD_ENV = "PHYSICALAI_GR00T_RESPONSE_FD"


class InferenceServer:
    """Hold one loaded GR00T policy and answer host requests."""

    def __init__(self) -> None:
        """Create a server with no policy loaded."""
        self.policy: Any = None
        self.format: str | None = None

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one request.

        Returns:
            Response message.

        Raises:
            ValueError: If the operation is unknown.
        """
        operation = message.get("op")
        if operation == "hello":
            return {"ok": True, "protocol": common.PROTOCOL_VERSION, **common.identity()}
        if operation == "load":
            return self.load(message)
        if operation == "predict":
            return self.predict(message)
        if operation == "reset":
            if self.policy is not None:
                self.policy.reset()
            return {"ok": True}
        msg = f"Unknown worker operation {operation!r}."
        raise ValueError(msg)

    def load(self, message: dict[str, Any]) -> dict[str, Any]:
        """Load a native checkpoint or an OpenVINO single-IR export.

        Returns:
            Model description used by the host to validate its I/O mapping.

        Raises:
            ValueError: If the package format is unknown.
        """
        common.disable_remote_code()
        package_format = message["format"]
        path = Path(message["path"])
        embodiment_tag = message["embodiment_tag"]
        started = time.monotonic()
        if package_format == "native":
            device = _torch_device(message.get("device", "auto"))
            from gr00t.policy.gr00t_policy import Gr00tPolicy

            self.policy = Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=str(path), device=device, strict=True)
        elif package_format == "openvino":
            device = _openvino_device(message.get("device", "auto"))
            self.policy = _load_openvino_policy(
                path,
                embodiment_tag,
                device,
                static_shape=bool(message.get("static_shape")),
                cache_dir=message.get("cache_dir"),
            )
        else:
            msg = f"Unknown GR00T package format {package_format!r}."
            raise ValueError(msg)
        self.format = package_format
        configs = self.policy.modality_configs
        action_head = self.policy.model.action_head.config
        return {
            "ok": True,
            "device": device,
            "load_seconds": time.monotonic() - started,
            "video_keys": list(configs["video"].modality_keys),
            "state_keys": list(configs["state"].modality_keys),
            "action_keys": list(configs["action"].modality_keys),
            "video_horizon": len(configs["video"].delta_indices),
            "state_horizon": len(configs["state"].delta_indices),
            "action_horizon": len(configs["action"].delta_indices),
            "language_key": self.policy.language_key,
            "model_action_horizon": int(action_head.action_horizon),
            "max_action_dim": int(action_head.max_action_dim),
        }

    def predict(self, message: dict[str, Any]) -> dict[str, Any]:
        """Run one GR00T action-chunk prediction.

        Returns:
            Unbatched per-key actions and the worker-side latency.

        Raises:
            RuntimeError: If no policy is loaded.
        """
        if self.policy is None:
            msg = "Load a GR00T package before predicting."
            raise RuntimeError(msg)
        observation = {
            "video": {key: value[np.newaxis] for key, value in message["video"].items()},
            "state": {key: value[np.newaxis].astype(np.float32) for key, value in message["state"].items()},
            "language": {self.policy.language_key: [[str(message["task"])]]},
        }
        started = time.monotonic()
        with _seeded_noise(message.get("seed")):
            actions, _ = self.policy.get_action(observation)
        latency = time.monotonic() - started
        return {
            "ok": True,
            "actions": {key: np.asarray(value[0], dtype=np.float32) for key, value in actions.items()},
            "latency_seconds": latency,
        }


def _load_openvino_policy(
    path: Path,
    embodiment_tag: str,
    device: str,
    *,
    static_shape: bool,
    cache_dir: str | None,
) -> Any:  # ruff: ignore[any-type]
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.run_gr00t_server import _load_policy_config_only  # ruff: ignore[import-private-name]

    deployment = common.source_dir() / "scripts" / "deployment"
    if str(deployment) not in sys.path:
        sys.path.insert(0, str(deployment))
    from ov_n1d7_forward import setup_openvino_models_n1d7_single

    model_dir = _cache_view(path, Path(cache_dir)) if cache_dir else path
    policy = _load_policy_config_only(str(model_dir), EmbodimentTag.resolve(embodiment_tag))
    setup_openvino_models_n1d7_single(policy, ov_model_path=str(model_dir), device=device, static_shape=static_shape)
    return policy


def _cache_view(model_dir: Path, view: Path) -> Path:
    """Expose *model_dir* through a symlink in a writable cache directory.

    The Suite loader writes OpenVINO's compiled-model cache to
    ``<model_dir>/../.ov_cache``. Loading through this view keeps that cache out
    of the Runtime package, which stays unchanged and may be read-only.

    Returns:
        The symlinked model directory inside *view*.
    """
    view.mkdir(parents=True, exist_ok=True)
    link = view / model_dir.name
    if link.is_symlink() and link.resolve() != model_dir.resolve():
        link.unlink()
    if not link.exists():
        link.symlink_to(model_dir.resolve(), target_is_directory=True)
    return link


def _torch_device(requested: str) -> str:
    import torch

    if requested in {"auto", ""}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _openvino_device(requested: str) -> str:
    if requested in {"auto", "", "cpu"}:
        return "CPU"
    return requested.upper()


@contextlib.contextmanager
def _seeded_noise(seed: int | None) -> Iterator[None]:
    """Draw GR00T's flow-matching noise from a seeded CPU generator.

    Native and OpenVINO inference each draw one ``torch.randn`` tensor for the
    initial action trajectory, but on different devices and dtypes. Generating
    it on the CPU in float32 from the same seed gives both backends identical
    noise, which makes their actions directly comparable.

    Yields:
        Control while ``torch.randn`` uses the seeded generator.
    """
    if seed is None:
        yield
        return
    import torch

    native_randn = torch.randn
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def randn(*size: Any, **kwargs: Any) -> Any:  # ruff: ignore[any-type]
        shape = kwargs.pop("size", None)
        if shape is None:
            shape = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else size
        noise = native_randn(*tuple(shape), generator=generator, dtype=torch.float32)
        return noise.to(device=kwargs.get("device") or "cpu", dtype=kwargs.get("dtype") or torch.float32)

    torch.randn = randn
    try:
        yield
    finally:
        torch.randn = native_randn


def main() -> int:
    """Serve requests until the host closes the request pipe.

    Returns:
        Process exit status.
    """
    requests = os.fdopen(int(os.environ[_REQUEST_FD_ENV]), "rb")
    responses = os.fdopen(int(os.environ[_RESPONSE_FD_ENV]), "wb")
    server = InferenceServer()
    while True:
        message = protocol.read_frame(requests)
        if message is None or message.get("op") == "close":
            break
        try:
            response = server.handle(message)
        except Exception as exc:  # ruff: ignore[blind-except]  # Return failures to the host instead of dying.
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
        protocol.write_frame(responses, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
