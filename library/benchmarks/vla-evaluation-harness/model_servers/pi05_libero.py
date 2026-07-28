# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Pi0.5 model server for LIBERO — Python API entry point.

Subclasses PhysicalAIHarness with Pi0.5 + LIBERO defaults baked in and
constructs the policy directly (no jsonargparse policy_config file). Run
standalone in the active Python environment::

    python model_servers/pi05_libero.py
    python model_servers/pi05_libero.py --port 8000
    python model_servers/pi05_libero.py \
        --config configs/pi05_libero_direct.yaml
    python model_servers/pi05_libero.py --port 8000 \
        --pretrained_name_or_path lerobot/pi05_libero_finetuned_v044 --device cuda

or import ``Pi05LiberoServer`` directly for the Python API path (tests,
notebooks, custom scripts) without spinning up a WebSocket server at all.

NOTE: this constructs Pi05 via its own __init__ (``Pi05(pretrained_name_or_path=...)``),
not ``load_from_checkpoint``. Confirm Pi05's __init__ actually resolves a HF
Hub id / local directory into trained weights — if your checkpoint is a raw
Lightning ``.ckpt`` instead, this will silently build Pi05 with default weights.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

try:
    from model_servers.physicalai_harness import PhysicalAIHarness
except ModuleNotFoundError:
    # When the file is run directly (python model_servers/pi05_libero.py)
    # only the script's directory is on sys.path, not the parent package.
    from physicalai_harness import PhysicalAIHarness

if TYPE_CHECKING:
    from vla_eval.model_servers.predict import PredictModelServer

logger = logging.getLogger(__name__)

# pi05_libero_finetuned declares two image features: image (base/agentview),
# image2 (wrist). chunk_size=10 matches the LeRobot/OpenPI LIBERO protocol.
_LIBERO_IMAGE_KEYS = {"agentview": "image", "wrist": "image2"}
_LIBERO_STATE_KEY = "observation.state"
_LIBERO_CHUNK_SIZE = 10
_DEFAULT_CHECKPOINT = "lerobot/pi05_libero_finetuned_v044"


class Pi05LiberoServer(PhysicalAIHarness):
    """Pi0.5 model server pre-configured for LIBERO, no policy_config needed."""

    def __init__(
        self,
        pretrained_name_or_path: str = _DEFAULT_CHECKPOINT,
        device: str = "cuda",
        *,
        image_keys: dict[str, str] | None = None,
        state_key: str | None = _LIBERO_STATE_KEY,
        chunk_size: int | None = _LIBERO_CHUNK_SIZE,
        action_ensemble: str = "newest",
        **vla_eval_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the Pi0.5 LIBERO server.

        Args:
            pretrained_name_or_path: HF Hub id or local path for Pi0.5 weights.
            device: Torch device the policy is moved to after construction.
            image_keys: Override the LIBERO agentview/wrist mapping if needed.
            state_key: Override the state feature name if needed.
            chunk_size: Override the LIBERO action chunk size if needed.
            action_ensemble: How to combine multiple action predictions.
            **vla_eval_kwargs: Extra kwargs forwarded to PredictModelServer
                (e.g. port/host are consumed by run_server, not here).
        """
        from physicalai.policies.pi05 import Pi05  # noqa: PLC0415

        logger.info("Loading Pi0.5 from: %s", pretrained_name_or_path)
        policy = Pi05(pretrained_name_or_path=pretrained_name_or_path)

        super().__init__(
            _policy=policy,
            image_keys=image_keys or dict(_LIBERO_IMAGE_KEYS),
            state_key=state_key,
            device=device,
            chunk_size=chunk_size,
            action_ensemble=action_ensemble,
            **vla_eval_kwargs,
        )
        self.pretrained_name_or_path = pretrained_name_or_path


def _run_current_env(server_cls: type[PredictModelServer]) -> None:
    """Run ``run_server`` after rewriting the YAML for current-env usage.

    ``vla-eval serve`` pops ``port`` / ``host`` out of the YAML's ``args:``
    block and strips the ``script:`` metadata key before invoking the server.
    ``run_server``'s own ``--config`` loader keeps them nested under ``args:``,
    which would pass ``port`` / ``host`` to the model-server ``__init__`` and
    fail.  This helper mirrors the ``vla-eval serve`` normalisation so the same
    config files work when the script is run directly with ``python``.

    The original config path is preserved for relative-path resolution by
    writing the sanitised copy next to it.
    """
    from vla_eval.model_servers.serve import run_server  # noqa: PLC0415

    pre_parser = argparse.ArgumentParser()
    pre_parser.add_argument("--config", required=True)
    pre_parser.add_argument("--port", type=int, default=None)
    pre_parser.add_argument("--host", default=None)
    known, remainder = pre_parser.parse_known_args()

    config_path = Path(known.config)
    with config_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # ``script`` was metadata for ``vla-eval serve``; drop it.
    raw.pop("script", None)

    args_block: dict[str, Any] = raw.setdefault("args", {})
    server_level: dict[str, Any] = {}
    for key in ("port", "host"):
        if key in args_block:
            server_level[key] = args_block.pop(key)

    # CLI flags take precedence over YAML values.
    if known.port is not None:
        server_level["port"] = known.port
    if known.host is not None:
        server_level["host"] = known.host

    # Keep the sanitised file next to the original so relative paths inside
    # the YAML resolve the same way ``vla-eval serve`` would have resolved them.
    config_dir = config_path.parent
    fd, temp_path_str = tempfile.mkstemp(
        suffix=".yaml",
        dir=config_dir,
        prefix=f"._sanitized_{server_cls.__name__.lower()}_",
    )
    temp_path = Path(temp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f)
    except Exception:
        os.close(fd)
        raise

    def _cleanup() -> None:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()

    atexit.register(_cleanup)

    new_argv = [sys.argv[0], "--config", str(temp_path)]
    new_argv.extend(f"--{key}={server_level[key]}" for key in ("port", "host") if key in server_level)
    new_argv.extend(remainder)
    sys.argv = new_argv

    run_server(server_cls)


if __name__ == "__main__":
    import sys

    # Allow either ``python model_servers/pi05_libero.py --config <yaml>``
    # (Mode 3) or ``python model_servers/pi05_libero.py --port 8000 ...``
    # (no YAML).  When no --config is given we delegate straight to
    # run_server, which builds the CLI from Pi05LiberoServer.__init__.
    if any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        _run_current_env(Pi05LiberoServer)
    else:
        from vla_eval.model_servers.serve import run_server

        run_server(Pi05LiberoServer)
