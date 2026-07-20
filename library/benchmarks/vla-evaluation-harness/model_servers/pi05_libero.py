# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pi0.5 model server for LIBERO — Python API entry point.

Subclasses PhysicalAIHarness with Pi0.5 + LIBERO defaults baked in and
constructs the policy directly (no jsonargparse policy_config file). Run
standalone:

    python model_servers/pi05_libero.py --port 8000
    python model_servers/pi05_libero.py --port 8000 --pretrained_name_or_path lerobot/pi05_libero_finetuned_v044 --device cuda

or import ``Pi05LiberoServer`` directly for the Python API path (tests,
notebooks, custom scripts) without spinning up a WebSocket server at all.

NOTE: this constructs Pi05 via its own __init__ (`Pi05(pretrained_name_or_path=...)`),
not `load_from_checkpoint`. Confirm Pi05's __init__ actually resolves a HF
Hub id / local directory into trained weights — if your checkpoint is a raw
Lightning `.ckpt` instead, this will silently build Pi05 with default weights.
"""

from __future__ import annotations

import logging
from typing import Any

from model_servers.physicalai_harness import PhysicalAIHarness

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
        """Args:
        pretrained_name_or_path: HF Hub id or local path for Pi0.5 weights.
        device: Torch device the policy is moved to after construction.
        image_keys: Override the LIBERO agentview/wrist mapping if needed.
        state_key: Override the state feature name if needed.
        chunk_size: Override the LIBERO action chunk size if needed.
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


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(Pi05LiberoServer)
