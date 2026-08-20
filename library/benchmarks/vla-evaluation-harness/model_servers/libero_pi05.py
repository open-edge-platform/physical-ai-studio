# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pi0.5 model server with maintained LIBERO defaults."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __name__ == "__main__" and __package__ is None:
    sys.path.pop(0)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_eval.model_servers.serve import run_server

from model_servers.physicalai import PhysicalAIModelServer
from physicalai.policies.pi05 import Pi05


class LiberoPi05ModelServer(PhysicalAIModelServer):
    """Run a Pi0.5 checkpoint with the LIBERO observation protocol."""

    def __init__(
        self,
        pretrained_name_or_path: str = "lerobot/pi05_libero_finetuned_v044",
        device: str = "cuda",
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Construct the reference Pi0.5 policy and LIBERO mapping."""
        policy = Pi05(pretrained_name_or_path=pretrained_name_or_path)
        super().__init__(
            _policy=policy,
            image_keys={"agentview": "image", "wrist": "image2"},
            state_key="observation.state",
            chunk_size=10,
            device=device,
            **kwargs,
        )


if __name__ == "__main__":
    run_server(LiberoPi05ModelServer)
