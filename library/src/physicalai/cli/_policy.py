# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Policy loading helpers shared by studio CLI subcommands."""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_policy(policy_path: str, ckpt_path: str | None) -> tuple[Any, str]:
    """Load a policy from a class path and optional checkpoint.

    Args:
        policy_path: Fully qualified policy class path.
        ckpt_path: Path to checkpoint file or export directory.

    Returns:
        Tuple of instantiated policy and selected device string.

    Raises:
        ImportError: If the policy class cannot be imported.
        ValueError: If ``InferenceModel`` is used without ``ckpt_path``.
    """
    from physicalai.devices import get_available_device  # noqa: PLC0415
    from physicalai.inference import InferenceModel  # noqa: PLC0415

    module_path, class_name = policy_path.rsplit(".", 1)
    try:
        policy_class = getattr(importlib.import_module(module_path), class_name)
    except (ImportError, AttributeError) as exc:
        msg = f"Could not import policy class '{policy_path}'"
        raise ImportError(msg) from exc

    is_inference_model = policy_class is InferenceModel or (
        isinstance(policy_class, type) and issubclass(policy_class, InferenceModel)
    )

    if is_inference_model:
        if not ckpt_path:
            msg = "InferenceModel requires --ckpt_path pointing to export directory"
            raise ValueError(msg)
        policy = InferenceModel.load(ckpt_path)
        return policy, policy.device

    device = get_available_device()
    if ckpt_path:
        policy = policy_class.load_from_checkpoint(ckpt_path)
    else:
        logger.warning("No checkpoint provided - using randomly initialized policy")
        policy = policy_class()

    policy.to(device)
    policy.eval()
    return policy, device
