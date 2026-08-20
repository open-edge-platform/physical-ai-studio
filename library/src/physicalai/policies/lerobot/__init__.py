# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot policy wrappers.

This package exposes thin :class:`NamedLeRobotPolicy` wrappers for curated
LeRobot policies, plus :class:`LeRobotPolicy` as a dynamic escape hatch for
any LeRobot-registered policy name.

Installation:

    Install LeRobot support with::

        pip install physicalai-train[lerobot]

Examples:
    Load a named wrapper from the HuggingFace Hub::

        >>> from physicalai.policies.lerobot import ACT
        >>> policy = ACT.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human")

    Use a recent VLA wrapper directly::

        >>> from physicalai.policies.lerobot import PI05
        >>> policy = PI05(dtype="bfloat16")

    Construct a wrapper from dataset metadata::

        >>> from physicalai.policies.lerobot import Diffusion
        >>> policy = Diffusion.from_dataset("lerobot/pusht", optimizer_lr=1e-4)

    Dispatch dynamically by policy name::

        >>> from physicalai.policies.lerobot import get_lerobot_policy
        >>> policy = get_lerobot_policy("act", optimizer_lr=1e-4)
"""

from typing import Any

from lightning_utilities.core.imports import module_available

from physicalai.policies.lerobot.aliases import (
    ACT,
    PI0,
    PI05,
    XVLA,
    Diffusion,
    Groot,
    PI0Fast,
    SmolVLA,
)
from physicalai.policies.lerobot.policy import LeRobotPolicy, NamedLeRobotPolicy
from physicalai.policies.lerobot.utils.checkpoint_converter import lerobot_to_lightning, lightning_to_lerobot

LEROBOT_AVAILABLE = module_available("lerobot")

SUPPORTED_POLICIES: tuple[str, ...] = (
    "act",
    "diffusion",
    "groot",
    "pi0",
    "pi05",
    "pi0_fast",
    "smolvla",
    "xvla",
)
"""First-class LeRobot policies exposed as named wrappers."""

VALIDATED_EQUIVALENCE_POLICIES: tuple[str, ...] = (
    "act",
    "diffusion",
    "smolvla",
)
"""Named wrappers covered by the generic wrapper-vs-native equivalence suite."""

_NAMED_WRAPPERS: tuple[type[NamedLeRobotPolicy], ...] = (
    ACT,
    Diffusion,
    Groot,
    PI0,
    PI05,
    PI0Fast,
    SmolVLA,
    XVLA,
)

_POLICY_MAP: dict[str, type[NamedLeRobotPolicy]] = {cls.POLICY_NAME: cls for cls in _NAMED_WRAPPERS}


__all__ = [
    "ACT",
    "PI0",
    "PI05",
    "SUPPORTED_POLICIES",
    "VALIDATED_EQUIVALENCE_POLICIES",
    "XVLA",
    "Diffusion",
    "Groot",
    "LeRobotPolicy",
    "NamedLeRobotPolicy",
    "PI0Fast",
    "SmolVLA",
    "get_lerobot_policy",
    "is_available",
    "lerobot_to_lightning",
    "lightning_to_lerobot",
    "list_available_policies",
]


def get_lerobot_policy(policy_name: str, **kwargs: Any) -> LeRobotPolicy:  # noqa: ANN401
    """Instantiate a LeRobot policy by name.

    Names in :data:`SUPPORTED_POLICIES` dispatch to the matching named
    wrapper. Other names fall through to the dynamic :class:`LeRobotPolicy`
    (best-effort, triggers a one-time :class:`UserWarning`). Validation of
    unknown names is deferred to LeRobot's own ``PreTrainedConfig`` registry
    and only fires when the policy is eagerly constructed (e.g. via
    :meth:`LeRobotPolicy.from_dataset` or a Lightning ``setup`` hook).

    Args:
        policy_name: Either a name in :data:`SUPPORTED_POLICIES` or any
            other entry in LeRobot's ``PreTrainedConfig.get_known_choices()``.
        **kwargs: Forwarded to the wrapper constructor (see
            :class:`NamedLeRobotPolicy` / :class:`LeRobotPolicy` for the
            shared signature).

    Returns:
        A configured policy wrapper instance.

    Raises:
        ImportError: If LeRobot is not installed.

    Examples:
        >>> from physicalai.policies.lerobot import get_lerobot_policy
        >>> policy = get_lerobot_policy("act", dim_model=512, chunk_size=10)
    """
    if not LEROBOT_AVAILABLE:
        msg = (
            "LeRobot is not installed. Please install it with:\n"
            "  pip install lerobot\n"
            "or install physicalai with LeRobot support:\n"
            "  pip install physicalai-train[lerobot]"
        )
        raise ImportError(msg)

    cls = _POLICY_MAP.get(policy_name.lower())
    if cls is not None:
        return cls(**kwargs)
    return LeRobotPolicy(policy_name=policy_name, **kwargs)


def is_available() -> bool:
    """Return ``True`` if LeRobot is importable in the current environment."""
    return LEROBOT_AVAILABLE


def list_available_policies() -> list[str]:
    """List first-class policy names with named wrappers and equivalence scope.

    Returns:
        ``list(SUPPORTED_POLICIES)`` if LeRobot is installed, else ``[]``.
        Note this excludes the best-effort escape-hatch policies still
        reachable through :class:`LeRobotPolicy` directly.
    """
    if LEROBOT_AVAILABLE:
        return list(SUPPORTED_POLICIES)
    return []
