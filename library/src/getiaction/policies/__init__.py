# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action trainer policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import lerobot
from .act import ACT, ACTConfig, ACTModel
from .dummy import Dummy, DummyConfig
from .lerobot import get_lerobot_policy
from .pi0 import Pi0, Pi0Config, Pi0Model

if TYPE_CHECKING:
    from .base import Policy

__all__ = [
    # ACT
    "ACT",
    "ACTConfig",
    "ACTModel",
    # Dummy
    "Dummy",
    "DummyConfig",
    # Pi0
    "Pi0",
    "Pi0Config",
    "Pi0Model",
    "get_policy",
    "lerobot",
]


def get_policy(policy_name: str, *, source: str = "getiaction", **kwargs) -> Policy:  # noqa: ANN003
    """Factory function to create policy instances by name.

    This is a convenience function for dynamically creating policies based on a string name.
    Useful for parameterized tests, CLI tools, or configuration-driven policy selection.

    Args:
        policy_name: Name of the policy to create. Supported values depend on source:
            - getiaction: "act", "dummy", "pi0", "pi05"
            - lerobot: "act", "diffusion", "vqbet", "tdmpc", "sac", "pi0", etc.
        source: Where the policy implementation comes from. Options:
            - "getiaction": First-party implementations (default)
            - "lerobot": LeRobot framework wrappers
        **kwargs: Additional keyword arguments passed to the policy constructor.

    Returns:
        Policy: Instance of the requested policy.

    Raises:
        ValueError: If the policy name or source is unknown.

    Examples:
        Create first-party ACT policy (default source):

            >>> from getiaction.policies import get_policy
            >>> policy = get_policy("act", learning_rate=1e-4)

        Create first-party Pi0 policy:

            >>> policy = get_policy("pi0", learning_rate=2.5e-5)

        Create Pi0.5 variant:

            >>> policy = get_policy("pi05", learning_rate=2.5e-5)

        Create LeRobot ACT policy explicitly:

            >>> policy = get_policy("act", source="lerobot", learning_rate=1e-4)

        Create LeRobot-only policy (Diffusion):

            >>> policy = get_policy("diffusion", source="lerobot", learning_rate=1e-4)

        Use in parameterized tests:

            >>> @pytest.mark.parametrize(
            ...     ("policy_name", "source"),
            ...     [("act", "getiaction"), ("pi0", "getiaction"), ("diffusion", "lerobot")],
            ... )
            >>> def test_policy(policy_name, source):
            ...     policy = get_policy(policy_name, source=source)
            ...     assert policy is not None

        Dynamic source selection:

            >>> use_lerobot = True
            >>> policy = get_policy("act", source="lerobot" if use_lerobot else "getiaction")
    """
    source = source.lower()

    if source == "getiaction":
        # First-party policies
        policy_name_lower = policy_name.lower()
        if policy_name_lower == "act":
            return ACT(**kwargs)
        if policy_name_lower == "pi0":
            return Pi0(variant="pi0", **kwargs)
        if policy_name_lower == "pi05":
            return Pi0(variant="pi05", **kwargs)

        msg = f"Unknown getiaction policy: {policy_name}. Supported policies: act, pi0, pi05"
        raise ValueError(msg)

    if source == "lerobot":
        # LeRobot policies via wrapper
        return get_lerobot_policy(policy_name, **kwargs)

    msg = f"Unknown source: {source}. Supported sources: getiaction, lerobot"
    raise ValueError(msg)
