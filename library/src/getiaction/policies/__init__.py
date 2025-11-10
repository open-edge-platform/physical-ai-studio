# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action trainer policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import lerobot
from .act import ACT, ACTConfig, ACTModel
from .dummy import Dummy, DummyConfig

if TYPE_CHECKING:
    from .base import Policy

__all__ = ["ACT", "ACTConfig", "ACTModel", "Dummy", "DummyConfig", "get_policy", "lerobot"]


def get_policy(policy_name: str, *, source: str = "getiaction", **kwargs) -> Policy:  # noqa: ANN003
    """Factory function to create policy instances by name.

    This is a convenience function for dynamically creating policies based on a string name.
    Useful for parameterized tests, CLI tools, or configuration-driven policy selection.

    Args:
        policy_name: Name of the policy to create. Supported values depend on source:
            - getiaction: "act", "dummy"
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

        Create LeRobot ACT policy explicitly:

            >>> policy = get_policy("act", source="lerobot", learning_rate=1e-4)

        Create LeRobot-only policy (Diffusion):

            >>> policy = get_policy("diffusion", source="lerobot", learning_rate=1e-4)

        Use in parameterized tests:

            >>> @pytest.mark.parametrize(
            ...     ("policy_name", "source"),
            ...     [("act", "getiaction"), ("act", "lerobot"), ("diffusion", "lerobot")],
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
        if policy_name == "act":
            return ACT(**kwargs)
        if policy_name == "dummy":
            return Dummy(**kwargs)

        msg = f"Unknown getiaction policy: {policy_name}. Supported policies: act, dummy"
        raise ValueError(msg)

    if source == "lerobot":
        # LeRobot policies
        return lerobot.get_lerobot_policy(policy_name, **kwargs)

    msg = f"Unknown source: {source}. Supported sources: getiaction, lerobot"
    raise ValueError(msg)
