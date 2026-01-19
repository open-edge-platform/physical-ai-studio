# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action trainer policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import lerobot
from .act import ACT, ACTConfig, ACTModel
from .dummy import Dummy, DummyConfig
from .groot import Groot, GrootConfig, GrootModel
from .lerobot import get_lerobot_policy
from .pi0 import Pi0, Pi0Config, Pi0Model
from .smolvla import SmolVLA, SmolVLAConfig, SmolVLAModel

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
    # Groot
    "Groot",
    "GrootConfig",
    "GrootModel",
    # Pi0
    "Pi0",
    "Pi0Config",
    "Pi0Model",
    # SmolVLA
    "SmolVLA",
    "SmolVLAConfig",
    "SmolVLAModel",
    # Utils
    "get_getiaction_policy_class",
    "get_policy",
    "lerobot",
]


def get_policy(policy_name: str, *, source: str = "getiaction", **kwargs) -> Policy:  # noqa: ANN003
    """Factory function to create policy instances by name.

    This is a convenience function for dynamically creating policies based on a string name.
    Useful for parameterized tests, CLI tools, or configuration-driven policy selection.

    Args:
        policy_name: Name of the policy to create. Supported values depend on source:
            - getiaction: "act", "dummy", "groot", "pi0", "pi05"
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

    # First-party policies
    if source == "getiaction":
        return get_getiaction_policy_class(policy_name)(**kwargs)

    if source == "lerobot":
        # LeRobot policies via wrapper
        return get_lerobot_policy(policy_name, **kwargs)

    msg = f"Unknown source: {source}. Supported sources: getiaction, lerobot"
    raise ValueError(msg)


def get_getiaction_policy_class(policy_name: str) -> type[Policy]:
    """Get policy class by name.

    Args:
        policy_name: Name of the policy class to retrieve.

    Returns:
        Policy class corresponding to the given name.

    Raises:
        ValueError: If the policy name is unknown.
    """
    policy_name = policy_name.lower()

    if policy_name == "act":
        return ACT
    if policy_name == "dummy":
        return Dummy
    if policy_name == "groot":
        return Groot
    if policy_name == "pi0":
        from functools import partial  # noqa: PLC0415

        return partial(Pi0, variant="pi0")  # type: ignore[return-value]
    if policy_name == "pi05":
        from functools import partial  # noqa: PLC0415

        return partial(Pi0, variant="pi05")  # type: ignore[return-value]
    if policy_name == "smolvla":
        return SmolVLA

    msg = f"Unknown getiaction policy: {policy_name}. Supported policies: act, dummy, groot, pi0, pi05"
    raise ValueError(msg)
