# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot policies integration.

Provides two approaches for using LeRobot's state-of-the-art policies:

1. **Explicit Wrappers** (Recommended for most users):
   - Full parameter definitions with IDE autocomplete
   - Type-safe with compile-time checking
   - Direct YAML configuration support
   - Currently available: ACT

2. **Universal Wrapper** (Flexible for advanced users):
   - Single wrapper for all LeRobot policies
   - Runtime policy selection
   - Minimal code overhead
   - Supports: act, diffusion, vqbet, tdmpc, sac, pi0, pi05, pi0fast, smolvla

Example - Explicit Wrapper:
    >>> from getiaction.policies.lerobot import ACT
    >>> policy = ACT(
    ...     dim_model=512,
    ...     chunk_size=100,
    ...     stats=dataset.meta.stats,
    ... )

Example - Universal Wrapper:
    >>> from getiaction.policies.lerobot import LeRobotPolicy
    >>> policy = LeRobotPolicy(
    ...     policy_name="diffusion",
    ...     input_features=features,
    ...     output_features=features,
    ...     num_steps=100,
    ...     stats=dataset.meta.stats,
    ... )

Example - Convenience Aliases:
    >>> from getiaction.policies.lerobot import Diffusion
    >>> policy = Diffusion(
    ...     input_features=features,
    ...     output_features=features,
    ...     num_steps=100,
    ...     stats=dataset.meta.stats,
    ... )

Install LeRobot to use these policies:
    pip install lerobot

Or install getiaction with LeRobot support:
    pip install getiaction[lerobot]

For more information, see: https://github.com/huggingface/lerobot
"""

from __future__ import annotations

try:
    from lerobot.policies.act.modeling_act import ACTPolicy as _LeRobotACTPolicy

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    _LeRobotACTPolicy = None

if LEROBOT_AVAILABLE:
    # Import actual implementations
    # Explicit wrappers (full parameter definitions, IDE autocomplete)
    from getiaction.policies.lerobot.act import ACT

    # Universal wrapper (flexible, all policies)
    from getiaction.policies.lerobot.universal import LeRobotPolicy

    # Convenience aliases for universal wrapper
    # These provide clean names while using the universal wrapper underneath
    def Diffusion(**kwargs):  # noqa: N802
        """Diffusion Policy via universal wrapper."""
        return LeRobotPolicy(policy_name="diffusion", **kwargs)

    def VQBeT(**kwargs):  # noqa: N802
        """VQ-BeT Policy via universal wrapper."""
        return LeRobotPolicy(policy_name="vqbet", **kwargs)

    def TDMPC(**kwargs):  # noqa: N802
        """TD-MPC Policy via universal wrapper."""
        return LeRobotPolicy(policy_name="tdmpc", **kwargs)

    def SAC(**kwargs):  # noqa: N802
        """SAC Policy via universal wrapper."""
        return LeRobotPolicy(policy_name="sac", **kwargs)

    def PI0(**kwargs):  # noqa: N802
        """PI0 Policy via universal wrapper."""
        return LeRobotPolicy(policy_name="pi0", **kwargs)

    def PI05(**kwargs):  # noqa: N802
        """PI0.5 Policy via universal wrapper."""
        return LeRobotPolicy(policy_name="pi05", **kwargs)

    def PI0Fast(**kwargs):  # noqa: N802
        """PI0Fast Policy via universal wrapper."""
        return LeRobotPolicy(policy_name="pi0fast", **kwargs)

    def SmolVLA(**kwargs):  # noqa: N802
        """SmolVLA Policy via universal wrapper."""
        return LeRobotPolicy(policy_name="smolvla", **kwargs)

    __all__ = [
        # Explicit wrappers
        "ACT",
        # Universal wrapper
        "LeRobotPolicy",
        # Convenience aliases
        "Diffusion",
        "VQBeT",
        "TDMPC",
        "SAC",
        "PI0",
        "PI05",
        "PI0Fast",
        "SmolVLA",
    ]
else:
    # Provide helpful error messages when LeRobot is not installed
    def _make_unavailable_class(name: str):
        """Create a class that raises helpful error on instantiation."""

        def __init__(self, *args, **kwargs):  # noqa: ARG001
            msg = (
                f"{name} requires LeRobot framework.\n\n"
                f"Install with:\n"
                f"    pip install lerobot\n\n"
                f"Or install getiaction with LeRobot support:\n"
                f"    pip install getiaction[lerobot]\n\n"
                f"For more information, see: https://github.com/huggingface/lerobot"
            )
            raise ImportError(msg)

        return type(name, (), {"__init__": __init__})

    ACT = _make_unavailable_class("ACT")

    __all__ = []


def is_available() -> bool:
    """Check if LeRobot is available.

    Returns:
        bool: True if LeRobot is installed and available, False otherwise.

    Examples:
        >>> from getiaction.policies import lerobot
        >>> if lerobot.is_available():
        ...     from getiaction.policies.lerobot import ACT
        ...     policy = ACT(hidden_dim=512)
        ... else:
        ...     print("LeRobot not available, using native policy")
    """
    return LEROBOT_AVAILABLE


def list_available_policies() -> list[str]:
    """List available LeRobot policies.

    Returns:
        list[str]: List of available policy names. Empty if LeRobot is not installed.

    Examples:
        >>> from getiaction.policies import lerobot
        >>> policies = lerobot.list_available_policies()
        >>> print(f"Available policies: {policies}")
    """
    if LEROBOT_AVAILABLE:
        return [
            # Explicit wrappers
            "ACT",
            # Universal wrapper (all LeRobot policies)
            "diffusion",
            "vqbet",
            "tdmpc",
            "sac",
            "pi0",
            "pi05",
            "pi0fast",
            "smolvla",
        ]
    return []
