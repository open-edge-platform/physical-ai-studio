# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot policies integration.

This module provides integration with LeRobot's state-of-the-art robot learning policies,
offering two flexible approaches for incorporating pre-trained models into your workflows.

Approaches:
    1. **Explicit Wrappers** (Recommended for most users):
        - Full parameter definitions with IDE autocomplete
        - Type-safe with compile-time checking
        - Direct YAML configuration support
        - Currently available: ACT, Diffusion

    2. **Universal Wrapper** (Flexible for advanced users):
        - Single wrapper for all LeRobot policies
        - Runtime policy selection
        - Minimal code overhead
        - Supports: act, diffusion, vqbet, tdmpc, sac, pi0, pi05, pi0fast, smolvla

Note:
    LeRobot must be installed to use these policies:
        ``pip install lerobot``

    Or install getiaction with LeRobot support:
        ``pip install getiaction[lerobot]``

    For more information, see: https://github.com/huggingface/lerobot

Examples:
    Loading pretrained models from HuggingFace Hub:

        >>> from getiaction.policies.lerobot import ACT, Diffusion

        >>> # Load pretrained ACT model
        >>> act_policy = ACT.from_pretrained(
        ...     "lerobot/act_aloha_sim_transfer_cube_human"
        ... )

        >>> # Load pretrained Diffusion model
        >>> diffusion_policy = Diffusion.from_pretrained(
        ...     "lerobot/diffusion_pusht"
        ... )

    Using the explicit ACT wrapper with full type safety and autocomplete:

        >>> from getiaction.policies.lerobot import ACT

        >>> # Create ACT policy with explicit parameters
        >>> policy = ACT(
        ...     dim_model=512,
        ...     chunk_size=10,
        ...     n_action_steps=10,
        ...     learning_rate=1e-5,
        ... )

    Using the explicit Diffusion wrapper:

        >>> from getiaction.policies.lerobot import Diffusion

        >>> # Create Diffusion policy with explicit parameters
        >>> policy = Diffusion(
        ...     n_obs_steps=2,
        ...     horizon=16,
        ...     n_action_steps=8,
        ...     learning_rate=1e-4,
        ... )

    Using the universal wrapper for runtime policy selection:

        >>> from getiaction.policies.lerobot import LeRobotPolicy

        >>> # Create any LeRobot policy dynamically by name
        >>> policy = LeRobotPolicy(
        ...     policy_name="vqbet",
        ...     learning_rate=1e-4,
        ... )

    Using convenience aliases for cleaner code:

        >>> from getiaction.policies.lerobot import VQBeT, TDMPC

        >>> # Convenience aliases wrap LeRobotPolicy with specific policy names
        >>> vqbet_policy = VQBeT(learning_rate=1e-4)
        >>> tdmpc_policy = TDMPC(learning_rate=1e-4)

    Checking availability before using LeRobot policies:

        >>> from getiaction.policies import lerobot

        >>> if lerobot.is_available():
        ...     policies = lerobot.list_available_policies()
        ...     print(f"Available policies: {policies}")
        ...     policy = lerobot.ACT(dim_model=512, chunk_size=10)
        ... else:
        ...     print("LeRobot not installed. Install with: pip install lerobot")
"""

from lightning_utilities.core.imports import module_available

from getiaction.policies.lerobot.act import ACT
from getiaction.policies.lerobot.diffusion import Diffusion
from getiaction.policies.lerobot.groot import Groot
from getiaction.policies.lerobot.universal import LeRobotPolicy

LEROBOT_AVAILABLE = module_available("lerobot")


# Convenience wrapper classes for universal policies
class VQBeT(LeRobotPolicy):
    """VQ-BeT Policy via universal wrapper.

    This is a convenience class that wraps LeRobotPolicy with policy_name="vqbet".
    """

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Initialize VQ-BeT policy."""
        super().__init__(policy_name="vqbet", **kwargs)


class TDMPC(LeRobotPolicy):
    """TD-MPC Policy via universal wrapper.

    This is a convenience class that wraps LeRobotPolicy with policy_name="tdmpc".
    """

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Initialize TD-MPC policy."""
        super().__init__(policy_name="tdmpc", **kwargs)


class SAC(LeRobotPolicy):
    """SAC Policy via universal wrapper.

    This is a convenience class that wraps LeRobotPolicy with policy_name="sac".
    """

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Initialize SAC policy."""
        super().__init__(policy_name="sac", **kwargs)


class PI0(LeRobotPolicy):
    """PI0 Policy via universal wrapper.

    This is a convenience class that wraps LeRobotPolicy with policy_name="pi0".
    """

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Initialize PI0 policy."""
        super().__init__(policy_name="pi0", **kwargs)


class PI05(LeRobotPolicy):
    """PI0.5 Policy via universal wrapper.

    This is a convenience class that wraps LeRobotPolicy with policy_name="pi05".
    """

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Initialize PI0.5 policy."""
        super().__init__(policy_name="pi05", **kwargs)


class PI0Fast(LeRobotPolicy):
    """PI0Fast Policy via universal wrapper.

    This is a convenience class that wraps LeRobotPolicy with policy_name="pi0fast".
    """

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Initialize PI0Fast policy."""
        super().__init__(policy_name="pi0fast", **kwargs)


class SmolVLA(LeRobotPolicy):
    """SmolVLA Policy via universal wrapper.

    This is a convenience class that wraps LeRobotPolicy with policy_name="smolvla".
    """

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Initialize SmolVLA policy."""
        super().__init__(policy_name="smolvla", **kwargs)


__all__ = [
    "ACT",
    "PI0",
    "PI05",
    "SAC",
    "TDMPC",
    "Diffusion",
    "Groot",
    "LeRobotPolicy",
    "PI0Fast",
    "SmolVLA",
    "VQBeT",
]


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
            "Diffusion",
            # Universal wrapper (all LeRobot policies)
            "groot",
            "pi0",
            "pi05",
            "pi0fast",
            "sac",
            "smolvla",
            "tdmpc",
            "vqbet",
        ]
    return []
