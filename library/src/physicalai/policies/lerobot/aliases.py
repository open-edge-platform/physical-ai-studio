# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Named LeRobot policy aliases.

Each first-class supported LeRobot policy is exposed as a thin
:class:`~physicalai.policies.lerobot.policy.NamedLeRobotPolicy` subclass that
binds a fixed ``POLICY_NAME``. Configuration flows through LeRobot's own
``PreTrainedConfig`` dataclasses (``ACTConfig``, ``DiffusionConfig``, …); the
aliases exist so YAML configs can target a stable class path and so
``isinstance(policy, ACT)`` discriminates between policy families.

The named set is intentionally curated. Other LeRobot-registered policies
(``vqbet``, ``tdmpc``, ``sac``, …) remain reachable through the dynamic
:class:`~physicalai.policies.lerobot.policy.LeRobotPolicy` wrapper as a
best-effort escape hatch (no equivalence guarantee, see
:data:`~physicalai.policies.lerobot.SUPPORTED_POLICIES`).
"""

from __future__ import annotations

from physicalai.policies.lerobot.policy import NamedLeRobotPolicy


class ACT(NamedLeRobotPolicy):
    """LeRobot ACT (Action Chunking Transformer) policy."""

    POLICY_NAME = "act"


class Diffusion(NamedLeRobotPolicy):
    """LeRobot Diffusion Policy."""

    POLICY_NAME = "diffusion"


class SmolVLA(NamedLeRobotPolicy):
    """LeRobot SmolVLA policy."""

    POLICY_NAME = "smolvla"


class PI0(NamedLeRobotPolicy):
    """LeRobot PI0 policy."""

    POLICY_NAME = "pi0"


class PI05(NamedLeRobotPolicy):
    """LeRobot PI0.5 policy."""

    POLICY_NAME = "pi05"


class PI0Fast(NamedLeRobotPolicy):
    """LeRobot PI0Fast policy."""

    POLICY_NAME = "pi0_fast"


class Groot(NamedLeRobotPolicy):
    """LeRobot Groot (GR00T-N1) policy from NVIDIA.

    Known limitations:
        Upstream LeRobot hardcodes ``flash_attention_2`` in the Eagle2
        backbone, which is unavailable on CPU and on GPUs without flash-attn
        installed. Wrapper-vs-native equivalence is therefore registered as
        :mod:`pytest` ``xfail`` in the integration suite. Construction works
        on supported hardware; on others the inner policy raises at first
        forward pass.
    """

    POLICY_NAME = "groot"


class XVLA(NamedLeRobotPolicy):
    """LeRobot X-VLA policy.

    Known limitations:
        Upstream ``XVLAConfig`` requires an explicit ``vision_config`` kwarg
        that cannot be derived from a ``LeRobotDataset`` alone, so
        ``from_dataset`` is not currently exercised in the equivalence
        suite (registered as ``xfail``). Direct construction with a
        user-supplied ``vision_config`` works.
    """

    POLICY_NAME = "xvla"
