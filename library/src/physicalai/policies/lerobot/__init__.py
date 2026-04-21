# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot policies integration.

Lightning-compatible wrappers around `LeRobot <https://github.com/huggingface/lerobot>`_
policies. Each first-class supported policy is exposed as a thin
:class:`NamedLeRobotPolicy` subclass that binds a single ``POLICY_NAME``;
configuration flows through LeRobot's own ``PreTrainedConfig`` dataclasses
rather than hand-mirrored constructor kwargs.

Two access tiers:

- **Named wrappers** (the eight in :data:`SUPPORTED_POLICIES`) get one-line
  factory classes for ergonomic use. The subset listed in
  :data:`VALIDATED_EQUIVALENCE_POLICIES` carries a hard equivalence guarantee
  enforced by the test suite; the remainder (currently ``groot``, ``xvla``)
  are named for API parity but cannot be validated end-to-end yet (see the
  ``Known limitations`` section in their alias docstrings).
- **Universal escape hatch** — :class:`LeRobotPolicy` accepts any
  ``policy_name`` that LeRobot's ``PreTrainedConfig`` registry knows
  (``vqbet``, ``tdmpc``, ``sac``, …). These work best-effort: a one-time
  :class:`UserWarning` is emitted and no equivalence is asserted. LeRobot's
  own validation runs only when the policy is eagerly constructed (e.g. via
  :meth:`LeRobotPolicy.from_dataset` or Lightning's ``setup`` hook).

Note:
    LeRobot must be installed::

        pip install physicalai-train[lerobot]

    See https://github.com/huggingface/lerobot for upstream documentation and
    the per-policy ``PreTrainedConfig`` dataclasses that define every tunable
    field (e.g. :class:`lerobot.policies.act.configuration_act.ACTConfig`).

Examples:
    Load a pretrained checkpoint from the HuggingFace Hub::

        >>> from physicalai.policies.lerobot import ACT
        >>> policy = ACT.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human")

    Construct from a dataset (features are inferred)::

        >>> from physicalai.policies.lerobot import Diffusion
        >>> policy = Diffusion.from_dataset("lerobot/pusht", optimizer_lr=1e-4)

    Override individual config fields at construction time::

        >>> from physicalai.policies.lerobot import ACT
        >>> policy = ACT(
        ...     input_features=input_features,
        ...     output_features=output_features,
        ...     dim_model=512,
        ...     chunk_size=10,
        ...     optimizer_lr=1e-5,
        ... )

    Dispatch dynamically by policy name (named or escape-hatch)::

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
"""First-class LeRobot policies exposed as named :class:`NamedLeRobotPolicy` subclasses.

Membership implies a wrapper class exists in :mod:`physicalai.policies.lerobot.aliases`
and the policy is in scope for the equivalence test suite. Other policies in
LeRobot's ``PreTrainedConfig`` registry remain reachable through
:class:`LeRobotPolicy` directly with no equivalence guarantee.
"""

VALIDATED_EQUIVALENCE_POLICIES: tuple[str, ...] = (
    "act",
    "diffusion",
    "smolvla",
    "pi0",
    "pi05",
    "pi0_fast",
)
"""Subset of :data:`SUPPORTED_POLICIES` with measured wrapper-vs-native numerical equivalence.

A policy listed here passes the equivalence test suite under tier-appropriate
tolerances:

- Unit tier (CPU, fp32, manual forward/backward): ``rtol=atol=1e-6`` for forward,
  gradient, and post-step weight comparisons. Applies to ``act``, ``diffusion``,
  ``smolvla``.
- Integration tier (Lightning Trainer, CUDA/XPU + bf16-mixed for VLAs): ``rtol=1e-5``
  for loss trajectories and ``rtol=5e-5`` for post-training weights (calibrated
  from observed float32-accumulation drift; see test docstrings). Applies to all
  six entries.

Membership implies a hard guarantee: any wrapper change that breaks equivalence
must show up as a failing test in ``library/tests/unit/policies/test_lerobot.py``
or ``library/tests/integration/test_lerobot_wrapper_equivalence.py``.

Policies in :data:`SUPPORTED_POLICIES` but not here (``groot``, ``xvla``) are
named for API ergonomics but cannot currently be validated end-to-end — see the
``Known limitations`` section in their alias docstrings and the
``_EQUIVALENCE_XFAIL_REASONS`` table in the integration test for details.
"""

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
