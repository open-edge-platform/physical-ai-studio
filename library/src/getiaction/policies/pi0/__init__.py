# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pi0/Pi0.5 Policy - Physical Intelligence's flow matching VLA model.

This module provides first-party PyTorch implementations of Pi0 and Pi0.5
vision-language-action models for robot control.

Pi0 uses flow matching for action generation with a PaliGemma backbone
and a smaller Gemma action expert. Pi0.5 extends this with discrete state
input and adaRMSNorm for improved timestep conditioning.

Example:
    >>> from getiaction.policies.pi0 import Pi0, Pi0Config

    >>> # Create Pi0 policy
    >>> policy = Pi0(chunk_size=50, learning_rate=2.5e-5)

    >>> # Or Pi0.5 variant
    >>> policy = Pi0(variant="pi05", chunk_size=50)

    >>> # Train with Lightning
    >>> trainer = L.Trainer(max_epochs=100)
    >>> trainer.fit(policy, datamodule)
"""

from .config import Pi0Config
from .model import Pi0Model
from .policy import Pi0, Pi05

__all__ = ["Pi0", "Pi0Config", "Pi0Model", "Pi05"]
