# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Rollout evaluation module for policy evaluation in gym environments.

This module provides both functional and metric-based interfaces for evaluating
policies in gym environments with multi-GPU support.
"""

from .functional import evaluate_policy, rollout
from .metric import Rollout

__all__ = ["Rollout", "evaluate_policy", "rollout"]
