# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ACT policy implementation via Lightning module."""

from getiaction.policies.act.torch_model import ACTModel
from getiaction.policies.base.policy import Policy


class ACTPolicy(Policy):
    """ACT policy wrapper."""

    def __init__(self) -> None:
        """Initialize the ACT policy wrapper."""
        super().__init__()
        self.model = ACTModel()
