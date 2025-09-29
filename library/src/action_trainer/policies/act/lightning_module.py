# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ACT policy implementation via Lightning module."""

from action_trainer.policies.act.torch_model import ACTModel
from action_trainer.policies.base.base_lightning_module import TrainerModule


class ACTPolicy(TrainerModule):
    """ACT policy wrapper."""

    def __init__(self) -> None:
        """Initialize the ACT policy wrapper."""
        super().__init__()
        self.model = ACTModel()
