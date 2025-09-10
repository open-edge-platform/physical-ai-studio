# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from action_trainer.policies.act.torch_model import ACTModel
from action_trainer.policies.base.base_lightning_module import ActionTrainerModule


class ACTPolicy(ActionTrainerModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = ACTModel()
