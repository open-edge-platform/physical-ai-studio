# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning module for Patch Policy."""

from torch import Tensor

from physicalai.data import Feature, Observation
from physicalai.export.mixin_policy import ExportablePolicyMixin
from physicalai.policies.base import Policy

from .model import PatchPolicyModel
from .processors import PatchPolicyPostprocessor, PatchPolicyPreprocessor, make_policy_processors


class PatchPolicy(ExportablePolicyMixin, Policy):
    """Patch Policy class."""

    def __init__(
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        *,
        n_action_steps: int = 50,
        chunk_size: int = 50,
        n_obs_steps: int = 1,
    ) -> None:
        """Initialize Patch Policy."""
        super().__init__(n_action_steps=n_action_steps)

        # config args
        self._input_features = input_features
        self._output_features = output_features
        self._n_action_steps = n_action_steps
        self._chunk_size = chunk_size
        self._n_obs_steps = n_obs_steps

        # processors
        self._preprocessor: PatchPolicyPreprocessor | None = None
        self._postprocessor: PatchPolicyPostprocessor | None = None

        # model
        self.model: PatchPolicyModel | None = None

        # save hyperparameters and initialize model
        self.save_hyperparameters()

        # eager init
        if self._input_features is not None and self._output_features is not None:
            self.configure_model()

    def configure_model(self) -> None:
        """Initialize the model."""
        # init model
        self.model = PatchPolicyModel(
            input_features=self._input_features,
            output_features=self._output_features,
            n_action_steps=self._n_action_steps,
            chunk_size=self._chunk_size,
            n_obs_steps=self._n_obs_steps,
        )

        # init pre and post processors here
        self._preprocessor, self._postprocessor = make_policy_processors(self.model.config)

    # Random implemenations
    def predict_action_chunk(self, batch: Observation) -> Tensor:
        actions = self.model.predict_action_chunk(self._preprocessor(batch))
        return self._postprocessor(actions)

    def forward(self, batch: Observation) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        if self.training:
            return self.model(self._preprocessor(batch))
        return self.predict_action_chunk(batch)
