# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VLA-Adapter model.

Default trainability: the head, proprio projector, visual projector and
action queries train, while the visual backbone and LLM stay frozen.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

from physicalai.data.constants import TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.data.observation import ACTION, IMAGES, STATE
from physicalai.policies.base.model import Model
from physicalai.policies.vla_adapter.components import VLM, L1RegressionActionHead, ProprioProjector

if TYPE_CHECKING:
    from physicalai.policies.vla_adapter.config import VLAAdapterConfig

logger = logging.getLogger(__name__)


class VLAAdapterModel(Model):
    """Prismatic VLM plus the VLA-Adapter Policy head.

    Mostly frozen: the vision towers and language model do not train, while the
    action head, proprio projector, visual projector and action queries do.

    Args:
        config: Resolved policy configuration.
        dataset_stats: Dataset statistics, retained for downstream consumers
            (normalization itself is applied by the preprocessor).
    """

    def __init__(
        self,
        config: VLAAdapterConfig,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
    ) -> None:
        """Build the VLM, proprio projector and action head.

        Args:
            config: Resolved policy configuration.
            dataset_stats: Retained for downstream consumers; normalization
                itself is applied by the preprocessor.
        """
        super().__init__()
        self._config = config
        self._dataset_stats = dataset_stats

        self.vlm = VLM(config)
        llm_dim = self.vlm.llm_dim

        self.proprio_projector: ProprioProjector | None = None
        if config.use_proprio:
            self.proprio_projector = ProprioProjector(
                llm_dim=llm_dim,
                proprio_dim=config.max_state_dim,
            )

        # The head's split point *is* the number of vision tokens, so derive it
        # from the VLM rather than trusting the config to agree. The config
        # value documents the upstream LIBERO layout (512 = 2 views x 256
        # patches); a mismatch means a different resolution or camera count.
        num_task_tokens = self.vlm.num_vision_tokens
        if num_task_tokens != config.num_task_tokens:
            logger.warning(
                "num_task_tokens=%d from config does not match the backbone's %d vision tokens "
                "(%d views x %d patches); using the backbone value.",
                config.num_task_tokens,
                num_task_tokens,
                config.num_images_in_input,
                self.vlm.vision_backbone.get_num_patches(),
            )

        self.action_head = L1RegressionActionHead(
            input_dim=llm_dim,
            hidden_dim=llm_dim,
            action_dim=config.max_action_dim,
            chunk_size=config.chunk_size,
            num_task_tokens=num_task_tokens,
            num_blocks=self.vlm.num_layers,
            num_heads=config.head_num_heads,
        )

        self.log_trainable_parameters()

    @property
    def config(self) -> VLAAdapterConfig:
        """The configuration this model was built from.

        Returns:
            The resolved policy configuration.
        """
        return self._config

    def set_dataset_stats(self, dataset_stats: dict) -> None:
        """Update dataset statistics.

        Args:
            dataset_stats: Dataset normalization statistics.
        """
        self._dataset_stats = dataset_stats

    def trainable_parameter_summary(self) -> dict[str, tuple[int, int]]:
        """Count trainable and total parameters per component.

        VLM submodules are reported individually, since the freeze split runs
        inside the VLM rather than at its boundary.

        Returns:
            Component name -> ``(trainable, total)``.
        """
        groups: dict[str, nn.Module] = {f"vlm.{key}": getattr(self.vlm, key) for key in self.vlm.all_module_keys}
        groups["action_head"] = self.action_head
        if self.proprio_projector is not None:
            groups["proprio_projector"] = self.proprio_projector

        return {
            name: (
                sum(p.numel() for p in module.parameters() if p.requires_grad),
                sum(p.numel() for p in module.parameters()),
            )
            for name, module in groups.items()
        }

    def log_trainable_parameters(self) -> None:
        """Log the per-component trainable/total split.

        Makes the freeze decision visible at build time, rather than something
        inferred later from a loss that will not move.
        """
        summary = self.trainable_parameter_summary()
        total = sum(t for _, t in summary.values())
        trainable = sum(t for t, _ in summary.values())
        logger.info(
            "VLA-Adapter parameters: %s trainable / %s total (%.2f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / total if total else 0.0,
        )
        for name, (tr, tot) in summary.items():
            logger.info("  %-24s %14s / %14s  %s", name, f"{tr:,}", f"{tot:,}", "trains" if tr else "frozen")

    def _predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the VLM plus head and return the raw action chunk.

        Args:
            batch: Preprocessed batch with channel-stacked images, a tokenized
                prompt and (optionally) state.

        Returns:
            ``(B, chunk_size, action_dim)``.

        Raises:
            KeyError: If a required batch key is missing.
        """
        missing = [k for k in (IMAGES, TOKENIZED_PROMPT) if k not in batch]
        if missing:
            msg = f"Batch is missing required key(s) for VLA-Adapter: {missing}"
            raise KeyError(msg)

        hidden_states = self.vlm(
            pixel_values=batch[IMAGES],
            input_ids=batch[TOKENIZED_PROMPT],
            attention_mask=batch.get(TOKENIZED_PROMPT_MASK),
        )

        proprio_features = None
        if self.proprio_projector is not None:
            state = batch[STATE]
            state = state.reshape(state.shape[0], -1).to(hidden_states.dtype)
            proprio_features = self.proprio_projector(state).unsqueeze(1)

        return self.action_head(hidden_states, proprio_features=proprio_features)

    def _target_actions(self, batch: dict[str, torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        """Slice ground-truth actions to the predicted chunk and dtype.

        Args:
            batch: Preprocessed batch containing ``action``.
            like: Prediction tensor supplying the target dtype.

        Returns:
            ``(B, chunk_size, action_dim)``.

        Raises:
            KeyError: If ground-truth actions are absent.
        """
        if ACTION not in batch:
            msg = "Ground-truth `action` is required to compute the VLA-Adapter loss."
            raise KeyError(msg)
        target = batch[ACTION][:, : self._config.chunk_size, : self._config.max_action_dim]
        return target.to(like.dtype)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Standard PyTorch forward.

        Args:
            batch: Preprocessed batch dict.

        Returns:
            ``(loss, loss_dict)`` in training mode, else the action chunk.
        """
        if self.training:
            return self.compute_loss(batch)
        return self._predict(batch)

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute the L1 training loss against the ground-truth chunk.

        Args:
            batch: Preprocessed batch dict; must contain ``action``.

        Returns:
            ``(loss, loss_dict)``. Dict values stay tensors so ``torch.compile``
            graphs are not broken by host syncs.
        """
        predicted = self._predict(batch)
        target = self._target_actions(batch, like=predicted)

        loss = nn.functional.l1_loss(predicted, target)
        return loss, {"loss": loss.detach(), "l1_loss": loss.detach()}

    @torch.no_grad()
    def compute_val_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute validation loss.

        The training objective is already a direct action-prediction error, not
        a stochastic surrogate, so validation reuses it. MSE is also reported
        for comparability with the flow-matching families.

        Args:
            batch: Preprocessed batch dict with ground-truth actions.

        Returns:
            ``(loss, loss_dict)``.
        """
        predicted = self._predict(batch)
        target = self._target_actions(batch, like=predicted)

        loss = nn.functional.l1_loss(predicted, target)
        mse = nn.functional.mse_loss(predicted, target)
        return loss, {"loss": loss.detach(), "l1_loss": loss.detach(), "mse_loss": mse.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict an action chunk without gradients.

        Args:
            batch: Preprocessed batch dict.

        Returns:
            ``(B, chunk_size, action_dim)``.
        """
        return self._predict(batch)

    @property
    def reward_delta_indices(self) -> None:
        """Return reward indices; rewards are not implemented.

        Returns:
            None
        """
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Get action indices relative to the current timestep.

        Returns:
            Relative action indices.
        """
        return list(range(self._config.chunk_size))

    @property
    def observation_delta_indices(self) -> list[int]:
        """Get observation indices relative to the current timestep.

        Returns:
            Relative observation indices.
        """
        return [0]
