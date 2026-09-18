# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model definition for the native policy design."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn
from physicalai.policies.mixins import PeftModelMixin, RTCModelMixin

from .base import TemplateModel


class TextModule(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embedding(input_ids)


class TransformerModule(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=intermediate_size,
                dropout=attention_dropout,
                layer_norm_eps=layer_norm_eps,
                batch_first=True,
            )
            for _ in range(num_hidden_layers)
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class NewPolicyModel(PeftModelMixin, RTCModelMixin, TemplateModel):
    """Flat PyTorch model implementing loss and full-chunk prediction.

    A policy model must be fully described by plain constructor arguments, implement
    ``compute_loss()`` and ``predict_action_chunk()``, and declare temporal delta
    indices consumed by data loading. It returns the complete native action chunk;
    processors own normalization and external shape adaptation. The policy, not the
    model, retains ``NewPolicyModelConfig``.
    """
    def __init__(
        self,
        *,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 18,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        vocab_size: int = 256_000,
        chunk_size: int = 32,
        action_dim: int = 32,
    ) -> None:
        super().__init__()
        self._chunk_size = chunk_size
        self._max_action_dim = action_dim

        self.text_model = TextModule(vocab_size=vocab_size, hidden_size=hidden_size)
        self.transformer = TransformerModule(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            layer_norm_eps=layer_norm_eps,
        )
        self.action_head = nn.Linear(hidden_size, action_dim * chunk_size)
        self.gradient_checkpointing_enabled = False
        self.weights_load_count = 0

    def load_weights(self, weights_path: str | Path) -> None:
        del weights_path
        self.weights_load_count += 1
        fake_state_dict = {name: value.detach().clone() for name, value in self.state_dict().items()}
        self.load_state_dict(fake_state_dict, strict=True)

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True

    @classmethod
    def get_default_peft_targets(cls) -> tuple[str, ...]:
        return ("action_head",)

    def _predict_actions(self, batch: Mapping[str, Tensor]) -> Tensor:
        hidden_states = self.text_model(batch["input_ids"])
        hidden_states = self.transformer(hidden_states)
        actions = self.action_head(hidden_states[:, -1])
        return actions.reshape(actions.shape[0], self._chunk_size, -1)

    def forward(self, batch: dict[str, Tensor]) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor | float]]:
        actions = self._predict_actions(batch)
        loss = torch.nn.functional.mse_loss(actions, batch["action"])
        return loss, {"loss": loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the full ``(batch, chunk_size, action_dim)`` model output."""
        actions = self._predict_actions(batch)
        if self.enable_rtc:
            actions = 0.5 * actions + 0.5 * actions[:, :1]
        return actions

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self._chunk_size))

    @property
    def observation_delta_indices(self) -> None:
        return None
