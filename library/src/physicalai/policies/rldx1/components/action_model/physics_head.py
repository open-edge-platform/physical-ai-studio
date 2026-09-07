# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Physics conditioning + flow matching stream for RLDXActionModel."""

import logging
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .physics import (
    PhysicalSignalDecoder,
    PhysicalSignalEncoder,
    PhysicsNoiseEncoder,
)

logger = logging.getLogger(__name__)


class PhysicsInferenceState(NamedTuple):
    """Immutable state carried through Euler inference loop."""

    embs: torch.Tensor | None  # current physics embeddings for MSAT
    hist_tok: torch.Tensor | None  # fixed history tokens (computed once)
    fut: torch.Tensor | None  # evolving future state (Euler updated)
    attn_mask: torch.Tensor | None  # fixed attention mask


# --- State dict key remapping for backward compatibility ---

_PHYSICS_KEY_RENAMES = [
    ("physics_encoder.", "physics.physics_cond_encoder."),  # very old → new
    ("physics_cond_encoder.", "physics.physics_cond_encoder."),  # old → new
    ("physics_fut_encoder.", "physics.physics_fut_encoder."),  # old → new
    ("physics_decoder.", "physics.physics_decoder."),  # old → new
]


def remap_physics_keys(state_dict: dict) -> dict:
    """Remap older physics state_dict keys onto the current PhysicsHead layout.

    Handles both ``physics_encoder.*`` and ``physics_cond_encoder.*`` prefixes.
    Keys already in the current ``physics.physics_*`` layout are left
    unchanged, including nested cases like
    ``action_model.physics.physics_cond_encoder.*``.

    Args:
        state_dict: The state dict whose keys should be remapped.

    Returns:
        A new state dict with all keys updated to the current layout.
    """
    remapped = {}
    renamed_count = 0
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in _PHYSICS_KEY_RENAMES:
            # Idempotence: if the target prefix is already present anywhere in
            # the key, this key is already in the new layout — do not rename.
            if new_prefix in key:
                break
            if old_prefix in key:
                new_key = key.replace(old_prefix, new_prefix, 1)
                renamed_count += 1
                break
        remapped[new_key] = value
    if renamed_count > 0:
        logger.info("[Physics] Remapped %d older-format keys → physics.* layout", renamed_count)
    return remapped


class NoOpPhysicsHead(nn.Module):
    """No-op physics head. Used when use_physics=False."""

    @staticmethod
    def prepare_train(
        action_input: Any,  # noqa: ANN401, ARG004
        t_raw: torch.Tensor,  # noqa: ARG004
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Return empty physics inputs for training (no-op).

        Args:
            action_input: Action input data (unused).
            t_raw: Raw timestep tensor (unused).

        Returns:
            A tuple of (None, None, None) for physics_embs, physics_attn_mask, and
            physics_velocity.
        """
        return None, None, None

    @staticmethod
    def compute_loss(
        physics_model_output: torch.Tensor | None,  # noqa: ARG004
        physics_velocity: torch.Tensor | None,  # noqa: ARG004
        action_mask: torch.Tensor | None,  # noqa: ARG004
        physics_attn_mask: torch.Tensor | None,  # noqa: ARG004
    ) -> torch.Tensor | None:
        """Return no physics loss (no-op).

        Args:
            physics_model_output: Model output tensor (unused).
            physics_velocity: Velocity tensor (unused).
            action_mask: Action mask tensor (unused).
            physics_attn_mask: Physics attention mask (unused).
        """
        return None

    @staticmethod
    def prepare_inference(
        action_input: Any,  # noqa: ANN401, ARG004
        batch_size: int,  # noqa: ARG004
        device: torch.device | str,  # noqa: ARG004
        dtype: torch.dtype,  # noqa: ARG004
    ) -> PhysicsInferenceState:
        """Return an empty physics inference state (no-op).

        Args:
            action_input: Action input data (unused).
            batch_size: Batch size (unused).
            device: Target device (unused).
            dtype: Target dtype (unused).

        Returns:
            PhysicsInferenceState with all fields set to None.
        """
        return PhysicsInferenceState(embs=None, hist_tok=None, fut=None, attn_mask=None)

    @staticmethod
    def build_tokens(
        state: PhysicsInferenceState,  # noqa: ARG004
        timesteps_tensor: torch.Tensor,  # noqa: ARG004
    ) -> torch.Tensor | None:
        """Return no physics tokens (no-op).

        Args:
            state: Current physics inference state (unused).
            timesteps_tensor: Timestep tensor (unused).
        """
        return None

    @staticmethod
    def update_state(
        state: PhysicsInferenceState,
        model_output: Any,  # noqa: ANN401, ARG004
        dt: float | torch.Tensor,  # noqa: ARG004
    ) -> PhysicsInferenceState:
        """Return the unchanged state (no-op).

        Args:
            state: Current physics inference state.
            model_output: Model output (unused).
            dt: Euler step size (unused).

        Returns:
            The unchanged state.
        """
        return state


class PhysicsHead(nn.Module):
    """Owns physics encoder/decoder modules and related config.

    Registered as a submodule of RLDXActionModel (self.physics).
    """

    def __init__(
        self,
        physics_dim: int,
        embed_dim: int,
        msat_output_dim: int,
        physics_delta_indices: list[int] | None,
        physics_use_flow_matching: bool,  # noqa: FBT001
        physics_loss_weight: float,
        action_horizon: int,
        physics_dropout_prob: float = 0.0,
    ) -> None:
        """Initialize PhysicsHead modules and validate configuration.

        Args:
            physics_dim: Dimensionality of the physics signal.
            embed_dim: Embedding dimension used by the encoder and decoder.
            msat_output_dim: Output dimension from MSAT consumed by the decoder.
            physics_delta_indices: List of time delta indices; non-positive values
                become history tokens, positive values become future tokens.
                Pass None to disable both history and future streams.
            physics_use_flow_matching: Whether to use flow matching for physics
                prediction.
            physics_loss_weight: Weight applied to the physics prediction loss.
            action_horizon: Number of action steps; must equal ``physics_fut_len``
                when flow matching is enabled.
            physics_dropout_prob: Probability of dropping physics conditioning tokens
                per sample during training.

        Raises:
            ValueError: If ``physics_use_flow_matching`` is True and
                ``physics_fut_len`` does not equal ``action_horizon``.
        """
        super().__init__()

        self.physics_dim = physics_dim
        self.embed_dim = embed_dim
        self.physics_loss_weight = physics_loss_weight
        self.physics_use_flow_matching = physics_use_flow_matching
        self.physics_dropout_prob = physics_dropout_prob

        delta = physics_delta_indices or []
        self.physics_hist_len = sum(1 for d in delta if d <= 0)
        self.physics_fut_len = sum(1 for d in delta if d > 0)

        self.physics_cond_encoder = PhysicalSignalEncoder(physics_dim, embed_dim, embed_dim)
        self.physics_fut_encoder = PhysicsNoiseEncoder(physics_dim, embed_dim, embed_dim)
        self.physics_decoder = PhysicalSignalDecoder(msat_output_dim, embed_dim, physics_dim)

        # Learned mask token used to replace dropped physics conditioning tokens.
        # Only created when dropout is on so checkpoints saved with prob=0 stay clean.
        self.physics_mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, embed_dim)) if physics_dropout_prob > 0 else None
        )

        if self.physics_use_flow_matching:
            if self.physics_fut_len != action_horizon:
                msg = (
                    f"physics_fut_len ({self.physics_fut_len}) must equal action_horizon "
                    f"({action_horizon}) so that action_mask can be reused for physics loss masking"
                )
                raise ValueError(msg)
        else:
            logger.info(
                "[Physics] Flow matching disabled. Physics used as conditioning only (no prediction loss)",
            )

        logger.info(
            "[Physics] Physics stream enabled (dim=%s, weight=%s)",
            physics_dim,
            physics_loss_weight,
        )
        logger.info(
            "[Physics] hist_len=%s, fut_len=%s, flow_matching=%s",
            self.physics_hist_len,
            self.physics_fut_len,
            self.physics_use_flow_matching,
        )
        if physics_dropout_prob > 0:
            mode = "hist-only" if self.physics_use_flow_matching else "all-conditioning"
            logger.info("[Physics] physics_dropout_prob=%s (%s)", physics_dropout_prob, mode)

    def _maybe_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply per-sample dropout by replacing dropped tokens with the learned mask token.

        For each sample in the batch, independently draws a Bernoulli variable and,
        if it fires, replaces the entire token slice with ``physics_mask_token``.
        Only active during training when ``physics_dropout_prob > 0``.

        Args:
            tokens: Physics token tensor of shape ``(B, T, embed_dim)``.

        Returns:
            Tensor of the same shape with per-sample dropout applied, or the
            original tensor unchanged when the conditions above are not met.
        """
        if not (
            self.training
            and self.physics_dropout_prob > 0
            and self.physics_mask_token is not None
            and tokens.shape[1] > 0
        ):
            return tokens
        do_dropout = torch.rand(tokens.shape[0], device=tokens.device) < self.physics_dropout_prob
        do_dropout = do_dropout[:, None, None].to(dtype=tokens.dtype)
        return tokens * (1 - do_dropout) + self.physics_mask_token * do_dropout

    def prepare_train(
        self,
        action_input: Any,  # noqa: ANN401
        t_raw: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Encode the physics signal and compute flow-matching targets for training.

        Args:
            action_input: Action input data; must expose a ``physics`` attribute
                (and optionally ``physics_mask``) when physics conditioning is used.
            t_raw: Raw timestep tensor of shape ``(B,)`` drawn from ``[0, 1]``.

        Returns:
            A tuple of:

            - **physics_embs** (``torch.Tensor | None``): Encoded physics tokens of
              shape ``(B, T, embed_dim)``, or ``None`` when no physics input is present.
            - **physics_attn_mask** (``torch.Tensor | None``): Per-sample validity mask
              of shape ``(B,)``, or ``None``.
            - **physics_velocity** (``torch.Tensor | None``): Flow-matching velocity
              targets of shape ``(B, F, physics_dim)``, or ``None`` when flow matching
              is disabled.

        Raises:
            ValueError: If the physics dimension in the data does not match the
                model's expected dimension.
        """
        physics_embs = None
        physics_attn_mask = None
        physics_velocity = None

        if not hasattr(action_input, "physics"):
            return physics_embs, physics_attn_mask, physics_velocity

        data_dim = action_input.physics.shape[-1]
        expected_dim = self.physics_cond_encoder.W1.in_features
        if data_dim != expected_dim:
            msg = (
                f"Physics dim mismatch: data has {data_dim} but model expects {expected_dim} "
                f"(from --physics-dims). Check that --physics-dims matches your dataset."
            )
            raise ValueError(msg)

        if hasattr(action_input, "physics_mask"):
            physics_attn_mask = action_input.physics_mask.view(-1)  # [B]

        if not self.physics_use_flow_matching:
            # Conditioning only: all tokens as conditioning. Dropout the whole sequence
            # since there's no flow-matching target to preserve.
            physics_embs = self.physics_cond_encoder(action_input.physics)
            physics_embs = self._maybe_dropout(physics_embs)
        else:
            # Flow matching: split hist/fut -> noise fut -> encode both
            physics_hist = action_input.physics[:, : self.physics_hist_len, :]  # (B, H, D)
            physics_fut_gt = action_input.physics[:, self.physics_hist_len :, :]  # (B, F, D)

            t_broad_p = t_raw[:, None, None]  # (B, 1, 1)
            physics_noise = torch.randn_like(physics_fut_gt)
            # noisy_fut = (1-t) * noise + t * gt  (flow matching interpolation)
            noisy_physics_fut = (1 - t_broad_p) * physics_noise + t_broad_p * physics_fut_gt
            physics_velocity = physics_fut_gt - physics_noise

            if self.physics_hist_len > 0:
                physics_hist_tok = self.physics_cond_encoder(physics_hist)
            else:
                physics_hist_tok = torch.zeros(
                    physics_hist.shape[0],
                    0,
                    self.embed_dim,
                    dtype=physics_hist.dtype,
                    device=physics_hist.device,
                )
            # Hist-only dropout: future tokens are the prediction target and must NOT be masked.
            physics_hist_tok = self._maybe_dropout(physics_hist_tok)
            physics_fut_tok = self.physics_fut_encoder(noisy_physics_fut, t_raw)
            physics_embs = torch.cat([physics_hist_tok, physics_fut_tok], dim=1)

        return physics_embs, physics_attn_mask, physics_velocity

    def compute_loss(
        self,
        physics_model_output: torch.Tensor | None,
        physics_velocity: torch.Tensor | None,
        action_mask: torch.Tensor,
        physics_attn_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Compute the physics prediction loss using flow matching.

        Decodes the predicted velocity from the model's physics hidden states and
        computes a masked MSE loss against the ground-truth flow-matching velocities.

        Args:
            physics_model_output: Hidden state output from the model for physics tokens,
                shape ``(B, T, hidden_dim)``.
            physics_velocity: Ground-truth flow-matching velocities,
                shape ``(B, F, physics_dim)``.
            action_mask: Boolean action validity mask, shape ``(B, T, action_dim)``.
            physics_attn_mask: Per-sample physics validity mask of shape ``(B,)``,
                or ``None``.

        Returns:
            Scalar physics MSE loss tensor, or ``None`` if flow matching is disabled
            or required inputs are missing.
        """
        if not (self.physics_use_flow_matching and physics_model_output is not None and physics_velocity is not None):
            return None

        physics_hidden_fut = physics_model_output[:, -self.physics_fut_len :, :]
        physics_pred_vel = self.physics_decoder(physics_hidden_fut)

        # Combine per-step action_mask (episode boundary) with per-sample physics_attn_mask
        step_mask = action_mask.any(dim=-1).float()  # (B, T) — per-step validity from action
        if physics_attn_mask is not None:
            step_mask *= physics_attn_mask.unsqueeze(1)  # (B, T) * (B, 1)
        mask_3d = step_mask.unsqueeze(-1)  # (B, T, 1)

        loss_unreduced = F.mse_loss(physics_pred_vel, physics_velocity, reduction="none")
        n_valid = mask_3d.sum() * physics_pred_vel.shape[-1]
        return (loss_unreduced * mask_3d).sum() / (n_valid + 1e-6)

    def prepare_inference(
        self,
        action_input: Any,  # noqa: ANN401
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> PhysicsInferenceState:
        """Initialize the physics state for the Euler integration inference loop.

        Encodes any available history tokens once (fixed across loop iterations) and
        initialises the future state as standard Gaussian noise.

        Args:
            action_input: Action input data; may expose ``physics`` and
                ``physics_mask`` attributes.
            batch_size: Batch size used when creating noise tensors from scratch.
            device: Target device for newly created tensors.
            dtype: Target dtype for newly created tensors.

        Returns:
            PhysicsInferenceState with encoded history tokens, initial future noise,
            and an optional per-sample attention mask ready for the Euler loop.
        """
        embs = None
        hist_tok = None
        fut = None
        attn_mask = None

        has_physics_input = action_input is not None and hasattr(action_input, "physics")

        if has_physics_input and hasattr(action_input, "physics_mask"):
            attn_mask = action_input.physics_mask.view(-1)

        if not self.physics_use_flow_matching:
            # Conditioning only: encode all tokens once (fixed outside loop)
            if has_physics_input:
                embs = self.physics_cond_encoder(action_input.physics)
        else:
            # Flow matching: hist conditioning + fut noise init
            if has_physics_input and self.physics_hist_len > 0:
                physics_hist = action_input.physics[:, : self.physics_hist_len, :]
                hist_tok = self.physics_cond_encoder(physics_hist)
            else:
                hist_tok = torch.zeros(
                    batch_size,
                    0,
                    self.embed_dim,
                    dtype=dtype,
                    device=device,
                )
            fut = torch.randn(
                batch_size,
                self.physics_fut_len,
                self.physics_dim,
                dtype=dtype,
                device=device,
            )

        return PhysicsInferenceState(embs=embs, hist_tok=hist_tok, fut=fut, attn_mask=attn_mask)

    def build_tokens(
        self,
        state: PhysicsInferenceState,
        timesteps_tensor: torch.Tensor,
    ) -> torch.Tensor | None:
        """Build physics token tensor for a single Euler integration step.

        Concatenates noise-encoded future tokens with the fixed history tokens, or
        returns the pre-computed conditioning embeddings when flow matching is disabled.

        Args:
            state: Current physics inference state holding history and future tensors.
            timesteps_tensor: Current timestep tensor of shape ``(B,)``.

        Returns:
            Physics token tensor of shape ``(B, T, embed_dim)`` combining history and
            noise-encoded future tokens, or the pre-computed conditioning embeddings.
        """
        if state.hist_tok is not None and state.fut is not None:
            fut_tok = self.physics_fut_encoder(state.fut, timesteps_tensor)
            return torch.cat([state.hist_tok, fut_tok], dim=1)
        return state.embs

    def update_state(
        self,
        state: PhysicsInferenceState,
        model_output: Any,  # noqa: ANN401
        dt: float | torch.Tensor,
    ) -> PhysicsInferenceState:
        """Advance the physics future state by one Euler step.

        Decodes the predicted velocity from the model output and integrates it into
        the current future state tensor.

        Args:
            state: Current physics inference state.
            model_output: Dict containing a ``"physics"`` key with hidden states of
                shape ``(B, T, hidden_dim)``.  If not a dict or the key is absent,
                the state is returned unchanged.
            dt: Euler step size as a scalar float or zero-dimensional tensor.

        Returns:
            New PhysicsInferenceState with the future physics state advanced by one
            Euler step, or the unchanged state if conditions are not met.
        """
        if state.fut is not None and isinstance(model_output, dict) and "physics" in model_output:
            physics_hidden_fut = model_output["physics"][:, -self.physics_fut_len :, :]
            physics_pred_vel = self.physics_decoder(physics_hidden_fut)
            return state._replace(fut=state.fut + dt * physics_pred_vel)
        return state
