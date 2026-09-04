# Copyright (C) 2026 Xiaomi Corporation.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Assembled XR0 Vision-Language-Action model on the framework base ``Model``.

``XR0Model`` is the XR0 implementation: it wires the Qwen3-VL
backbone to the DiT action expert (ported from the source
``xr0/mibot/models/VLA/XR0.py`` ``XR0.forward``) -- continuing the VLM's MRoPE
sequence into the DiT tokens, building the joint ``[VLM-cache | local-causal]``
attention mask, and computing the flow-matching loss.
"""

from __future__ import annotations

import logging
import random
from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from physicalai.policies.base import Model

from .dit import XR0FlowModel
from .export_openvino import install_export_rmsnorm
from .qwen3_vlm import XR0Qwen3VL

logger = logging.getLogger(__name__)

# Extra MRoPE offset applied to the (non-prefix) action tokens so they do not
# collide with the state / prefix positions (matches the source implementation).
_ACTION_POSITION_OFFSET = 10
# Number of trailing prefix tokens always kept visible when randomly masking.
_PREFIX_KEEP_LAST_K = 2
# Probability of drawing a random action prefix during asynchronous training.
_ASYNC_PREFIX_PROB = 0.5


class XR0Model(Model):
    """Qwen3-VL backbone + DiT rectified-flow action expert as a framework Model."""

    saved_causal_mask: torch.Tensor

    def __init__(  # noqa: PLR0913
        self,
        *,
        vlm: XR0Qwen3VL | None = None,
        vlm_model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        vlm_attn_implementation: str = "flash_attention_2",
        state_shape: tuple[int, int] = (1, 32),
        action_shape: tuple[int, int] = (30, 32),
        dit_num_layers: int = 16,
        dit_hidden_size: int = 1024,
        dit_head_dim: int = 128,
        dit_kv_heads: int = 8,
        num_steps: int = 5,
        flow_sampling: str = "beta",
        local_window: int = 4,
        training_repeat: int = 4,
        enable_freq: bool = False,
        prefix_mask_prob: float = 0.5,
        async_train: bool = False,
        gradient_checkpointing: bool = False,
        freeze_vision_encoder: bool = False,
        freeze_input_embeddings: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Assemble the VLM backbone and DiT action expert.

        Args:
            vlm: Pre-built Qwen3-VL shim. When ``None`` it is loaded from
                ``vlm_model_id`` via ``from_pretrained`` (inject a small one in
                tests to avoid the model download).
            vlm_model_id: HuggingFace id used when ``vlm`` is not supplied.
            vlm_attn_implementation: Attention backend for the loaded VLM.
            state_shape: ``(state_len, state_dim)`` of the bimanual state.
            action_shape: ``(action_len, action_dim)`` of the action chunk.
            dit_num_layers: Number of DiT decoder layers (<= VLM text layers).
            dit_hidden_size: DiT hidden width.
            dit_head_dim: DiT attention head dim (must match the VLM head dim so
                the DiT can consume the VLM KV-cache).
            dit_kv_heads: DiT key/value heads (must match the VLM kv heads).
            num_steps: Euler integration steps for inference.
            flow_sampling: Training timestep distribution (``"beta"`` /
                ``"logit_normal"`` / other -> uniform).
            local_window: Local-attention window for the action tokens.
            training_repeat: Per-sample training repeat factor.
            enable_freq: Add the frequency-domain loss term.
            prefix_mask_prob: Probability of masking a prefix token in training.
            async_train: Randomly condition on an action prefix during training.
            gradient_checkpointing: Enable gradient checkpointing on the VLM
                vision tower to trade compute for activation memory.
            freeze_vision_encoder: Freeze the VLM vision tower parameters.
            freeze_input_embeddings: Freeze the VLM token-embedding table.
            dtype: Model dtype.
        """
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.local_window = local_window
        self.training_repeat = training_repeat
        self.freq_coefficient = 1.0 if enable_freq else 0.0
        self.prefix_mask_prob = prefix_mask_prob
        self.async_train = async_train
        self._dtype = dtype
        # When True (set only during ``action_mode="delta"`` OpenVINO export) the
        # traced eval graph emits the current-frame ``state`` as a second output
        # so the Runtime ``xr0_denormalize`` step can re-add it to the predicted
        # delta.
        self.export_state_passthrough = False

        # VLM backbone (surfaces the 3D MRoPE position_ids for the DiT).
        if vlm is None:
            vlm = XR0Qwen3VL.from_pretrained(
                vlm_model_id,
                attn_implementation=vlm_attn_implementation,
                dtype=dtype,
            )
        self.vlm = vlm

        # Memory-footprint controls (mirror the original XR0 training recipe):
        # checkpoint the vision tower's activations and freeze the (large)
        # token-embedding table so it carries no gradients / optimizer state.
        if gradient_checkpointing:
            self.vlm.model.visual.gradient_checkpointing_enable()
        if freeze_vision_encoder:
            self.vlm.model.visual.requires_grad_(requires_grad=False)
        if freeze_input_embeddings:
            self.vlm.model.get_input_embeddings().requires_grad_(requires_grad=False)

        # DiT action expert + rectified-flow orchestration.
        self.flow = XR0FlowModel(
            state_shape=state_shape,
            action_shape=action_shape,
            dit_num_layers=dit_num_layers,
            dit_hidden_size=dit_hidden_size,
            dit_head_dim=dit_head_dim,
            dit_kv_heads=dit_kv_heads,
            num_steps=num_steps,
            flow_sampling=flow_sampling,
            dtype=dtype,
        )

        # RoPE continuation into the DiT (same config as the VLM text stack).
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(cast("Any", self.vlm.config.text_config))

        # Local causal mask over [sink + state + action] tokens.
        saved = self._local_causal_mask(state_shape[-2], action_shape[-2]).unsqueeze(0).int()
        self.register_buffer("saved_causal_mask", saved, persistent=False)

    # ------------------------------------------------------------------ #
    # Framework Model interface                                          #
    # ------------------------------------------------------------------ #

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]] | torch.Tensor:
        """Training: flow-matching loss. Eval: predicted action chunk.

        Returns:
            Training: ``(loss, loss_dict)``. Eval: action tensor.
        """
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    def compute_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute the rectified-flow training loss.

        Returns:
            Tuple of (loss tensor with grad, dict of float loss components).
        """
        loss_dict = cast("dict[str, torch.Tensor]", self._run(batch, return_loss=True))
        return loss_dict["loss"], {key: float(value.detach()) for key, value in loss_dict.items()}

    def predict_action_chunk(self, batch: dict[str, Any]) -> torch.Tensor:
        """Denoise an action chunk from the batch.

        Returns:
            Predicted action tensor ``(B, action_len, action_dim)``.
        """
        return cast("torch.Tensor", self._run(batch, return_loss=False))

    def prepare_ingraph_export(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.LongTensor,
    ) -> None:
        """Bake the fixed vision geometry into the VLM for a self-contained export.

        Args:
            input_ids: Token ids of the representative padded prompt ``(1, L)``.
            attention_mask: Attention mask of the same prompt ``(1, L)``.
            image_grid_thw: The fixed vision geometry ``(num_images, 3)``.
        """
        self.vlm.prepare_ingraph_export(input_ids, attention_mask, image_grid_thw)
        install_export_rmsnorm(self)

    @property
    def reward_delta_indices(self) -> None:
        """Rewards are not modelled.

        Returns:
            None
        """
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Relative indices of the predicted action chunk.

        Returns:
            ``list(range(action_len))``.
        """
        return list(range(self.action_shape[-2]))

    @property
    def observation_delta_indices(self) -> None:
        """Only the current observation is used.

        Returns:
            None
        """
        return None

    # ------------------------------------------------------------------ #
    # Attention masks                                                    #
    # ------------------------------------------------------------------ #

    def _local_causal_mask(
        self,
        state_length: int,
        action_length: int,
        device: torch.device | None = None,
        *,
        local: bool = True,
    ) -> torch.Tensor:
        """Build the 2D local causal mask over ``[sink, state, action]`` tokens.

        When ``local`` is False the action-action block is a plain (full) causal
        ``tril`` -- the deployed inference checkpoint (modeling_mibot.py) does not
        apply the banded ``local_window``; only training (XR0.py) does.

        Returns:
            The 2D causal mask over the concatenated token sequence.
        """
        s_len = state_length + 1
        a_len = action_length
        mask_ss = torch.tril(torch.ones(s_len, s_len, device=device))
        mask_sa = torch.zeros(s_len, a_len, device=device)
        mask_as = torch.ones(a_len, s_len, device=device)
        mask_aa = torch.tril(torch.ones(a_len, a_len, device=device))
        if local:
            mask_aa *= torch.triu(torch.ones(a_len, a_len, device=device), diagonal=-self.local_window)
        top = torch.cat([mask_ss, mask_sa], dim=1)
        bottom = torch.cat([mask_as, mask_aa], dim=1)
        return torch.cat([top, bottom], dim=0)

    def _make_local_causal_mask(
        self,
        batch_size: int,
        state_length: int,
        action_length: int,
        device: torch.device,
        *,
        local: bool = True,
    ) -> torch.Tensor:
        """Batched local causal mask, reusing the cached buffer for default shapes.

        Returns:
            The batched causal mask of shape ``(batch_size, q_len, q_len)``.
        """
        if local and state_length == self.state_shape[-2] and action_length == self.action_shape[-2]:
            return self.saved_causal_mask.expand(batch_size, -1, -1)
        mask = self._local_causal_mask(state_length, action_length, device, local=local)
        return mask.unsqueeze(0).int().expand(batch_size, -1, -1)

    def _random_mask_prefix(
        self,
        causal_mask: torch.Tensor,
        prefix_length: int,
        state_length: int,
    ) -> torch.Tensor:
        """Randomly hide part of the action prefix from the suffix tokens.

        Returns:
            The (possibly cloned) causal mask with part of the prefix hidden.
        """
        if prefix_length <= _PREFIX_KEEP_LAST_K:
            return causal_mask

        action_start = 1 + state_length
        masked_prefix_end = action_start + prefix_length - _PREFIX_KEEP_LAST_K
        suffix_start = action_start + prefix_length
        if suffix_start >= causal_mask.shape[-1]:
            return causal_mask

        causal_mask = causal_mask.clone()
        num_maskable = prefix_length - _PREFIX_KEEP_LAST_K
        rand_mask = torch.rand(num_maskable, device=causal_mask.device) < self.prefix_mask_prob
        causal_mask[:, suffix_start:, action_start:masked_prefix_end] *= (~rand_mask).int()
        return causal_mask

    # ------------------------------------------------------------------ #
    # Training-time repeat                                               #
    # ------------------------------------------------------------------ #

    def _repeat_tensor(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Repeat a tensor ``training_repeat`` times along ``dim`` (training only).

        Returns:
            The repeated tensor (or ``x`` unchanged outside training).
        """
        if not self.training or self.training_repeat <= 1:
            return x
        return x.repeat_interleave(self.training_repeat, dim=dim)

    def _repeat_past_key_values(
        self,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Repeat every KV-cache entry to match the repeated batch.

        Returns:
            The KV-cache with every entry repeated (or unchanged outside training).
        """
        if not self.training or self.training_repeat <= 1:
            return past_key_values
        return [(self._repeat_tensor(key), self._repeat_tensor(value)) for key, value in past_key_values]

    # ------------------------------------------------------------------ #
    # Batch helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_prefix_length(prefix_length: int | torch.Tensor | None, action_length: int) -> int:
        """Coerce a possibly-tensor ``prefix_length`` into a bounded int.

        Returns:
            The prefix length clamped to ``[0, action_length]``.
        """
        if isinstance(prefix_length, torch.Tensor):
            prefix_length = 0 if prefix_length.numel() == 0 else int(prefix_length.flatten()[0].item())
        elif prefix_length is None:
            prefix_length = 0
        else:
            prefix_length = int(prefix_length)
        return max(0, min(prefix_length, action_length))

    def get_action_input(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pop ``action`` / ``action_mask`` / ``state`` from the batch.

        Zero-filled defaults are returned for inference when they are absent.

        Returns:
            Tuple of ``(action, action_mask, state)``.
        """
        device = batch["input_ids"].device
        if "action" in batch:
            action = batch.pop("action").to(self._dtype)
            action_mask = batch.pop("action_mask", None)
            if action_mask is None:
                action_mask = torch.ones_like(action, dtype=torch.int32)
        else:
            action = torch.zeros((1, *self.action_shape), device=device, dtype=self._dtype)
            action_mask = torch.ones_like(action, dtype=torch.int32)
        # Use the real current state whenever the batch provides it. The DiT
        # conditions on it, and for ``action_mode="delta"`` the OpenVINO export
        # echoes it as the ``state_passthrough`` output the postprocessor adds
        # back to recover the absolute action. (Zero-filling here would bake a
        # constant-zero passthrough, leaving the exported delta un-inverted.)
        if "state" in batch:
            state = batch.pop("state").to(self._dtype)
        else:
            state = torch.zeros((1, *self.state_shape), device=device, dtype=self._dtype)
        return action, action_mask, state

    def _sample_noise(self, action: torch.Tensor, seed: int | torch.Tensor | None) -> torch.Tensor:
        """Draw the rectified-flow starting noise.

        When ``seed`` is provided (inference only), the draw is made
        deterministic per observation by seeding the RNG and restoring the
        previous global state afterwards -- byte-compatible with the source
        model's ``torch.manual_seed(seed)`` around a single ``randn_like``.

        At inference the noise is drawn in float32 and cast to the action dtype.
        The model runs in bf16, but the Intel GPU OpenVINO plugin has no layout
        for a bf16 ``RandomUniform``;  an f32 draw exports to a GPU-compatible
        ``RandomUniform`` + cast and is numerically equivalent. Training keeps the
        native ``randn_like`` draw so its RNG stream is unchanged.

        Returns:
            Gaussian noise tensor shaped like ``action``.
        """

        def _draw() -> torch.Tensor:
            if self.training:
                return torch.randn_like(action)
            return torch.randn(action.shape, dtype=torch.float32, device=action.device).to(action.dtype)

        if seed is None or self.training:
            return _draw()

        seed_val = int(seed.flatten()[0].item()) if isinstance(seed, torch.Tensor) else int(seed)
        cpu_rng_state = torch.get_rng_state()
        gpu_rng_state = torch.cuda.get_rng_state(action.device) if action.is_cuda else None
        torch.manual_seed(seed_val)
        noise = _draw()
        torch.set_rng_state(cpu_rng_state)
        if gpu_rng_state is not None:
            torch.cuda.set_rng_state(gpu_rng_state, action.device)
        return noise

    # ------------------------------------------------------------------ #
    # Core orchestration                                                 #
    # ------------------------------------------------------------------ #

    def _run(  # noqa: PLR0914, PLR0915
        self,
        batch: dict[str, Any],
        *,
        return_loss: bool,
    ) -> torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """VLM encode -> MRoPE continuation -> rectified-flow train / inference.

        Returns:
            The predicted actions (inference) or the loss dict (training).
        """
        prefix_length = batch.pop("prefix_length", 0)
        seed = batch.pop("seed", None)

        # VLM forward with KV-cache; the shim also surfaces the 3D position ids.
        vlm_outputs = self.vlm(**batch, use_cache=True)
        past_key_values = [(layer.keys, layer.values) for layer in vlm_outputs.past_key_values.layers]

        action, action_mask, state = self.get_action_input(batch)
        action_bs, action_length, _ = action.shape
        _, state_length, _ = state.shape
        q_len = action_length + state_length + 1  # +1 sink token
        prefix_length = self._normalize_prefix_length(prefix_length, action_length)

        if self.training:
            prefix_length = 0
            if self.async_train and random.random() < _ASYNC_PREFIX_PROB:  # noqa: S311
                prefix_length = random.randint(1, min(6, action_length))  # noqa: S311
        prefix = action[:, :prefix_length]

        # Continue the VLM MRoPE sequence into the DiT tokens.
        position_ids = (
            torch.arange(0, q_len, device=action.device).view(1, 1, -1).repeat(3, action_bs, 1)
            + vlm_outputs.position_ids.max(dim=-1)[0][..., None]
            + 1
        )
        # The published inference checkpoint (modeling_mibot.py) does NOT offset the
        # action tokens; only the training path (XR0.py) applies +10 to open a gap
        # between prefix and noisy tokens. Gate the offset to training for parity.
        if self.training and action_length > prefix_length:
            position_ids[:, :, -(action_length - prefix_length) :] += _ACTION_POSITION_OFFSET

        # Joint attention mask: [VLM-cache | local-causal DiT block].
        cache_mask = batch["attention_mask"][:, None, :].expand(-1, q_len, -1)
        # Deployed inference uses a full causal DiT mask; the banded local window
        # (local_window) is a training-only attention scheme.
        causal_mask = self._make_local_causal_mask(
            action_bs,
            state_length,
            action_length,
            action.device,
            local=self.training,
        )
        if self.training and prefix_length > _PREFIX_KEEP_LAST_K:
            causal_mask = self._random_mask_prefix(causal_mask, prefix_length, state_length)
        attn_mask = torch.cat([cache_mask, causal_mask], dim=-1)[:, None].bool()

        state_embed = self.flow.state_projector(state)

        if self.training and self.training_repeat > 1:
            position_ids = self._repeat_tensor(position_ids, dim=1)
            action = self._repeat_tensor(action)
            prefix = self._repeat_tensor(prefix)
            action_mask = self._repeat_tensor(action_mask)
            state_embed = self._repeat_tensor(state_embed)
            attn_mask = self._repeat_tensor(attn_mask)
            past_key_values = self._repeat_past_key_values(past_key_values)

        position_embeds = self.rotary_emb(action, position_ids)
        noise = self._sample_noise(action, seed)

        if self.training:
            pred, target, action_mask, weight = self._training_step(
                action,
                noise,
                action_mask,
                state_embed,
                position_embeds,
                past_key_values,
                attn_mask,
                prefix,
                prefix_length,
            )
        else:
            target = action
            dit_kwargs = self._dit_kwargs(
                action_mask,
                state_embed,
                position_embeds,
                past_key_values,
                attn_mask,
                prefix_length,
            )
            pred = self.flow._flow_generate(torch.cat([prefix, noise[:, prefix_length:]], dim=1), dit_kwargs)  # noqa: SLF001
            weight = None

        if return_loss:
            # One-shot magnitude diagnostic: isolates the source of a loss
            # explosion (raw state scale vs VLM cache vs DiT output). Fires once
            # per process on the first training forward.
            if self.training and not getattr(self, "_xr0_diag_logged", False):
                self._xr0_diag_logged = True
                with torch.no_grad():

                    def amax(tensor: torch.Tensor) -> float:
                        return float(tensor.detach().float().abs().max())

                    pkv_k = max(amax(layer[0]) for layer in past_key_values)
                    pkv_v = max(amax(layer[1]) for layer in past_key_values)
                    logger.warning(
                        "[XR0 diag] state|max=%.4g state_embed|max=%.4g pkv_k|max=%.4g "
                        "pkv_v|max=%.4g noise|max=%.4g action|max=%.4g pred|max=%.4g target|max=%.4g",
                        amax(state),
                        amax(state_embed),
                        pkv_k,
                        pkv_v,
                        amax(noise),
                        amax(action),
                        amax(pred),
                        amax(target),
                    )
            return self._flow_loss(pred, target, action_mask, weight)
        # The DiT action head runs in bf16, which NumPy cannot represent, so the
        # Runtime OpenVINO adapter would fail to read a bf16 ``action`` output. In
        # in-graph export mode, cast the action to f32 so ``to_openvino`` bakes a
        # single, self-consistent f32-output IR -- no fragile post-hoc re-save of
        # the multi-GB ``.bin``. Numerically identical at inference.
        if getattr(self.vlm, "_ingraph_export", False):
            # For ``action_mode="delta"`` also echo the current-frame state as a
            # second f32 output so the Runtime postprocessor can reconstruct the
            # absolute action (``delta + state``) without needing the observation.
            if self.export_state_passthrough:
                return pred.float(), state.float()
            return pred.float()
        return pred

    @staticmethod
    def _dit_kwargs(
        action_mask: torch.Tensor,
        state_embed: torch.Tensor,
        position_embeds: tuple[torch.Tensor, torch.Tensor],
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        attn_mask: torch.Tensor,
        prefix_length: int,
    ) -> dict[str, Any]:
        """Bundle the shared keyword args forwarded to ``dit_forward``.

        Returns:
            The keyword-argument dict forwarded to ``dit_forward``.
        """
        return {
            "action_mask": action_mask,
            "state_embed": state_embed,
            "position_embeds": position_embeds,
            "past_key_values": past_key_values,
            "attn_mask": attn_mask,
            "prefix_length": prefix_length,
        }

    def _training_step(
        self,
        action: torch.Tensor,
        noise: torch.Tensor,
        action_mask: torch.Tensor,
        state_embed: torch.Tensor,
        position_embeds: tuple[torch.Tensor, torch.Tensor],
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        attn_mask: torch.Tensor,
        prefix: torch.Tensor,
        prefix_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One rectified-flow training prediction (velocity + loss weighting).

        Returns:
            Tuple of ``(pred, target, action_mask, weight)`` for the loss.
        """
        t = self.flow._sample_timestep(action.shape[0], dtype=action.dtype, device=action.device)  # noqa: SLF001
        t = t.unsqueeze(1).unsqueeze(1)
        noisy_action = self.flow._flow_interpolate(noise, action, t)  # noqa: SLF001
        target = self.flow._flow_velocity_target(noise, action)  # noqa: SLF001

        dit_kwargs = self._dit_kwargs(
            action_mask,
            state_embed,
            position_embeds,
            past_key_values,
            attn_mask,
            prefix_length,
        )
        pred = self.flow.dit_forward(
            torch.cat([prefix, noisy_action[:, prefix_length:]], dim=1),
            t,
            **dit_kwargs,
        )[:, prefix_length:]
        target = target[:, prefix_length:]

        if prefix_length > 0:
            with torch.no_grad():
                pred_prefix = self.flow._flow_generate(  # noqa: SLF001
                    torch.cat([prefix, noise[:, prefix_length:]], dim=1),
                    dit_kwargs,
                )
            weight = (pred_prefix[:, prefix_length:] - action[:, prefix_length:]).abs()
        else:
            weight = torch.ones_like(pred)

        return pred, target, action_mask[:, prefix_length:], weight

    def _flow_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        action_mask: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Masked MSE (+ optional frequency-domain) rectified-flow loss.

        Returns:
            A dict with ``loss``, ``loss_mse`` and ``loss_freq`` entries.
        """
        pred = pred.float()
        target = target.float()
        action_mask = action_mask.bool()
        weight = torch.ones_like(pred) if weight is None else weight.float()

        if not torch.any(action_mask):
            zero = (pred.reshape(-1)[0] - target.reshape(-1)[0]) * 0.0
            return {"loss": zero, "loss_mse": zero, "loss_freq": zero}

        with torch.no_grad():
            masked_weight = weight[action_mask]
            if masked_weight.numel() > 0:
                weight = weight.clone()
                weight[action_mask] /= masked_weight.mean()
                weight = torch.clamp(weight, min=0.5, max=5.0)

        loss_mse = (F.mse_loss(pred, target, reduction="none") * weight)[action_mask].mean()

        if self.freq_coefficient > 0.0:
            # Frequency-domain (chunk-axis FFT) term over valid action channels
            # only. Padded channels have target ``-noise`` (fresh Gaussian), which
            # is unpredictable and would add an irreducible floor while diluting
            # the real-channel gradient. The time-axis FFT is per-channel, so
            # dropping padded channels afterwards is exact.
            freq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs()
            weight_dct = weight.mean(dim=[1, 2])
            freq *= weight_dct.unsqueeze(1).unsqueeze(2)
            chan_mask = action_mask.any(dim=1, keepdim=True).to(freq.dtype)  # (B, 1, D)
            denom = (chan_mask.sum() * freq.shape[1]).clamp_min(1.0)
            loss_freq = (freq * chan_mask).sum() / denom
        else:
            loss_freq = loss_mse * 0.0

        loss = 0.5 * loss_mse + self.freq_coefficient * loss_freq
        return {"loss": loss, "loss_mse": loss_mse, "loss_freq": loss_freq}
