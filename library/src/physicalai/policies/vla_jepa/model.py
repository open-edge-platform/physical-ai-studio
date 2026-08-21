# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""VLA-JEPA model implementation.

Ported from LeRobot's ``lerobot.policies.vla_jepa.modeling_vla_jepa``: this module merges the
native VLA-JEPA model and the batch-adaptation layer of ``VLAJEPAPolicy`` into a single Studio
:class:`~physicalai.policies.base.Model`.

Components:
    - Qwen3-VL: vision-language backbone producing the conditioning tokens
    - DiT-B: flow-matching action head predicting the action chunk
    - V-JEPA2: action-conditioned video world model, used as an auxiliary training loss only
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from physicalai.data.observation import ACTION, EXTRA, IMAGES, STATE, TASK, Observation
from physicalai.policies.base import Model
from physicalai.policies.vla_jepa.components.action_head import VLAJEPAActionHead
from physicalai.policies.vla_jepa.components.qwen_interface import Qwen3VLInterface, resolve_torch_dtype
from physicalai.policies.vla_jepa.components.world_model import ActionConditionedVideoPredictor

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .config import VLAJEPAConfig

logger = logging.getLogger(__name__)

ACTION_IS_PAD = EXTRA + ".action_is_pad"


def _lazy_import_transformers() -> tuple:
    """Lazy import the transformers classes the world model needs.

    Returns:
        Tuple containing (AutoModel, AutoVideoProcessor).

    Raises:
        ImportError: If transformers is not installed.
    """
    try:
        from transformers import AutoModel, AutoVideoProcessor  # noqa: PLC0415
    except ImportError as e:
        msg = (
            "VLA-JEPA's world model requires the transformers library.\n\nInstall with:\n"
            "    uv pip install 'physicalai-train[vla_jepa]'"
        )
        raise ImportError(msg) from e
    else:
        return AutoModel, AutoVideoProcessor


def _autocast(device_type: str, dtype: torch.dtype) -> Any:  # noqa: ANN401
    """Return a device-safe autocast context manager.

    A hardcoded ``torch.autocast(dtype=torch.bfloat16)`` breaks on backends without AMP and
    misbehaves on pre-Ampere CUDA, so fall back to a null context where autocast does not apply.

    Args:
        device_type: Device type string, e.g. "cuda", "xpu", "cpu".
        dtype: Requested autocast dtype.

    Returns:
        A context manager suitable for the device.
    """
    if not torch.amp.is_autocast_available(device_type):
        return nullcontext()
    if device_type in {"cpu", "mps"} and dtype not in {torch.bfloat16, torch.float16}:
        # torch.autocast accepts these and then warns-and-disables on every call.
        return nullcontext()
    ampere = 8
    if (
        device_type == "cuda"
        and dtype == torch.bfloat16
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] < ampere
    ):
        dtype = torch.float16
    return torch.autocast(device_type=device_type, dtype=dtype)


class VLAJEPAModel(Model):
    """VLA-JEPA vision-language-action model.

    Unlike the other Studio families, this model takes its :class:`VLAJEPAConfig` directly instead
    of ~45 explicit keyword arguments: the ported submodules (:class:`Qwen3VLInterface`,
    :class:`VLAJEPAActionHead`) are themselves config-driven, and keeping one object avoids
    threading every field through three layers.

    Submodule attribute names deliberately match LeRobot's (``qwen``, ``action_model``,
    ``video_encoder``, ``video_predictor``) so published checkpoints map onto this module.

    Args:
        config: Policy configuration.
        dataset_stats: Dataset normalization statistics, used to report the true action
            dimensionality. Optional.
    """

    def __init__(self, config: VLAJEPAConfig, dataset_stats: dict[str, Any] | None = None) -> None:
        """Build the backbone, the action head and, when enabled, the world model.

        Args:
            config: Policy configuration.
            dataset_stats: Dataset normalization statistics. Optional.
        """
        super().__init__()
        self.config = config
        self._dataset_stats = dataset_stats or {}

        # Vision-language backbone.
        self.qwen = Qwen3VLInterface(config)

        # Tokenizer expansion for the special action tokens.
        self.action_tokens, self.action_token_ids, self.embodied_action_token_id = self.qwen.expand_tokenizer()
        self.register_buffer(
            "_action_token_ids_t",
            torch.tensor(self.action_token_ids, dtype=torch.long),
            persistent=False,
        )

        # Flow-matching DiT action head.
        self.action_model = VLAJEPAActionHead(config, cross_attention_dim=self.qwen.model.config.hidden_size)

        # JEPA world model components (training only).
        # Typed as Any: these come from `transformers.AutoModel` / `AutoVideoProcessor`, whose
        # attributes are not statically known.
        self.video_encoder: Any = None
        self.video_processor: Any = None
        self.video_predictor: ActionConditionedVideoPredictor | None = None
        if config.enable_world_model:
            auto_model_cls, auto_video_processor_cls = _lazy_import_transformers()
            self.video_encoder = auto_model_cls.from_pretrained(
                config.jepa_encoder_name,
                dtype=resolve_torch_dtype(config.torch_dtype),
            )
            self.video_processor = auto_video_processor_cls.from_pretrained(config.jepa_encoder_name)
            self.tubelet_size = int(self.video_encoder.config.tubelet_size)
            self.video_predictor = ActionConditionedVideoPredictor(
                num_frames=config.num_video_frames // self.tubelet_size,
                img_size=self._world_model_image_size(),
                patch_size=16,
                tubelet_size=1,
                embed_dim=self.video_encoder.config.hidden_size * config.num_world_model_views,
                action_embed_dim=self.qwen.model.config.hidden_size,
                predictor_embed_dim=self.video_encoder.config.hidden_size,
                depth=config.predictor_depth,
                num_heads=config.predictor_num_heads,
                mlp_ratio=config.predictor_mlp_ratio,
                num_action_tokens_per_step=config.num_action_tokens_per_timestep,
                dropout=config.predictor_dropout,
            )
        else:
            # The encoder's own tubelet size is authoritative when it exists; without it the
            # config value is all the prompt placeholders have to go on.
            self.tubelet_size = config.jepa_tubelet_size

        if config.freeze_qwen:
            self.qwen.requires_grad_(requires_grad=False)

        # Build the prompt placeholders from the resolved tubelet size.
        num_action_prompt_steps = config.num_video_frames // self.tubelet_size - 1
        self.replace_prompt = "".join(
            token * config.num_action_tokens_per_timestep for token in self.action_tokens[:num_action_prompt_steps]
        )
        self.embodied_replace_prompt = config.embodied_action_token * config.num_embodied_action_tokens_per_instruction

    def _world_model_image_size(self) -> tuple[int, int]:
        """Resolve the spatial size the world-model predictor is built for.

        Returns:
            The ``(height, width)`` the V-JEPA encoder emits tokens for.

        Raises:
            ValueError: If neither the encoder config nor `resize_images_to` provides a size.
        """
        encoder_config = getattr(self.video_encoder, "config", None)
        image_size = getattr(encoder_config, "image_size", None)
        if image_size is not None:
            return (image_size, image_size)
        if self.config.resize_images_to is not None:
            return tuple(self.config.resize_images_to)  # type: ignore[return-value]
        msg = (
            f"The V-JEPA encoder '{self.config.jepa_encoder_name}' does not declare an `image_size`. "
            f"Set `resize_images_to` so the world-model predictor knows its token grid."
        )
        raise ValueError(msg)

    # ---- Qwen encoding -----------------------------------------------------

    def _qwen_last_decoder_hidden(self, qwen_inputs: dict[str, Tensor]) -> Tensor:
        """Return the last decoder hidden state before the final RMSNorm.

        The model was trained on the last block's pre-RMSNorm output, but in transformers 5.x
        ``hidden_states[-1]`` is post-norm, so hook ``language_model.layers[-1]`` instead.

        Calls the inner ``Qwen3VLModel``, not the ``Qwen3VLForConditionalGeneration`` wrapper, whose
        forward would build and discard full-sequence logits over the 151936-token vocab (~3.4 GB in
        bf16 at batch 8). The wrapper stays as ``self.qwen.model`` so ``lm_head`` keeps its
        checkpoint key; only this path skips it.

        Args:
            qwen_inputs: Tokenized chat-template inputs for the backbone.

        Returns:
            Hidden states of shape ``[B, seq_len, H]``.
        """
        captured: list[Tensor] = []

        def _hook(module: torch.nn.Module, inputs: tuple, output: Any) -> None:  # noqa: ANN401, ARG001
            captured.append(output[0] if isinstance(output, tuple) else output)

        last_layer = self.qwen.model.model.language_model.layers[-1]
        handle = last_layer.register_forward_hook(_hook)
        try:
            self.qwen.model.model(**qwen_inputs)
        finally:
            handle.remove()

        return captured[0]

    def _encode_qwen(
        self,
        images: list[list[Tensor]],
        instructions: list[str],
        *,
        need_action_tokens: bool,
    ) -> tuple[Tensor, Tensor | None]:
        """Run Qwen and gather the embodied-action (and optionally action) token hidden states.

        Args:
            images: Per-sample, per-view image tensors.
            instructions: Per-sample language instruction.
            need_action_tokens: Whether the per-timestep action tokens are needed (world model).

        Returns:
            Tuple of (embodied action tokens, action tokens or None).
        """
        qwen_inputs = self.qwen.build_inputs(
            images=images,
            instructions=instructions,
            action_prompt=self.replace_prompt,
            embodied_prompt=self.embodied_replace_prompt,
        )
        input_ids = qwen_inputs["input_ids"]
        embodied_idx = (input_ids == self.embodied_action_token_id).nonzero(as_tuple=True)
        action_idx = None
        if need_action_tokens:
            action_mask = torch.isin(input_ids, self._action_token_ids_t)  # pyrefly: ignore[no-matching-overload]
            action_idx = action_mask.nonzero(as_tuple=True)

        device_type = next(self.parameters()).device.type
        with _autocast(device_type, torch.bfloat16):
            last_hidden = self._qwen_last_decoder_hidden(qwen_inputs)  # [B, seq_len, H]
            b, _, h = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_idx[0], embodied_idx[1], :].view(b, -1, h)
            action_tokens = (
                last_hidden[action_idx[0], action_idx[1], :].view(b, -1, h) if action_idx is not None else None
            )
        return embodied_action_tokens, action_tokens

    # ---- World model -------------------------------------------------------

    @staticmethod
    def _causal_video_embeddings(
        encoder: Any,  # noqa: ANN401
        video_pixels: Tensor,
        tubelet_size: int,
        num_positions: int,
    ) -> Tensor:
        """Encode leading temporal positions from only their own raw-frame prefix.

        A single V-JEPA2 pass over the full clip lets bidirectional attention leak future frames into
        every position's embedding, including the context positions used as predictor input. Running
        one prefix-only pass per context position keeps position i blind to frames after it, at the
        cost of `num_positions` encoder calls instead of one.

        Args:
            encoder: The V-JEPA video encoder (a transformers model).
            video_pixels: Preprocessed video tensor for the encoder.
            tubelet_size: Temporal tubelet size of the encoder.
            num_positions: Number of leading temporal positions to encode.

        Returns:
            Concatenated per-position embeddings.
        """
        positions = []
        for position in range(num_positions):
            prefix = encoder.get_vision_features(
                pixel_values_videos=video_pixels[:, : (position + 1) * tubelet_size],
            )
            tokens_per_position = prefix.shape[1] // (position + 1)
            positions.append(prefix[:, -tokens_per_position:])
        return torch.cat(positions, dim=1)

    @staticmethod
    def _merge_views(embeddings: Tensor, b: int, v: int) -> Tensor:
        """Merge per-view features: ``[B*V, N, H] -> [B, N, V*H]``.

        Rows run view-fastest, since ``videos.reshape(b * v, ...)`` flattens (B, V) row-major. A
        ``chunk(chunks=v, dim=0)`` + ``cat(dim=2)`` merge assumes view-slowest and so concatenates
        features from *different* samples: shape-valid, so it fails silently for b > 1.

        Args:
            embeddings: Per-view embeddings of shape ``[B*V, N, H]``.
            b: Batch size.
            v: Number of views.

        Returns:
            Merged embeddings of shape ``[B, N, V*H]``.
        """
        n_tokens, hidden = embeddings.shape[1], embeddings.shape[2]
        return embeddings.reshape(b, v, n_tokens, hidden).permute(0, 2, 1, 3).reshape(b, n_tokens, v * hidden)

    def _world_model_loss(  # noqa: PLR0914
        self,
        videos: Tensor,
        action_tokens: Tensor,
        reduction: str = "mean",
    ) -> Tensor:
        """Compute the JEPA encode + predictor L1 loss.

        Args:
            videos: Video tensor ``[B, V, T, C, H, W]``, float in [0, 1].
            action_tokens: Qwen action-token hidden states.
            reduction: ``"mean"`` for the scalar loss, ``"none"`` for a per-sample loss ``(B,)``.

        Returns:
            The world-model loss.

        Raises:
            RuntimeError: If the world model was never built.
        """
        if self.video_encoder is None or self.video_processor is None or self.video_predictor is None:
            msg = "The world-model loss requires `enable_world_model=True`."
            raise RuntimeError(msg)
        encoder, processor, predictor = self.video_encoder, self.video_processor, self.video_predictor

        # Match the world model's expected view count: pad with the first view, or trim extras.
        num_views = self.config.num_world_model_views
        if videos.shape[1] < num_views:
            missing = num_views - videos.shape[1]
            videos = torch.cat([videos, videos[:, :1].repeat(1, missing, 1, 1, 1, 1)], dim=1)
        elif videos.shape[1] > num_views:
            videos = videos[:, :num_views]

        b, v, t_frames, c, h_img, w_img = videos.shape
        flat = videos.reshape(b * v, t_frames, c, h_img, w_img)
        # Fast (torchvision) video processor on-device, do_rescale=False (frames already in [0, 1]).
        video_pixels = processor(
            videos=list(flat),
            return_tensors="pt",
            device=encoder.device,
            do_rescale=False,
        )["pixel_values_videos"]  # [B*V, T, C, H, W]

        tubelet_size = self.tubelet_size
        with torch.no_grad():
            video_embeddings = encoder.get_vision_features(pixel_values_videos=video_pixels)
            video_embeddings = self._merge_views(video_embeddings, b, v)

        # num_video_frames raw frames -> t_enc_total temporal positions after tubelet compression.
        t_enc_total = self.config.num_video_frames // tubelet_size
        min_positions = 2
        if t_enc_total < min_positions:
            zero_shape = (video_embeddings.shape[0],) if reduction == "none" else ()
            return torch.zeros(zero_shape, device=video_embeddings.device)

        # Shift-by-one JEPA split: input_states = positions 0..T-2, gt_states = positions 1..T-1.
        t_enc_ctx = t_enc_total - 1
        tokens_per_frame = video_embeddings.shape[1] // t_enc_total
        if self.config.causal_world_model_context:
            # The shared pass above lets bidirectional attention leak future frames into the context
            # positions used as predictor input. Recompute input_states causally instead; gt_states
            # keeps the full-pass embeddings (a target encoder seeing full context is fine).
            with torch.no_grad():
                input_states = self._causal_video_embeddings(encoder, video_pixels, tubelet_size, t_enc_ctx)
                input_states = self._merge_views(input_states, b, v)
        else:
            input_states = video_embeddings[:, : tokens_per_frame * t_enc_ctx, :]
        gt_states = video_embeddings[:, tokens_per_frame:, :]

        expected_actions = t_enc_ctx * self.config.num_action_tokens_per_timestep
        if action_tokens.shape[1] < expected_actions:
            pad = action_tokens[:, -1:].repeat(1, expected_actions - action_tokens.shape[1], 1)
            action_tokens = torch.cat([action_tokens, pad], dim=1)

        predicted_states = predictor(
            input_states.float(),
            action_tokens[:, :expected_actions].float(),
        )
        if reduction == "none":
            # Per-sample loss (B,): mean over all non-batch dims (tokens, feature).
            elementwise = F.l1_loss(predicted_states, gt_states.float(), reduction="none")
            return elementwise.mean(dim=tuple(range(1, elementwise.ndim)))
        return F.l1_loss(predicted_states, gt_states.float(), reduction="mean")

    # ---- Action head -------------------------------------------------------

    def _action_loss(
        self,
        embodied_action_tokens: Tensor,
        actions: Tensor,
        state: Tensor | None,
        action_is_pad: Tensor | None,
        reduction: str = "mean",
    ) -> Tensor:
        """Compute the flow-matching action-head loss over `repeated_diffusion_steps` noise draws.

        Args:
            embodied_action_tokens: Conditioning tokens from the backbone.
            actions: Ground-truth action chunk ``[B, T, action_dim]``.
            state: Current state ``[B, 1, state_dim]``, or None.
            action_is_pad: Padding mask ``[B, T]``, or None.
            reduction: ``"mean"`` for the scalar loss, ``"none"`` for a per-sample loss ``(B,)``.

        Returns:
            The action-head loss.
        """
        device_type = next(self.parameters()).device.type
        with _autocast(device_type, torch.float32):
            r = self.config.repeated_diffusion_steps
            horizon = self.config.chunk_size
            b = embodied_action_tokens.shape[0]
            actions_target = actions[:, -horizon:, :].to(torch.float32).repeat(r, 1, 1)
            # The action head's parameters are float32 while the backbone runs in `torch_dtype`,
            # so its inputs are cast here rather than left to a global autocast the way LeRobot
            # does. `_predict_action` already feeds the head float32; this keeps the two paths
            # consistent and lets training run without AMP (e.g. on CPU).
            embodied = embodied_action_tokens.to(torch.float32).repeat(r, 1, 1)
            state_rep = state.to(torch.float32).repeat(r, 1, 1) if state is not None else None
            pad_rep = action_is_pad[:, -horizon:].repeat(r, 1) if action_is_pad is not None else None
            loss = self.action_model(embodied, actions_target, state_rep, pad_rep, reduction=reduction)
            if reduction == "none":
                # `.repeat(r, 1, 1)` tiles as [rep0(b0..b_{B-1}), rep1(...), ...] -> (r, B).
                return loss.view(r, b).mean(dim=0)
            return loss

    # ---- Batch adaptation --------------------------------------------------

    @staticmethod
    def _image_keys(batch: dict[str, Any]) -> list[str]:
        """List the camera keys present in a batch, excluding padding masks.

        Args:
            batch: Flattened observation dict.

        Returns:
            The image keys, in batch order.

        Raises:
            ValueError: If the batch carries no image feature.
        """
        keys = [key for key in Observation.get_flattened_keys(batch, IMAGES) if "is_pad" not in key]
        keys = [key for key in keys if batch.get(key) is not None]
        if not keys:
            msg = "VLA-JEPA requires at least one image feature."
            raise ValueError(msg)
        return keys

    def _prepare_model_inputs(self, batch: dict[str, Any], *, training: bool) -> dict[str, Any]:
        """Convert a Studio batch to the model's batched, on-device inputs.

        Everything stays on the batch device; only the Qwen message assembly regroups the current
        frame per sample, since the chat template takes one image per content entry.

        Args:
            batch: Flattened observation dict with ``images.*``, ``state``, ``action`` and ``task``.
            training: Whether the world-model videos should be assembled.

        Returns:
            Keyword arguments for :meth:`_forward_native` / :meth:`_predict_action`.
        """
        image_keys = self._image_keys(batch)
        batch_size = batch[image_keys[0]].shape[0]

        # Current-frame image per view ([B, C, H, W]); regroup per sample for the Qwen messages.
        # Resize to `resize_images_to` as `_predict_action` does, so training and inference feed Qwen
        # the same resolution and native frames (e.g. 720x1280) do not blow up the patch count.
        resize_hw = tuple(self.config.resize_images_to) if self.config.resize_images_to else None
        video_dims = 5
        frames = []
        for key in image_keys:
            t = batch[key]
            if t.ndim == video_dims:  # [B, T, C, H, W] -> current observation (delta=0)
                t = t[:, 0]
            px = self.qwen.to_pixel_values(t)  # [B, C, H, W]
            if resize_hw is not None and tuple(px.shape[-2:]) != resize_hw:
                px = F.interpolate(px.float(), size=resize_hw, mode="area")
            frames.append(px)
        images = [[frame[b] for frame in frames] for b in range(batch_size)]

        inputs: dict[str, Any] = {
            "images": images,
            "instructions": self._instructions(batch.get(TASK), batch_size),
        }

        # Videos [B, V, T, C, H, W]: only assembled while training, when the world model uses them.
        if self.config.enable_world_model and training:
            inputs["videos"] = self._prepare_videos(batch, image_keys)

        actions = batch.get(ACTION)
        if actions is not None:
            inputs["actions"] = (actions.unsqueeze(1) if actions.ndim == 2 else actions).float()  # noqa: PLR2004
            if (pad := batch.get(ACTION_IS_PAD)) is not None:
                inputs["action_is_pad"] = pad

        state = batch.get(STATE)
        if state is not None:
            if state.ndim > 2:  # noqa: PLR2004
                # Deltas are forward-looking here, so index 0 is the current observation, not -1.
                state = state[:, 0, :]
            inputs["state"] = (state.unsqueeze(1) if state.ndim == 2 else state).float()  # noqa: PLR2004

        return inputs

    @staticmethod
    def _instructions(tasks: Any, batch_size: int) -> list[str]:  # noqa: ANN401
        """Normalize the task field into one instruction string per sample.

        Args:
            tasks: The batch's task entry: None, a single string, or a sequence of strings.
            batch_size: Number of samples in the batch.

        Returns:
            One instruction per sample.
        """
        if tasks is None:
            return ["Execute the robot action."] * batch_size
        if isinstance(tasks, str):
            return [tasks] * batch_size
        return list(tasks)

    def _prepare_videos(self, batch: dict[str, Any], image_keys: list[str]) -> Tensor:
        """Stack the per-camera frame windows into a single world-model video tensor.

        A single stacked ``[B, V, T, C, H, W]`` tensor needs one spatial size for every view, and
        cameras can differ (base 480x640 vs wrist 720x1280). Resize to `resize_images_to`, else to
        the first view's size (a no-op for single-resolution datasets). The V-JEPA video processor
        handles the final resize to the encoder resolution.

        Args:
            batch: Flattened observation dict.
            image_keys: Camera keys to stack.

        Returns:
            Video tensor of shape ``[B, V, T, C, H, W]``.
        """
        image_dims = 4
        views = [batch[k].unsqueeze(1) if batch[k].ndim == image_dims else batch[k] for k in image_keys]
        target_hw = tuple(self.config.resize_images_to) if self.config.resize_images_to else tuple(views[0].shape[-2:])
        resized = []
        for view in views:
            if tuple(view.shape[-2:]) != target_hw:
                b, t, c = view.shape[0], view.shape[1], view.shape[2]
                view = F.interpolate(  # noqa: PLW2901
                    view.reshape(b * t, c, view.shape[3], view.shape[4]).float(),
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, t, c, target_hw[0], target_hw[1])
            resized.append(view)
        return self.qwen.to_pixel_values(torch.stack(resized, dim=1))

    # ---- Native forward / prediction ---------------------------------------

    def _forward_native(
        self,
        images: list[list[Tensor]],
        instructions: list[str],
        videos: Tensor | None = None,
        actions: Tensor | None = None,
        state: Tensor | None = None,
        action_is_pad: Tensor | None = None,
        reduction: str = "mean",
    ) -> dict[str, Tensor]:
        """Run the native forward: Qwen encode, optional world-model loss, optional action loss.

        Args:
            images: Per-sample, per-view image tensors.
            instructions: Per-sample language instruction.
            videos: World-model video tensor ``[B, V, T, C, H, W]``, or None.
            actions: Ground-truth action chunk, or None for the world-model loss alone.
            state: Current state ``[B, 1, state_dim]``, or None.
            action_is_pad: Padding mask ``[B, T]``, or None.
            reduction: ``"mean"`` for scalar losses, ``"none"`` for per-sample losses.

        Returns:
            Dict with ``wm_loss`` and, when actions are given, ``action_loss``.
        """
        embodied_action_tokens, action_tokens = self._encode_qwen(
            images,
            instructions,
            need_action_tokens=self.config.enable_world_model,
        )

        if self.config.enable_world_model and videos is not None and action_tokens is not None:
            wm_loss = self._world_model_loss(videos, action_tokens, reduction=reduction)
        else:
            zero_shape = (embodied_action_tokens.shape[0],) if reduction == "none" else ()
            wm_loss = torch.zeros(zero_shape, device=embodied_action_tokens.device)

        if actions is None:
            return {"wm_loss": wm_loss}

        action_loss = self._action_loss(embodied_action_tokens, actions, state, action_is_pad, reduction=reduction)
        return {"action_loss": action_loss, "wm_loss": wm_loss * self.config.world_model_loss_weight}

    @torch.no_grad()
    def _predict_action(
        self,
        images: list[list[Tensor]],
        instructions: list[str],
        state: Tensor | None = None,
    ) -> Tensor:
        """Predict an action chunk from the current observation.

        Args:
            images: Per-sample, per-view float [0, 1] image tensors.
            instructions: Per-sample language instruction.
            state: Current state ``[B, 1, state_dim]``, or None.

        Returns:
            Action chunk of shape ``[B, chunk_size, action_dim]``.
        """
        if self.config.resize_images_to is not None:
            height, width = self.config.resize_images_to
            images = [
                [F.interpolate(img[None], size=(height, width), mode="area")[0] for img in views] for views in images
            ]

        embodied_action_tokens, _ = self._encode_qwen(images, instructions, need_action_tokens=False)
        return self.action_model.predict_action(
            embodied_action_tokens.float(),
            state.float() if state is not None else None,
        )

    # ---- Studio Model contract ---------------------------------------------

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]] | Tensor:
        """Compute the training loss or predict an action chunk, depending on the mode.

        Args:
            batch: Preprocessed batch dict.

        Returns:
            Training mode: tuple of (loss, loss dict). Eval mode: the predicted action chunk.
        """
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute the training loss: flow-matching action loss plus the world-model loss.

        Args:
            batch: Preprocessed batch dict, which must contain ground-truth actions.

        Returns:
            Tuple of (loss tensor with grad, loss dict with ``loss``, ``action_loss`` and
            ``wm_loss``).

        Raises:
            ValueError: If the batch carries no ground-truth action.
        """
        if batch.get(ACTION) is None:
            msg = "VLA-JEPA's training loss requires ground-truth actions in the batch."
            raise ValueError(msg)

        outputs = self._forward_native(**self._prepare_model_inputs(batch, training=True))
        action_loss = outputs["action_loss"]
        wm_loss = outputs["wm_loss"]
        loss = action_loss + wm_loss
        # Detached tensors, not `.item()` floats: see Model.compute_loss docstring.
        return loss, {
            "loss": loss.detach(),
            "action_loss": action_loss.detach(),
            "wm_loss": wm_loss.detach(),
        }

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute the validation loss: MSE between predicted and ground-truth actions.

        Runs the full flow-matching integration and compares against the ground truth, which is
        deterministic, unlike the stochastic training loss.

        Args:
            batch: Preprocessed batch dict containing ground-truth actions.

        Returns:
            Tuple of (MSE loss tensor, loss dict with a ``"loss"`` key).

        Raises:
            ValueError: If the batch carries no ground-truth action.
        """
        gt_actions = batch.get(ACTION)
        if gt_actions is None:
            msg = "VLA-JEPA's validation loss requires ground-truth actions in the batch."
            raise ValueError(msg)

        predicted = self.predict_action_chunk(batch)
        action_dim = min(gt_actions.shape[-1], predicted.shape[-1])
        min_len = min(gt_actions.shape[1], predicted.shape[1])
        loss = F.mse_loss(
            predicted[:, :min_len, :action_dim],
            gt_actions[:, :min_len, :action_dim].to(predicted.dtype),
        )
        return loss, {"loss": loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any]) -> Tensor:
        """Predict a chunk of actions from an observation batch.

        Args:
            batch: Preprocessed batch dict.

        Returns:
            Action chunk of shape ``[B, chunk_size, action_dim]``, in normalized action space.
        """
        inputs = self._prepare_model_inputs(batch, training=False)
        return self._predict_action(inputs["images"], inputs["instructions"], inputs.get("state"))

    @property
    def reward_delta_indices(self) -> None:
        """Return reward indices.

        Returns:
            None, as rewards are not used by this policy.
        """
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Get indices of actions relative to the current timestep.

        Returns:
            One index per step of the predicted chunk.
        """
        return self.config.action_delta_indices

    @property
    def observation_delta_indices(self) -> list[int]:
        """Get indices of observations relative to the current timestep.

        Returns:
            ``[0]`` without the world model, otherwise the video frame window.
        """
        return self.config.observation_delta_indices

    def set_dataset_stats(self, dataset_stats: dict[str, Any]) -> None:
        """Update the dataset statistics used to report feature dimensions.

        Args:
            dataset_stats: Dataset normalization statistics.
        """
        self._dataset_stats = dataset_stats

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Iterate over the parameters that receive gradients.

        Returns:
            Iterator over trainable parameters.
        """
        return (p for p in self.parameters() if p.requires_grad)
