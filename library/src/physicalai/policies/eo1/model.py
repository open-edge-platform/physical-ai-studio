# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""EO-1 model implementation.

Ported from LeRobot's ``lerobot.policies.eo1.modeling_eo1``: this module merges the native
``EO1VisionFlowMatchingModel`` and the batch-adaptation half of ``EO1Policy`` into a single Studio
:class:`~physicalai.policies.base.Model`.

The architecture is a Qwen2.5-VL backbone whose prompt reserves one placeholder token for the robot
state and `chunk_size` placeholder tokens for the action chunk. The state placeholder is filled with
a linear projection of the state; the action placeholders are filled with a projection of the noisy
action chunk plus its flow-matching timestep. The hidden states read back at the action positions
are projected to the velocity field the flow head is trained on.

Submodule attribute names deliberately match LeRobot's (``vlm_backbone``, ``state_proj``,
``action_in_proj``, ``action_out_proj``, ``action_time_mlp_in``, ``action_time_mlp_out``) so
published EO-1 checkpoints map onto this module after the leading ``model.`` prefix is stripped.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint
from torch import Tensor, nn

from physicalai.data.observation import ACTION, EXTRA, IMAGES, STATE, TASK, Observation
from physicalai.policies.base import Model
from physicalai.policies.eo1.components.flow_matching import (
    create_sinusoidal_pos_embedding,
    euler_integrate,
    pad_vector,
    sample_noise,
    sample_time_beta,
)
from physicalai.policies.eo1.components.qwen_interface import EO1QwenInterface

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from .config import EO1Config

logger = logging.getLogger(__name__)

ACTION_IS_PAD = EXTRA + ".action_is_pad"

_STATE_WITH_TIME_DIM = 2
_IMAGE_WITH_TIME_DIM = 4
_ACTION_CHUNK_NDIM = 3
_DEFAULT_TASK = "Execute the robot action."


def _lazy_import_transformers() -> tuple:
    """Lazy import the transformers symbols the EO-1 backbone needs.

    Returns:
        Tuple of (Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, torch_compilable_check).

    Raises:
        ImportError: If transformers is not installed.
    """
    try:
        from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration  # noqa: PLC0415
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig  # noqa: PLC0415
        from transformers.utils import torch_compilable_check  # noqa: PLC0415
    except ImportError as e:
        msg = "EO-1 requires the transformers library.\n\nInstall with:\n    uv pip install 'physicalai-train[eo1]'"
        raise ImportError(msg) from e
    else:
        return Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, torch_compilable_check


def _activation(name: str) -> nn.Module:
    """Resolve an activation module by name through ``transformers.ACT2FN``.

    Args:
        name: Activation name, e.g. "linear", "gelu", "silu".

    Returns:
        A new activation module instance.

    Raises:
        ImportError: If transformers is not installed.
    """
    try:
        from transformers.activations import ACT2FN  # noqa: PLC0415
    except ImportError as e:
        msg = "EO-1 requires the transformers library.\n\nInstall with:\n    uv pip install 'physicalai-train[eo1]'"
        raise ImportError(msg) from e
    return ACT2FN[name]


class EO1ActionProjector(nn.Sequential):
    """Multi-layer perceptron mapping backbone hidden states to the flow-matching velocity.

    Kept as a :class:`torch.nn.Sequential` subclass with LeRobot's layer ordering so the published
    checkpoints' ``action_out_proj.<i>.weight`` keys line up.

    Args:
        in_channels: Input width, the backbone hidden size.
        out_channels: Output width, ``max_action_dim``.
        num_layers: Number of linear layers.
        activation_layer: Activation inserted between them, resolved through ``ACT2FN``.
        bias: Whether the linear layers carry a bias.
        dtype: Parameter dtype. The flow head stays in fp32.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        activation_layer: str = "linear",
        *,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Build the projector layers.

        Args:
            in_channels: Input width, the backbone hidden size.
            out_channels: Output width, ``max_action_dim``.
            num_layers: Number of linear layers.
            activation_layer: Activation inserted between them.
            bias: Whether the linear layers carry a bias.
            dtype: Parameter dtype.
        """
        layers: list[nn.Module] = []
        in_dim = in_channels
        hidden_channels = [in_dim] * (num_layers - 1) + [out_channels]
        for hidden_dim in hidden_channels[:-1]:
            layers.extend((
                nn.Linear(in_dim, hidden_dim, bias=bias, dtype=dtype),
                _activation(activation_layer),
            ))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_channels[-1], bias=bias, dtype=dtype))
        super().__init__(*layers)

    @property
    def dtype(self) -> torch.dtype:
        """Parameter dtype of the projector's first layer."""
        return cast("nn.Linear", self[0]).weight.dtype


class EO1Model(Model):
    """EO-1 vision-language-action model.

    Unlike the other Studio families, this model takes its :class:`EO1Config` directly instead of
    one keyword argument per field: the flow head is driven end to end by the config, and the
    Qwen2.5-VL conversation interface already takes the config object.

    Args:
        config: Policy configuration.
        dataset_stats: Dataset normalization statistics, kept for reporting the true feature
            dimensions. Optional.
    """

    # transformers is imported lazily, so the backbone's real class is not nameable here; without
    # this annotation every attribute access resolves through `nn.Module.__getattr__`.
    vlm_backbone: Any

    def __init__(self, config: EO1Config, dataset_stats: dict[str, Any] | None = None) -> None:
        """Build the Qwen backbone and the flow-matching head.

        Args:
            config: Policy configuration.
            dataset_stats: Dataset normalization statistics. Optional.
        """
        super().__init__()
        self.config = config
        self._dataset_stats = dataset_stats or {}

        self.vlm_backbone = self._build_backbone(config)
        self.qwen = EO1QwenInterface(config)
        self.qwen.maybe_resize_embeddings(self.vlm_backbone)

        self.hidden_size: int = self.vlm_backbone.config.text_config.hidden_size
        # The flow head stays in fp32 regardless of the backbone dtype: it is small, and denoising
        # is numerically sensitive.
        self.state_proj = nn.Linear(config.max_state_dim, self.hidden_size, dtype=torch.float32)
        self.action_in_proj = nn.Linear(config.max_action_dim, self.hidden_size, dtype=torch.float32)
        self.action_out_proj = EO1ActionProjector(
            self.hidden_size,
            config.max_action_dim,
            config.num_action_layers,
            config.action_act,
            dtype=torch.float32,
        )
        self.action_time_mlp_in = nn.Linear(self.hidden_size * 2, self.hidden_size, dtype=torch.float32)
        self.action_time_mlp_out = nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.float32)

        self.gradient_checkpointing_enabled = False
        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable()

    @staticmethod
    def _build_backbone(config: EO1Config) -> Any:  # noqa: ANN401
        """Build the Qwen2.5-VL backbone.

        With no `vlm_config`, the pretrained `vlm_base` weights are downloaded, which is the right
        start for training a new policy. With one - the case when a published EO-1 checkpoint is
        being loaded - the architecture is rebuilt from the stored config with random weights, and
        the checkpoint fills them in. That avoids downloading multi-gigabyte backbone weights only
        to overwrite them.

        Args:
            config: Policy configuration.

        Returns:
            The Qwen backbone.
        """
        qwen_cls, qwen_config_cls, _ = _lazy_import_transformers()

        if config.vlm_config is None:
            return qwen_cls.from_pretrained(
                config.vlm_base,
                dtype=config.dtype,
                attn_implementation=config.attn_implementation,
            )

        config_dict = dict(config.vlm_config)
        if config.attn_implementation is not None:
            config_dict["attn_implementation"] = config.attn_implementation
        backbone_config = qwen_config_cls(**config_dict)
        dtype = backbone_config.dtype if config.dtype == "auto" else config.dtype
        # `_from_config` is the only entry point transformers exposes for building an
        # architecture without weights; LeRobot's EO-1 uses it for the same reason.
        return qwen_cls._from_config(backbone_config, dtype=dtype)  # noqa: SLF001

    # ---- Training utilities -------------------------------------------------

    def get_input_embeddings(self) -> nn.Module:
        """Return the backbone's token embedding table.

        Returns:
            The embedding module.
        """
        return self.vlm_backbone.get_input_embeddings()

    def flow_head_autocast_context(self) -> contextlib.AbstractContextManager:
        """Return the context the flow head's projections run under.

        Returns:
            An autocast-disabling context when ``force_fp32_autocast`` is set, otherwise a no-op.
        """
        if self.config.force_fp32_autocast:
            return torch.autocast(device_type=self.state_proj.weight.device.type, enabled=False)
        return contextlib.nullcontext()

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing on the backbone and the flow-head computations."""
        self.gradient_checkpointing_enabled = True
        self.vlm_backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("Enabled gradient checkpointing for EO1Model")

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on the backbone and the flow-head computations."""
        self.gradient_checkpointing_enabled = False
        self.vlm_backbone.gradient_checkpointing_disable()
        logger.info("Disabled gradient checkpointing for EO1Model")

    def _apply_checkpoint(self, func: Callable, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Run `func` under gradient checkpointing when it is enabled and gradients are needed.

        Args:
            func: The computation to run.
            *args: Positional arguments for `func`.
            **kwargs: Keyword arguments for `func`.

        Returns:
            Whatever `func` returns.
        """
        if self.gradient_checkpointing_enabled and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                func,
                *args,
                use_reentrant=False,
                preserve_rng_state=False,
                **kwargs,
            )
        return func(*args, **kwargs)

    def sample_time(self, bsize: int, device: torch.device) -> Tensor:
        """Draw flow-matching timesteps for a batch.

        Args:
            bsize: Batch size.
            device: Device to allocate on.

        Returns:
            Float32 timesteps of shape ``(bsize,)``.
        """
        return sample_time_beta(
            bsize,
            device,
            alpha=self.config.time_sampling_beta_alpha,
            beta=self.config.time_sampling_beta_beta,
            scale=self.config.time_sampling_scale,
            offset=self.config.time_sampling_offset,
        )

    def get_placeholder_mask(
        self,
        input_ids: Tensor,
        inputs_embeds: Tensor,
        state_features: Tensor | None = None,
        action_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Locate the EO-1 state and action placeholder positions, Qwen's multimodal-mask style.

        Args:
            input_ids: Token ids of shape ``(B, L)``.
            inputs_embeds: Token embeddings of shape ``(B, L, hidden)``.
            state_features: Projected state, checked against the number of state placeholders.
            action_features: Projected action chunk, checked against the number of action
                placeholders.

        Returns:
            Tuple of (state mask, action mask), both broadcast to the shape of `inputs_embeds`.
        """
        _, _, torch_compilable_check = _lazy_import_transformers()

        special_state_mask = input_ids == self.qwen.state_token_id
        special_action_mask = input_ids == self.qwen.action_token_id

        n_state_tokens = special_state_mask.sum()
        special_state_mask = special_state_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if state_features is not None:
            torch_compilable_check(
                inputs_embeds[special_state_mask].numel() == state_features.numel(),
                f"State features and state tokens do not match, tokens: {n_state_tokens}, "
                f"features: {state_features.shape[0]}",
            )

        n_action_tokens = special_action_mask.sum()
        special_action_mask = special_action_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if action_features is not None:
            torch_compilable_check(
                inputs_embeds[special_action_mask].numel() == action_features.numel(),
                f"Action features and action tokens do not match, tokens: {n_action_tokens}, "
                f"features: {action_features.shape[0]}",
            )

        return special_state_mask, special_action_mask

    def embed_prefix(self, input_ids: Tensor, states: Tensor) -> Tensor:
        """Embed the prompt and scatter the projected robot state into its placeholder.

        Args:
            input_ids: Token ids of shape ``(B, L)``.
            states: Padded robot state of shape ``(B, max_state_dim)``.

        Returns:
            Token embeddings of shape ``(B, L, hidden)``.
        """

        def input_embed_func(input_ids: Tensor) -> Tensor:
            return self.get_input_embeddings()(input_ids)

        inputs_embeds = self._apply_checkpoint(input_embed_func, input_ids)

        def state_proj_func(states: Tensor) -> Tensor:
            with self.flow_head_autocast_context():
                states = states.to(dtype=self.state_proj.weight.dtype)
                return self.state_proj(states)

        state_embs = self._apply_checkpoint(state_proj_func, states)
        state_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, state_features=state_embs)
        state_embs = state_embs.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(state_mask, state_embs)

    def embed_suffix(self, timestep: Tensor, noisy_actions: Tensor) -> Tensor:
        """Embed the noisy action chunk together with its flow-matching timestep.

        Args:
            timestep: Timesteps of shape ``(B,)``.
            noisy_actions: Noisy action chunk of shape ``(B, chunk_size, max_action_dim)``.

        Returns:
            Action-token embeddings of shape ``(B, chunk_size, hidden)``.
        """

        def action_proj_func(noisy_actions: Tensor) -> Tensor:
            with self.flow_head_autocast_context():
                noisy_actions = noisy_actions.to(dtype=self.action_in_proj.weight.dtype)
                return self.action_in_proj(noisy_actions)

        action_embs = self._apply_checkpoint(action_proj_func, noisy_actions)
        time_embs = create_sinusoidal_pos_embedding(
            timestep,
            self.hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=action_embs.device,
        )
        time_embs = time_embs.to(dtype=action_embs.dtype)
        time_embs = time_embs[:, None, :].expand_as(action_embs)
        action_time_embs = torch.cat([action_embs, time_embs], dim=2)

        def mlp_func(action_time_embs: Tensor) -> Tensor:
            with self.flow_head_autocast_context():
                action_time_embs = action_time_embs.to(dtype=self.action_time_mlp_in.weight.dtype)
                action_time_embs = self.action_time_mlp_in(action_time_embs)
                action_time_embs = F.silu(action_time_embs)
                return self.action_time_mlp_out(action_time_embs)

        return self._apply_checkpoint(mlp_func, action_time_embs)

    def _flow_matching_loss(  # noqa: PLR0914
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        mm_token_type_ids: Tensor,
        states: Tensor,
        action: Tensor,
        action_is_pad: Tensor | None = None,
    ) -> Tensor:
        """Run the training forward pass and compute the flow-matching loss.

        Args:
            input_ids: Token ids of shape ``(B, L)``.
            attention_mask: Attention mask of shape ``(B, L)``.
            pixel_values: Flattened image patches for the vision tower.
            image_grid_thw: Patch grid sizes per image.
            mm_token_type_ids: Multimodal token type ids of shape ``(B, L)``.
            states: Padded robot state of shape ``(B, max_state_dim)``.
            action: Padded ground-truth chunk of shape ``(B, chunk_size, max_action_dim)``.
            action_is_pad: Per-step padding mask of shape ``(B, chunk_size)``. Required when
                `supervise_padding_actions` is False.

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: If `action_is_pad` is missing while padded actions must be excluded.
        """
        # 1. Build the EO-1 prefix with the state placeholder resolved.
        inputs_embeds = self.embed_prefix(input_ids, states=states)

        # 2. Sample the flow-matching target and replace the action placeholders.
        time = self.sample_time(action.shape[0], inputs_embeds.device)
        noise = sample_noise(action.shape, inputs_embeds.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * action
        u_t = noise - action
        action_time_embs = self.embed_suffix(time, x_t)
        _, action_mask = self.get_placeholder_mask(input_ids, inputs_embeds, action_features=action_time_embs)
        action_time_embs = action_time_embs.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(action_mask, action_time_embs)

        # 3. Optionally drop padded action tokens from backbone attention.
        attention_mask = attention_mask.to(inputs_embeds.device)
        padded_actions: Tensor | None = None
        if not self.config.supervise_padding_actions:
            if action_is_pad is None:
                msg = (
                    "`supervise_padding_actions=False` needs an `action_is_pad` mask in the batch to "
                    "know which action rows to exclude."
                )
                raise ValueError(msg)
            padded_actions = action_is_pad.to(device=inputs_embeds.device, dtype=torch.bool)
            action_token_mask = action_mask[..., 0]
            action_padding_mask = torch.zeros_like(action_token_mask)
            action_padding_mask = action_padding_mask.masked_scatter(action_token_mask, padded_actions.reshape(-1))
            attention_mask = attention_mask.masked_fill(action_padding_mask, 0)

        # 4. Run the Qwen backbone on the fused EO-1 sequence.
        def vlm_forward_func(
            input_ids: Tensor,
            attention_mask: Tensor,
            inputs_embeds: Tensor,
            pixel_values: Tensor,
            image_grid_thw: Tensor,
            mm_token_type_ids: Tensor,
        ) -> Tensor:
            outputs = self.vlm_backbone.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            return outputs.last_hidden_state

        hidden_states = self._apply_checkpoint(
            vlm_forward_func,
            input_ids,
            attention_mask,
            inputs_embeds,
            pixel_values,
            image_grid_thw,
            mm_token_type_ids,
        )
        action_hidden_states = hidden_states[action_mask[..., 0]]

        # 5. Project the action-token hidden states back to the flow target space.
        def action_out_proj_func(action_hidden_states: Tensor) -> Tensor:
            with self.flow_head_autocast_context():
                action_hidden_states = action_hidden_states.to(dtype=self.action_out_proj.dtype)
                return self.action_out_proj(action_hidden_states)

        v_t = self._apply_checkpoint(action_out_proj_func, action_hidden_states)
        v_t = v_t.reshape(u_t.shape).to(dtype=u_t.dtype)
        losses = F.mse_loss(u_t, v_t, reduction="none")

        # 6. Apply the configured supervision mask and reduce.
        if not self.config.supervise_padding_action_dims:
            losses = losses[..., : self.config.action_dim]
        if padded_actions is not None:
            losses = losses[~padded_actions]

        return losses.mean()

    @torch.no_grad()
    def _sample_actions(  # noqa: PLR0914
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        mm_token_type_ids: Tensor,
        states: Tensor,
    ) -> Tensor:
        """Denoise an action chunk from the rollout prompt.

        The prompt prefix is encoded once and its KV cache reused across every Euler step, so each
        step only runs the backbone over the `chunk_size` action tokens.

        Args:
            input_ids: Left-padded token ids of shape ``(B, L)``.
            attention_mask: Attention mask of shape ``(B, L)``.
            pixel_values: Flattened image patches for the vision tower.
            image_grid_thw: Patch grid sizes per image.
            mm_token_type_ids: Multimodal token type ids of shape ``(B, L)``.
            states: Padded robot state of shape ``(B, max_state_dim)``.

        Returns:
            Action chunk of shape ``(B, chunk_size, max_action_dim)``.

        Raises:
            ValueError: If the action placeholders are not a contiguous, batch-aligned span of
                exactly `chunk_size` tokens.
        """
        chunk_size = self.config.chunk_size

        # 1. Resolve the left-padded rollout prompt and locate the action span.
        inputs_embeds = self.embed_prefix(input_ids, states=states).clone()
        _, action_placeholder_mask = self.get_placeholder_mask(input_ids, inputs_embeds)
        action_mask = action_placeholder_mask[..., 0]
        token_counts = action_mask.sum(dim=1)
        if not torch.all(token_counts == chunk_size):
            msg = f"Each sample must contain exactly {chunk_size} action tokens, got {token_counts.tolist()}."
            raise ValueError(msg)
        if action_mask.ne(action_mask[:1]).any():
            msg = "Batch inference expects all samples to share the same action token mask after left padding."
            raise ValueError(msg)
        act_start = int(action_mask[0].to(torch.int64).argmax().item())
        act_end = act_start + chunk_size
        if not torch.all(action_mask[:, act_start:act_end]):
            msg = "Action tokens must form a contiguous chunk of length chunk_size."
            raise ValueError(msg)
        act_slice = slice(act_start, act_end)

        # 2. Encode the fixed prefix once and cache its KV state.
        batch_size = input_ids.shape[0]
        device = inputs_embeds.device
        attention_mask = attention_mask.to(device)
        mm_token_type_ids = mm_token_type_ids.to(device)
        position_ids, _ = self.vlm_backbone.model.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )
        position_ids = position_ids.to(device)

        outputs = self.vlm_backbone.model(
            input_ids=input_ids[:, :act_start],
            attention_mask=attention_mask[:, :act_start],
            position_ids=position_ids[..., :act_start],
            inputs_embeds=inputs_embeds[:, :act_start],
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids[:, :act_start],
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values

        x_t = sample_noise((batch_size, chunk_size, self.config.max_action_dim), device).to(
            dtype=self.action_in_proj.weight.dtype,
        )

        # 3. Denoise only the action chunk while keeping the prefix cache invariant.
        def denoise_fn(input_x_t: Tensor, current_timestep: Tensor) -> Tensor:
            action_time_embs = self.embed_suffix(current_timestep, input_x_t)
            inputs_embeds[:, act_slice] = action_time_embs.to(inputs_embeds.dtype)

            past_key_values.crop(act_start)
            step_outputs = self.vlm_backbone.model(
                attention_mask=attention_mask[:, :act_end],
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds[:, act_slice],
                position_ids=position_ids[..., act_slice],
                use_cache=True,
                return_dict=True,
            )
            with self.flow_head_autocast_context():
                hidden_states = step_outputs.last_hidden_state[:, :chunk_size]
                hidden_states = hidden_states.to(dtype=self.action_out_proj.dtype)
                v_t = self.action_out_proj(hidden_states)
            return v_t.reshape(input_x_t.shape).to(input_x_t.dtype)

        return euler_integrate(denoise_fn, x_t, self.config.num_denoise_steps)

    # ---- Batch adaptation ---------------------------------------------------

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
            msg = "EO-1 requires at least one image feature."
            raise ValueError(msg)
        return keys

    @staticmethod
    def _tasks(tasks: Any, batch_size: int) -> list[str]:  # noqa: ANN401
        """Normalize the task field into one instruction string per sample.

        Args:
            tasks: The batch's task entry: None, a single string, or a sequence of strings.
            batch_size: Number of samples in the batch.

        Returns:
            One instruction per sample.
        """
        if tasks is None:
            return [_DEFAULT_TASK] * batch_size
        if isinstance(tasks, str):
            return [tasks] * batch_size
        return [str(task) for task in tasks]

    def _prepare_model_inputs(self, batch: dict[str, Any], *, training: bool) -> dict[str, Any]:
        """Convert a Studio batch into the tokenized, on-device inputs the model consumes.

        Args:
            batch: Flattened observation dict with ``images.*``, ``state``, ``action`` and ``task``.
            training: Whether this is a supervised batch. Supervised batches are right-padded;
                rollouts are left-padded so the action span sits at the same offset in every sample.

        Returns:
            Keyword arguments for :meth:`_flow_matching_loss` or :meth:`_sample_actions`.

        Raises:
            ValueError: If the batch carries no robot state.
        """
        actions = batch.get(ACTION)
        if actions is not None:
            # Checked before tokenizing, so a mismatched horizon fails fast rather than deep inside
            # the backbone's placeholder-count check.
            self._validate_action_chunk(actions)

        image_keys = self._image_keys(batch)
        batch_size = batch[image_keys[0]].shape[0]

        frames = []
        for key in image_keys:
            frame = batch[key]
            if frame.ndim > _IMAGE_WITH_TIME_DIM:  # [B, T, C, H, W]
                # EO-1 declares no observation deltas, so a time axis only appears with
                # n_obs_steps > 1, where the current frame is the last one.
                frame = frame[:, -1]
            frames.append(frame)
        images = [[frame[index] for frame in frames] for index in range(batch_size)]

        qwen_inputs = self.qwen.build_inputs(
            images,
            self._tasks(batch.get(TASK), batch_size),
            padding_side="right" if training else "left",
        )
        device = self.state_proj.weight.device
        inputs: dict[str, Any] = {key: value.to(device) for key, value in qwen_inputs.items()}

        state = batch.get(STATE)
        if state is None:
            msg = "EO-1 requires a robot state observation."
            raise ValueError(msg)
        if state.ndim > _STATE_WITH_TIME_DIM:
            state = state[:, -1]
        inputs["states"] = pad_vector(state.to(device).float(), self.config.max_state_dim)

        if actions is not None:
            inputs["action"] = pad_vector(actions.to(device).float(), self.config.max_action_dim)
            if (pad := batch.get(ACTION_IS_PAD)) is not None:
                inputs["action_is_pad"] = pad.to(device)

        return inputs

    def _validate_action_chunk(self, actions: Tensor) -> None:
        """Check the ground-truth actions carry exactly the horizon the prompt reserves room for.

        The prompt holds exactly `chunk_size` action placeholder tokens, so a dataset that hands
        back a single action - or a chunk of a different length - would otherwise fail deep inside
        the backbone's placeholder-count check.

        Args:
            actions: Ground-truth actions from the batch.

        Raises:
            ValueError: If the actions are not ``[B, chunk_size, action_dim]``.
        """
        if actions.ndim != _ACTION_CHUNK_NDIM or actions.shape[1] != self.config.chunk_size:
            msg = (
                f"EO-1 expects ground-truth actions of shape [B, chunk_size="
                f"{self.config.chunk_size}, action_dim], got {tuple(actions.shape)}. The dataset must "
                f"deliver an action chunk; `reformat_dataset_to_match_policy` sets that up from "
                f"`action_delta_indices`."
            )
            raise ValueError(msg)

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
        """Compute the flow-matching training loss.

        Args:
            batch: Preprocessed batch dict, which must contain ground-truth actions.

        Returns:
            Tuple of (loss tensor with grad, loss dict with a ``"loss"`` key).

        Raises:
            ValueError: If the batch carries no ground-truth action.
        """
        if batch.get(ACTION) is None:
            msg = "EO-1's training loss requires ground-truth actions in the batch."
            raise ValueError(msg)

        loss = self._flow_matching_loss(**self._prepare_model_inputs(batch, training=True))
        # Detached tensor, not an `.item()` float: see Model.compute_loss docstring.
        return loss, {"loss": loss.detach()}

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute the validation loss: MSE between predicted and ground-truth actions.

        Runs the full flow-matching integration, which is deterministic given the sampled noise,
        rather than reusing the stochastic training loss.

        Args:
            batch: Preprocessed batch dict containing ground-truth actions.

        Returns:
            Tuple of (MSE loss tensor, loss dict with a ``"loss"`` key).

        Raises:
            ValueError: If the batch carries no ground-truth action.
        """
        gt_actions = batch.get(ACTION)
        if gt_actions is None:
            msg = "EO-1's validation loss requires ground-truth actions in the batch."
            raise ValueError(msg)

        predicted = self.predict_action_chunk(batch)
        action_dim = min(gt_actions.shape[-1], predicted.shape[-1])
        min_len = min(gt_actions.shape[1], predicted.shape[1])
        loss = F.mse_loss(
            predicted[:, :min_len, :action_dim],
            gt_actions[:, :min_len, :action_dim].to(predicted.device, predicted.dtype),
        )
        return loss, {"loss": loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any]) -> Tensor:
        """Predict a chunk of actions from an observation batch.

        Args:
            batch: Preprocessed batch dict.

        Returns:
            Action chunk of shape ``[B, chunk_size, action_dim]``, in normalized action space and
            cropped back from the padded ``max_action_dim`` width.
        """
        inputs = self._prepare_model_inputs(batch, training=False)
        inputs.pop("action", None)
        inputs.pop("action_is_pad", None)
        actions = self._sample_actions(**inputs).to(torch.float32)
        return actions[:, :, : self.config.action_dim]

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
    def observation_delta_indices(self) -> None:
        """Get indices of observations relative to the current timestep.

        Returns:
            None, as EO-1 conditions on the current frame only.
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
