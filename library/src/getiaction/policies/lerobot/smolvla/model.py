# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA with XAI.

This module provides a descendent of LeRobot's SmolVLA policy with added explainability (XAI). Some of the original
    code has been moved to this file.
"""

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel, apply_rope
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION
from torch import nn
from torch.nn import functional

if TYPE_CHECKING:
    from collections import deque


class SmolVLMWithExpertModelWithXAI(SmolVLMWithExpertModel):
    """This class replaces the original SmolVLMWithExpertModel and adds extra xAI functionality."""

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,  # noqa: FBT001, FBT002
        train_expert_only: bool = True,  # noqa: FBT001, FBT002
        freeze_vision_encoder: bool = False,  # noqa: FBT001, FBT002
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
    ) -> None:
        """Initializes the SmolVLMWithExpertModel.

        Args:
            model_id: Which model ID to download fron HF hub.
            load_vlm_weights: Download the VLM weights.
            train_expert_only: Determines if the expert should be only be trained.
            freeze_vision_encoder: Freeze the encoder.
            attention_mode: Which attention mode to use (default is self attention)
            num_expert_layers: Number of layers for the expert model.
            num_vlm_layers: Number of VLM layers.
            self_attn_every_n_layers: Self attention per layer.
            expert_width_multiplier: Width modifier for teh expert model.
        """
        super().__init__(
            model_id,
            load_vlm_weights,
            train_expert_only,
            freeze_vision_encoder,
            attention_mode,
            num_expert_layers,
            num_vlm_layers,
            self_attn_every_n_layers,
            expert_width_multiplier,
        )
        self.qk: dict[int, torch.Tensor] = {}

    def eager_attention_forward_with_qk(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create an alternative for eager_attention_forward that also return attention probs (qk).

        Args:
            attention_mask: Attention mask.
            batch_size: Batch size.
            head_dim: Dimensionality of the attention heads.
            query_states: State of the Q.
            key_states: State of the K.
            value_states: State of the V.

        Returns:
            A Tuple containing the attention result and the additional attention probability for XAI.
        """
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size,
            sequence_length,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
        )
        key_states = key_states.reshape(
            batch_size,
            sequence_length,
            num_key_value_heads * num_key_value_groups,
            head_dim,
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size,
            sequence_length,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
        )
        value_states = value_states.reshape(
            batch_size,
            sequence_length,
            num_key_value_heads * num_key_value_groups,
            head_dim,
        )

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min  # -2.3819763e38  # See gemma/modules.py
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output, probs.to(dtype=torch.float32)  # Also return att_probabilities

    def forward_attn_layer(  # noqa: PLR0914
        self,
        model_layers: list[list],
        inputs_embeds: list[torch.Tensor],
        layer_idx: int,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        use_cache: bool = True,  # noqa: FBT001, FBT002
        fill_kv_cache: bool = True,  # noqa: FBT001, FBT002
        past_key_values: dict[int, Any] | None = None,
    ) -> tuple[list[torch.Tensor], dict[Any, Any] | Any]:
        """Override this method to store the qk for each attention layer.

        Args:
            model_layers: Decoding layers of the model.
            inputs_embeds: Input embeddings.
            layer_idx: layer index to return the attention for xAI.
            position_ids: Positional IDs.
            attention_mask: Attentions mask.
            batch_size: Batch size.
            head_dim: Dimensionality of the attention head.
            use_cache: If caching of values should be used.
            fill_kv_cache: If the KV cache should be filled.
            past_key_values: What the past key values are.

        Returns:
            A list of attention outputs and a history of value for caching.
        """
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            hidden_states = layer.input_layernorm(hidden_states)  # noqa: PLW2901

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)  # noqa: PLW2901
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

        # B,L,H,D with L sequence length, H number of heads, D head dim
        # concatenate on the number of embeddings/tokens
        query_states_cat = torch.cat(query_states, dim=1)
        key_states_cat = torch.cat(key_states, dim=1)
        value_states_cat = torch.cat(value_states, dim=1)
        seq_len = query_states_cat.shape[1]
        if seq_len < position_ids.shape[1]:
            _position_ids = position_ids[:, :seq_len]
            _attention_mask = attention_mask[:, :seq_len, :seq_len]
        else:
            _position_ids = position_ids  # noqa: RUF052
            _attention_mask = attention_mask  # noqa: RUF052

        attention_mask_ = _attention_mask
        position_ids_ = _position_ids

        query_states_cat = apply_rope(query_states_cat, position_ids_)
        key_states_cat = apply_rope(key_states_cat, position_ids_)

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache and past_key_values is not None:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states_cat,
                    "value_states": value_states_cat,
                }
            else:
                key_states_cat = torch.cat([past_key_values[layer_idx]["key_states"], key_states_cat], dim=1)
                value_states_cat = torch.cat([past_key_values[layer_idx]["value_states"], value_states_cat], dim=1)

        att_output, att_weights = self.eager_attention_forward_with_qk(
            attention_mask_,
            batch_size,
            head_dim,
            query_states_cat,
            key_states_cat,
            value_states_cat,
        )

        # store qk probs
        if fill_kv_cache:  # This is skipped for the denoising step
            self.qk[layer_idx] = att_weights.detach().cpu()

        return [att_output], past_key_values


class VLAFlowMatchingWithXAI(VLAFlowMatching):
    """This replaces the original VLAFlowMatching implementation."""

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None) -> None:
        """Intialize the VLAFlowMatching class.

        Args:
            config: Config for SmolVLA.
            rtc_processor: The RTC Processor.
        """
        nn.Module.__init__(self)  # Call grandparent instead of parent to prevent redundant initialization
        self.config = config

        # Create custom model
        self.vlm_with_expert = SmolVLMWithExpertModelWithXAI(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim,
            self.vlm_with_expert.config.text_config.hidden_size,
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2,
            self.vlm_with_expert.expert_hidden_size,
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size,
            self.vlm_with_expert.expert_hidden_size,
        )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token],
            dtype=torch.long,
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor


class SmolVLAPolicyWithXAI(SmolVLAPolicy):
    """This augments the original SmolVLAPolicy implementation."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
        # XAI parameters
        layer_idx: int = -1,
        head_idx: int | None = None,
    ) -> None:
        """Descendent of SmolVLA which adds XAI capabilities.

        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of the
                configuration class is used.
            layer_idx: Which layer to display
            head_idx: Which head to display (None means average from all heads) -1 takes the max instead of mean.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatchingWithXAI(config)
        self.reset()

        # Initialize XAI
        self.layer_idx = layer_idx
        self.head_idx = head_idx

        self.image_shapes = {k: v.shape for k, v in self.model.config.image_features.items()}
        self.image_resized_padded_shapes = dict.fromkeys(
            self.model.config.image_features.keys(),
            self.model.config.resize_imgs_with_padding,
        )
        self.image_tile_shapes = {k: [8, 8] for k in self.model.config.image_features}
        self.num_robot_tokens = 1
        self.attention_modes: dict[str, Any] | None = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None) -> torch.Tensor:
        """Create an alternative for select_action that also returns the attention maps.

        Args:
            batch: Batch of data.
            noise: Noise tensor.

        Returns:
              A tenson containing the action.
        """
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues: dict[str, deque] = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        # Also get the weights and task
        index = max(self.model.vlm_with_expert.qk.keys()) + self.layer_idx + 1 if self.layer_idx < 0 else self.layer_idx
        attention_maps = self.model.vlm_with_expert.qk[index]

        self.attention_modes = self._map_attention(attention_maps)
        self.attention_modes["task"] = batch["task"]

        return self._queues[ACTION].popleft()

    @staticmethod
    def _apply_colormap(batch: torch.Tensor, colormap: str = "jet") -> torch.Tensor:
        images_resize = batch.squeeze(1)
        if colormap == "viridis":
            r = torch.clamp(-2.5 * images_resize + 2.5, 0, 1)
            g = torch.clamp(-4 * torch.abs(images_resize - 0.5) + 1.5, 0, 1)
            b = torch.clamp(4 * images_resize - 2, 0, 1)
        elif colormap == "jet":
            r = torch.clamp(1.5 - torch.abs(4 * images_resize - 3), 0, 1)
            g = torch.clamp(1.5 - torch.abs(4 * images_resize - 2), 0, 1)
            b = torch.clamp(1.5 - torch.abs(4 * images_resize - 1), 0, 1)
        else:
            msg = f"Unsupported colormap: {colormap}"
            raise RuntimeError(msg)
        return torch.stack([r, g, b], dim=1)

    def explain(self) -> dict[str, torch.Tensor]:
        """Get the XAI layer.

        Returns:
            An array of xAI output.
        """
        if self.attention_modes is None:
            return {}
        visualizations = {}
        for key, imgs in self.attention_modes["image_att"].items():
            # Add singleton dimension
            images = imgs.unsqueeze(-3)
            # Resize and interpolate
            h, w = self.image_shapes[key][1:]
            images_resize = functional.interpolate(images, size=(h, w), mode="bilinear")
            images_color = self._apply_colormap(images_resize, "jet")
            # Convert to 8 bit image
            images_color = (images_color * 255).type(torch.uint8)
            visualizations[key] = images_color

        # Text
        visualizations["text"] = (self.attention_modes["text_att"] * 255).type(torch.uint8)

        # proprioception
        visualizations["state"] = (self.attention_modes["state_att"] * 255).type(torch.uint8)

        return visualizations

    @staticmethod
    def _draw_text(text: str, image: np.ndarray, width: int, height: int, margin: int = 5) -> None:
        """Draw text on image.

        Args:
            text: Text to be drawn.
            image: Image to be drawn onto the image.
            width: Width of the image.
            height: Height of the image.
            margin: Margin border.
        """
        width -= margin * 2

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        # Calculate initial text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Adjust font scale to fit target width
        font_scale = width / text_width
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw text
        cv2.putText(
            image,
            text,
            (margin, height - text_height),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.FILLED,
        )

    def _map_attention(self, attention: torch.Tensor) -> dict[str, np.ndarray]:
        """Maps the attention output to the space of the respective modality.

        Args:
            attention: The attention output of the model.

        Returns:
            A dictionary containing several xAI outputs:
                image_att: attention map in image space.
                text_att: attention map in text space.
                state_att: attention in proprio state space.

        """
        # SmolVLA's attention gives a [nbatch, nheads, ntokens, ntokens] att weights. This is 8x8(=16) tokes per image,
        # 48 tokes for text and 1 for the robot state
        min_img_value = float("inf")
        max_img_value = float("-inf")
        offset = 0
        images_att = {}
        if self.head_idx is None:
            attention_heads_select = attention.mean(dim=1)
        elif self.head_idx < 0:
            attention_heads_select, _ = attention.max(dim=1)
        else:
            attention_heads_select = attention[:, self.head_idx, :, :]

        for key, (height, width) in self.image_tile_shapes.items():
            # Select the appropriate attention type display method
            image_att = attention_heads_select[:, :, offset : offset + (height * width)].mean(dim=1)
            # Reshape into tile size
            image_att = image_att.resize(image_att.shape[0], height, width)
            # Calculate min and max add to list
            min_img_value, max_img_value = (
                min(torch.min(image_att).item(), min_img_value),
                max(torch.max(image_att).item(), max_img_value),
            )
            images_att[key] = image_att
            offset += height * width

        # Select the appropriate attention type display method for the text
        num_text_tokens = attention.shape[-1] - offset - 1  # seems to based on tokenizer output
        text_att = attention_heads_select[:, :, offset : offset + num_text_tokens].mean(dim=1)
        min_text_value, max_text_value = torch.min(text_att).item(), torch.max(text_att).item()
        offset += num_text_tokens

        # Select the appropriate attention type display method for the robot state
        state_att = attention_heads_select[:, :, offset : offset + 1].mean(dim=1)

        # rescale all modalities using min, max values
        images_att = {key: (img - min_img_value) / (max_img_value - min_img_value) for key, img in images_att.items()}
        text_att = (text_att - min_text_value) / (max_text_value - min_text_value)
        state_att = torch.clamp(state_att, 0.0, 1.0)

        # return all attention maps
        return {"image_att": images_att, "text_att": text_att, "state_att": state_att}
