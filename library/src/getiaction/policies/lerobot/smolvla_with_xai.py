# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA with XAI

This module provides a descendent of LeRobot's SmolVLA policy with added explainability (XAI).
"""


from typing import Any

import cv2
import numpy as np
import torch
from lerobot.constants import ACTION
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel, apply_rope
from lerobot.policies.utils import populate_queues
from torch import Tensor, nn
from transformers import AutoProcessor


class SmolVLMWithExpertModelWithXAI(SmolVLMWithExpertModel):

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
    ):
        super().__init__(model_id, load_vlm_weights, train_expert_only, freeze_vision_encoder, attention_mode, num_expert_layers, num_vlm_layers, self_attn_every_n_layers, expert_width_multiplier)
        self.qk = {}

    def eager_attention_forward_with_qk(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states,
    ):
        """Create an alternative for eager_attention_forward that also return attention probs (qk)
        """
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim,
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim,
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim,
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim,
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

    def forward_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ) -> tuple[list[Tensor], dict[Any, Any] | Any]:
        """Override this method to store the qk for each attention layer
        """
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            hidden_states = layer.input_layernorm(hidden_states)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

        # B,L,H,D with L sequence length, H number of heads, D head dim
        # concatenate on the number of embeddings/tokens
        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        seq_len = query_states.shape[1]
        if seq_len < position_ids.shape[1]:
            _position_ids = position_ids[:, :seq_len]
            _attention_mask = attention_mask[:, :seq_len, :seq_len]
        else:
            _position_ids = position_ids
            _attention_mask = attention_mask

        attention_mask_ = _attention_mask
        position_ids_ = _position_ids

        query_states = apply_rope(query_states, position_ids_)
        key_states = apply_rope(key_states, position_ids_)

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)

        att_output, att_weights = self.eager_attention_forward_with_qk(
            attention_mask_, batch_size, head_dim, query_states, key_states, value_states,
        )

        # store qk probs
        if fill_kv_cache:  # This is skipped for the denoising step
            self.qk[layer_idx] = att_weights.detach().cpu()

        return [att_output], past_key_values


class VLAFlowMatchingWithXAI(VLAFlowMatching):
    def __init__(self, config: SmolVLAConfig):
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
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size,
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size,
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size,
        )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long,
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length


class SmolVLAPolicyWithXAI(SmolVLAPolicy):
    def __init__(
        self,
        config: SmolVLAConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,

        # XAI parameters
        layer_idx: int = -1,
        head_idx: int = None,
    ):
        """Descendent of SmolVLA which adds XAI capabilities

        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
           layer_idx: int (default=-1) Which layer to display (None means average from all layers)
           head_idx: int (default=None) Which head to display (None means average from all heads)
            -1 takes the max instead of mean.

        """
        PreTrainedPolicy.__init__(self, config)  # Call grandparent instead of parent to prevent redundant initialization.
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats,
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats,
        )

        self.language_tokenizer = AutoProcessor.from_pretrained(self.config.vlm_model_name).tokenizer
        self.model = VLAFlowMatchingWithXAI(config)
        self.reset()

        # Initialize XAI
        self.layer_idx = layer_idx
        self.head_idx = head_idx

        self.image_shapes = {k: v.shape for k, v in self.model.config.image_features.items()}
        self.image_resized_padded_shapes = dict.fromkeys(self.model.config.image_features.keys(), self.model.config.resize_imgs_with_padding)
        self.image_tile_shapes = {k: [8, 8] for k in self.model.config.image_features.keys()}
        self.num_text_tokens = self.model.config.tokenizer_max_length
        self.num_robot_tokens = 1
        self.attention_modes = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Create an alternative for select_action that also returns the attention maps"""
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._queues[ACTION]) == 0:
            actions = self.get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            # d = {"actions": actions, "attention_maps": attention_maps}
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        # Also get the weights and task
        index = max(
            self.policy.model.vlm_with_expert.qk.keys()) + self.layer_idx + 1 if self.layer_idx < 0 else self.layer_idx
        attention_maps = self.policy.model.vlm_with_expert.qk[index]

        self.attention_modes = self._map_attention_to_images(attention_maps)
        self.attention_modes["task"] = batch["task"]

        return self._queues[ACTION].popleft()

    def get_xai(self) -> list[np.ndarray]:
        """Get the XAI layer"""
        visualizations = []
        for key, img in self.attention_modes["image_att"].items():
            # Scale pixel values to 0, 255 and permute channels
            obs_image = self.attention_modes["observation"][key].permute(0, 2, 3, 1)[0].cpu().numpy()
            mi, ma = np.min(obs_image), np.max(obs_image)
            obs_image = (((obs_image - mi) / (ma - mi)) * 255).astype(np.uint8)
            att_image = img[0, :, :, None].cpu().numpy()
            att_image = (att_image * 255).astype(np.uint8)

            # Resize attention maps
            h, w = self.image_shapes[key][1:]
            att_image = cv2.resize(att_image, (w, h))

            # Create heatmap from attention map
            heatmap = cv2.applyColorMap(att_image, cv2.COLORMAP_JET)

            # blend images
            vis = cv2.addWeighted(
                obs_image, 1 - 0.5,
                heatmap, 0.5, 0,
            )

            modalities_border: int = 20

            # Show robot attention as border
            arr = (self.attention_modes["state_att"].cpu().numpy() * 255).astype(np.uint8)
            border_color = cv2.applyColorMap(arr, cv2.COLORMAP_JET).squeeze().astype(int).tolist()
            cv2.rectangle(vis, (0, 0), (w - 1, h - 1), border_color, modalities_border)

            # Show text as bottom ribbon
            arr = (self.attention_modes["text_att"].cpu().numpy() * 255).astype(np.uint8)
            arr = arr[:, (arr != 0)[0]]  # Filter zeros
            colors = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
            colored = cv2.resize(colors, (w, modalities_border * 2))
            vis[h - modalities_border * 2:h, 0:w] = colored
            self._draw_text(self.attention_modes["task"], vis, w, h, 5)

            visualizations.append(vis)

        return visualizations

    def _draw_text(self, text, image, width, height, margin=5):
        text = text
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
        cv2.putText(image, text, (margin, height - text_height), font, font_scale, (255, 255, 255), thickness, cv2.FILLED)

    def _map_attention_to_images(self, attention):
        # SmolVLA's attention gives a [nbatch, nheads, ntokens, ntokens] att weights. This is 8x8(=16) tokes per image, 48 tokes for text and 1 for the robot state
        min_img_value = float("inf")
        max_img_value = float("-inf")
        offset = 0
        images_att = {}
        if self.head_idx is None:
            if self.head_idx < 0:
                attention_heads_select, _ = attention.max(dim=1)
            else:
                attention_heads_select = attention.mean(dim=1)
        else:
            attention_heads_select = attention[:, self.head_idx, :, :]
        for key, (height, width) in self.image_tile_shapes.items():
            # Select the appropriate attention type display method
            image_att = attention_heads_select[:, :, offset:offset + (height * width)].mean(dim=1)
            # Reshape into tile size
            image_att = image_att.resize(image_att.shape[0], height, width)
            # Calculate min and max add to list
            min_img_value, max_img_value = min(torch.min(image_att).item(), min_img_value), max(torch.max(image_att).item(), max_img_value)
            images_att[key] = image_att
            offset += height * width

        # Select the appropriate attention type display method for the text
        text_att = self._get_attention_of_type(attention_heads_select, offset, self.num_text_tokens)
        min_text_value, max_text_value = torch.min(text_att).item(), torch.max(text_att).item()
        offset += self.num_text_tokens

        # Select the appropriate attention type display method for the robot state
        state_att = self._get_attention_of_type(attention_heads_select, offset, offset)

        # rescale all modalities using min, max values
        images_att = {key: (img - min_img_value) / (max_img_value - min_img_value) for key, img in images_att.items()}
        text_att = (text_att - min_text_value) / (max_text_value - min_text_value)
        state_att = torch.clamp(state_att, 0.0, 1.0)

        # return all attention maps
        return {"image_att": images_att, "text_att": text_att, "state_att": state_att}
