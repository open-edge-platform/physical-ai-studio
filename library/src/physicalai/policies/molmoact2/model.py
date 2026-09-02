# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 model architecture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, override

import torch
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from torch.nn import functional

from physicalai.data.observation import ACTION, FeatureType
from physicalai.policies.base import Model

from .components import (
    ActionExpert,
    MolmoAct2Backbone,
    MolmoAct2ForConditionalGeneration,
    MolmoAct2TextModel,
    MolmoAct2VisionBackbone,
)

if TYPE_CHECKING:
    from .config import MolmoAct2Config


_SAFE_WEIGHTS_NAME = "model.safetensors"
_SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
_VLM_LORA_LINEAR_LEAVES = "att_proj|attn_out|ff_proj|ff_out|wq|wk|wv|wo|w1|w2|w3|patch_embedding"
_ACTION_EXPERT_LORA_LINEAR_LEAVES = (
    r"time_embed\.(1|3)|"
    r"action_embed|"
    r"context_k_proj|context_v_proj|"
    r"blocks\.\d+\.self_attn\.(qkv|out_proj)|"
    r"blocks\.\d+\.cross_attn\.(q_proj|out_proj)|"
    r"blocks\.\d+\.mlp\.(up_proj|gate_proj|down_proj)|"
    r"blocks\.\d+\.modulation\.linear|"
    r"final_layer\.(modulation\.linear|linear)"
)

_MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "images",
    "token_pooling",
    "action_dim_is_pad",
)


def _masked_action_mse(
    predicted: Tensor,
    target: Tensor,
    *,
    action_horizon_is_pad: Tensor | None,
    action_dim_is_pad: Tensor | None,
) -> Tensor:
    """Return MSE over non-padded action steps and dimensions."""
    squared_error = functional.mse_loss(predicted, target, reduction="none")
    valid = torch.ones_like(squared_error, dtype=torch.bool)
    if action_horizon_is_pad is not None:
        horizon = (~action_horizon_is_pad.to(squared_error.device, dtype=torch.bool)).view(
            squared_error.shape[0],
            *([1] * (squared_error.ndim - 3)),
            squared_error.shape[-2],
            1,
        )
        valid &= horizon
    if action_dim_is_pad is not None:
        dimensions = (~action_dim_is_pad.to(squared_error.device, dtype=torch.bool)).view(
            squared_error.shape[0],
            *([1] * (squared_error.ndim - 3)),
            1,
            squared_error.shape[-1],
        )
        valid &= dimensions
    mask = valid.to(squared_error.dtype)
    return (squared_error * mask).sum() / mask.sum().clamp_min(1)


def _lora_target_modules(*, enable_action_expert: bool) -> str:
    """Build the PEFT target-module regex for MolmoAct2 linear layers.

    Returns:
        A regex matching VLM linears and, optionally, action-expert linears.
    """
    vlm_targets = rf"model\.(transformer|vision_backbone)\.(?:.*\.)?({_VLM_LORA_LINEAR_LEAVES})$"
    if not enable_action_expert:
        return vlm_targets
    return f"({vlm_targets}|model\\.action_expert\\.(?:{_ACTION_EXPERT_LORA_LINEAR_LEAVES})$)"


def _resolve_weights_path(weights_path: str | Path) -> Path:
    """Resolve a checkpoint directory or safetensors file to its load entrypoint.

    Returns:
        The single-file checkpoint or sharded checkpoint index path.

    Raises:
        FileNotFoundError: If no supported checkpoint exists at the path.
    """
    path = Path(weights_path)
    if path.is_dir():
        index_path = path / _SAFE_WEIGHTS_INDEX_NAME
        if index_path.is_file():
            return index_path
        single_file_path = path / _SAFE_WEIGHTS_NAME
        if single_file_path.is_file():
            return single_file_path
    elif path.is_file():
        return path

    msg = f"No MolmoAct2 safetensors checkpoint found at {str(path)!r}."
    raise FileNotFoundError(msg)


class MolmoAct2Model(Model):
    """Native MolmoAct2 architecture assembled from explicit model arguments."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        # Text transformer
        hidden_size: int = 2560,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        vocab_size: int = 154_624,
        additional_vocab_size: int = 128,
        qkv_bias: bool = False,
        num_hidden_layers: int = 36,
        intermediate_size: int = 9728,
        hidden_act: str = "silu",
        rope_theta: float = 5_000_000.0,
        use_qk_norm: bool = True,
        qk_norm_type: str = "qwen3",
        layer_norm_eps: float = 1e-6,
        norm_after: bool = False,
        # Vision transformer
        vision_hidden_size: int = 1152,
        vision_intermediate_size: int = 4304,
        vision_num_hidden_layers: int = 27,
        vision_num_attention_heads: int = 16,
        vision_num_key_value_heads: int = 16,
        vision_head_dim: int = 72,
        vision_hidden_act: str = "gelu_pytorch_tanh",
        vision_layer_norm_eps: float = 1e-6,
        image_default_input_size: tuple[int, int] = (378, 378),
        image_patch_size: int = 14,
        image_num_pos: int = 729,
        vision_attention_dropout: float = 0.0,
        vision_residual_dropout: float = 0.0,
        # Vision adapter
        adapter_vit_layers: tuple[int, ...] = (-3, -9),
        adapter_pooling_attention_mask: bool = True,
        adapter_hidden_size: int = 1152,
        adapter_num_attention_heads: int = 16,
        adapter_num_key_value_heads: int = 16,
        adapter_head_dim: int = 72,
        adapter_attention_dropout: float = 0.0,
        adapter_residual_dropout: float = 0.0,
        adapter_hidden_act: str = "silu",
        adapter_intermediate_size: int = 9728,
        adapter_text_hidden_size: int = 2560,
        image_feature_dropout: float = 0.0,
        # Action expert
        add_action_expert: bool = True,
        action_expert_max_action_dim: int = 32,
        action_expert_hidden_size: int = 768,
        action_expert_num_layers: int = 36,
        action_expert_num_heads: int = 8,
        action_expert_mlp_ratio: float = 4.0,
        action_expert_ffn_multiple_of: int = 256,
        action_expert_timestep_embed_dim: int = 256,
        action_expert_context_layer_norm: bool = True,
        action_expert_qk_norm: bool = True,
        action_expert_qk_norm_eps: float = 1e-6,
        action_expert_rope: bool = True,
        action_expert_causal_attn: bool = False,
        # Flow matching
        image_patch_id: int = 154_626,
        mask_action_dim_padding: bool = True,
        flow_matching_num_steps: int = 10,
        max_action_dim: int = 32,
        num_flow_timesteps: int = 8,
        flow_matching_cutoff: float = 1.0,
        flow_matching_time_offset: float = 0.001,
        flow_matching_time_scale: float = 0.999,
        flow_matching_beta_alpha: float = 1.0,
        flow_matching_beta_beta: float = 1.5,
        chunk_size: int = 30,
        n_action_steps: int = 30,
        action_dim: int | None = None,
        use_random_input_noise: bool = False,
        # LoRA
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_bias: Literal["all", "lora_only", "none"] = "none",
    ) -> None:
        """Construct the text, vision, and action components."""
        super().__init__()
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha
        self._lora_dropout = lora_dropout
        self._lora_bias: Literal["all", "lora_only", "none"] = lora_bias
        self._chunk_size = chunk_size
        self._n_action_steps = n_action_steps
        self._action_dim = action_dim or max_action_dim
        self._use_random_input_noise = use_random_input_noise
        self._vlm_frozen = False

        transformer = MolmoAct2TextModel(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            additional_vocab_size=additional_vocab_size,
            qkv_bias=qkv_bias,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            rope_theta=rope_theta,
            use_qk_norm=use_qk_norm,
            qk_norm_type=qk_norm_type,
            layer_norm_eps=layer_norm_eps,
            norm_after=norm_after,
        )
        vision_backbone = MolmoAct2VisionBackbone(
            vision_hidden_size=vision_hidden_size,
            vision_intermediate_size=vision_intermediate_size,
            vision_num_hidden_layers=vision_num_hidden_layers,
            vision_num_attention_heads=vision_num_attention_heads,
            vision_num_key_value_heads=vision_num_key_value_heads,
            vision_head_dim=vision_head_dim,
            vision_hidden_act=vision_hidden_act,
            vision_layer_norm_eps=vision_layer_norm_eps,
            image_default_input_size=image_default_input_size,
            image_patch_size=image_patch_size,
            image_num_pos=image_num_pos,
            vision_attention_dropout=vision_attention_dropout,
            vision_residual_dropout=vision_residual_dropout,
            adapter_vit_layers=adapter_vit_layers,
            adapter_pooling_attention_mask=adapter_pooling_attention_mask,
            adapter_hidden_size=adapter_hidden_size,
            adapter_num_attention_heads=adapter_num_attention_heads,
            adapter_num_key_value_heads=adapter_num_key_value_heads,
            adapter_head_dim=adapter_head_dim,
            adapter_attention_dropout=adapter_attention_dropout,
            adapter_residual_dropout=adapter_residual_dropout,
            adapter_hidden_act=adapter_hidden_act,
            adapter_intermediate_size=adapter_intermediate_size,
            adapter_text_hidden_size=adapter_text_hidden_size,
            image_feature_dropout=image_feature_dropout,
        )
        action_expert = None
        if add_action_expert:
            action_expert = ActionExpert(
                max_action_dim=action_expert_max_action_dim,
                hidden_size=action_expert_hidden_size,
                num_layers=action_expert_num_layers,
                num_heads=action_expert_num_heads,
                mlp_ratio=action_expert_mlp_ratio,
                ffn_multiple_of=action_expert_ffn_multiple_of,
                timestep_embed_dim=action_expert_timestep_embed_dim,
                context_layer_norm=action_expert_context_layer_norm,
                qk_norm=action_expert_qk_norm,
                qk_norm_eps=action_expert_qk_norm_eps,
                rope=action_expert_rope,
                causal_attn=action_expert_causal_attn,
                llm_kv_dim=num_key_value_heads * head_dim,
                llm_num_layers=num_hidden_layers,
            )

        model = MolmoAct2Backbone(
            transformer=transformer,
            vision_backbone=vision_backbone,
            action_expert=action_expert,
            image_patch_id=image_patch_id,
            mask_action_dim_padding=mask_action_dim_padding,
            flow_matching_num_steps=flow_matching_num_steps,
            max_action_dim=max_action_dim,
            num_flow_timesteps=num_flow_timesteps,
            flow_matching_cutoff=flow_matching_cutoff,
            flow_matching_time_offset=flow_matching_time_offset,
            flow_matching_time_scale=flow_matching_time_scale,
            flow_matching_beta_alpha=flow_matching_beta_alpha,
            flow_matching_beta_beta=flow_matching_beta_beta,
        )
        self.backbone = MolmoAct2ForConditionalGeneration(
            model=model,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )

    def load_weights(self, weights_path: str | Path) -> None:
        """Strictly load a single-file or sharded safetensors checkpoint.

        Raises:
            RuntimeError: If checkpoint keys do not exactly match the model.
            TypeError: If a sharded checkpoint index has an invalid weight map.
        """
        resolved_path = _resolve_weights_path(weights_path)
        if resolved_path.name != _SAFE_WEIGHTS_INDEX_NAME:
            self.backbone.load_state_dict(load_safetensors_file(resolved_path, device="cpu"), strict=True)
            return

        with resolved_path.open(encoding="utf-8") as index_file:
            index = json.load(index_file)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not all(
            isinstance(key, str) and isinstance(shard, str) for key, shard in weight_map.items()
        ):
            msg = f"Invalid safetensors weight map in {resolved_path}."
            raise TypeError(msg)

        model_keys = set(self.backbone.state_dict())
        checkpoint_keys = set(weight_map)
        missing = sorted(model_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - model_keys)
        if missing or unexpected:
            msg = f"MolmoAct2 checkpoint keys mismatch. Missing: {missing[:6]} Unexpected: {unexpected[:6]}"
            raise RuntimeError(msg)

        loaded_keys: set[str] = set()
        for shard_name in sorted(set(weight_map.values())):
            shard_path = resolved_path.parent / shard_name
            state_dict = load_safetensors_file(shard_path, device="cpu")
            expected_shard_keys = {key for key, mapped_shard in weight_map.items() if mapped_shard == shard_name}
            if set(state_dict) != expected_shard_keys:
                msg = f"MolmoAct2 shard keys mismatch for {shard_name!r}."
                raise RuntimeError(msg)
            self.backbone.load_state_dict(state_dict, strict=False)
            loaded_keys.update(state_dict)

        if loaded_keys != model_keys:
            msg = "MolmoAct2 sharded checkpoint did not load every model parameter."
            raise RuntimeError(msg)

    @property
    def _unwrapped_backbone(self) -> MolmoAct2ForConditionalGeneration:
        """Checkpoint-root model, unwrapped from PEFT when necessary."""
        base_model = getattr(self.backbone, "base_model", None)
        if base_model is not None and hasattr(base_model, "model"):
            return base_model.model  # type: ignore[no-any-return]
        return self.backbone  # type: ignore[return-value]

    def enable_lora(self, *, enable_action_expert: bool = False) -> None:
        """Attach PEFT LoRA adapters to configured MolmoAct2 linear layers.

        Raises:
            ImportError: If PEFT is not installed.
            RuntimeError: If adapters are already enabled or no action expert exists
                when full action-expert training is requested.
        """
        if getattr(self.backbone, "base_model", None) is not None:
            msg = "MolmoAct2 LoRA adapters are already enabled."
            raise RuntimeError(msg)
        try:
            from peft import LoraConfig, get_peft_model  # noqa: PLC0415
        except ImportError as error:
            msg = "MolmoAct2 LoRA requires peft. Install with: pip install 'physicalai-train[molmoact2]'"
            raise ImportError(msg) from error

        lora_config = LoraConfig(
            r=self._lora_rank,
            lora_alpha=self._lora_alpha,
            lora_dropout=self._lora_dropout,
            target_modules=_lora_target_modules(enable_action_expert=enable_action_expert),
            bias=self._lora_bias,
        )
        self.backbone = get_peft_model(self.backbone, lora_config)  # type: ignore[assignment, arg-type]  # pyrefly: ignore[bad-assignment]
        if not enable_action_expert:
            action_expert = self._unwrapped_backbone.model.action_expert
            if action_expert is None:
                msg = "LoRA without action-expert adapters requires an action expert to train."
                raise RuntimeError(msg)
            for parameter in action_expert.parameters():
                parameter.requires_grad = True
        self.train(self.training)

    def enable_gradient_checkpointing(self) -> None:
        """Enable activation checkpointing on text, vision, and action stacks."""
        model = self._unwrapped_backbone.model
        model.transformer.gradient_checkpointing = True
        model.vision_backbone.gradient_checkpointing = True
        if model.action_expert is not None:
            model.action_expert.gradient_checkpointing = True

    def enable_compile(self) -> None:
        """Compile model training and inference entrypoints."""
        torch.set_float32_matmul_precision("high")
        self.forward = torch.compile(self.forward, mode="default")  # type: ignore[method-assign]
        self.predict_action_chunk = torch.compile(
            self.predict_action_chunk,
            mode="default",
        )

    def gradient_checkpointing_disable(self) -> None:
        """Disable activation checkpointing on all component stacks."""
        model = self._unwrapped_backbone.model
        model.transformer.gradient_checkpointing = False
        model.vision_backbone.gradient_checkpointing = False
        if model.action_expert is not None:
            model.action_expert.gradient_checkpointing = False

    def freeze_vlm(self) -> None:
        """Freeze and evaluate the VLM while leaving the action expert trainable.

        Raises:
            RuntimeError: If the model has no action expert.
        """
        action_expert = self._unwrapped_backbone.model.action_expert
        if action_expert is None:
            msg = "Cannot freeze the VLM because MolmoAct2 has no action expert to train."
            raise RuntimeError(msg)
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in action_expert.parameters():
            parameter.requires_grad = True
        self._vlm_frozen = True
        self.train(self.training)

    @override
    def train(self, mode: bool = True) -> MolmoAct2Model:
        """Set module mode while keeping a frozen VLM in evaluation mode.

        Returns:
            This model.
        """
        super().train(mode)
        if self._vlm_frozen:
            checkpoint_root = self._unwrapped_backbone
            checkpoint_root.eval()
            action_expert = checkpoint_root.model.action_expert
            if action_expert is not None:
                action_expert.train(mode)
        return self

    @override
    def forward(self, batch: dict[str, Any]) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        """Compute training loss or predict actions according to module mode.

        Returns:
            A loss tuple in training mode or a normalized action chunk in evaluation mode.
        """
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    @override
    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute the masked continuous flow-matching objective.

        Returns:
            The differentiable loss and detached metrics.
        """
        predicted, target = self._unwrapped_backbone.model.predict_flow_velocity(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
            images=batch.get("images"),
            token_pooling=batch.get("token_pooling"),
            actions=batch[ACTION],
            action_dim_is_pad=batch.get("action_dim_is_pad"),
            freeze_encoder=self._vlm_frozen,
        )
        loss = _masked_action_mse(
            predicted,
            target,
            action_horizon_is_pad=batch.get("action_horizon_is_pad"),
            action_dim_is_pad=batch.get("action_dim_is_pad")
            if self._unwrapped_backbone.model.mask_action_dim_padding
            else None,
        )
        metric = loss.detach()
        return loss, {"action_flow_loss": metric, "loss": metric}

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Any],
        *,
        sample_noise: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate a normalized action chunk with continuous flow matching.

        Returns:
            Normalized actions trimmed to the configured horizon and action dimension.
        """
        model_inputs = {key: batch[key] for key in _MODEL_INPUT_KEYS if key in batch}
        actions = self._unwrapped_backbone.model.generate_actions_from_inputs(
            **model_inputs,
            action_horizon=self._chunk_size,
            sample_noise=self._use_random_input_noise if sample_noise is None else sample_noise,
            generator=generator,
        )
        return actions[:, : self._n_action_steps, : self._action_dim].float()

    @torch.no_grad()
    @override
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute denoised action MSE and the flow-matching validation loss.

        Returns:
            Denoised action MSE and detached action-MSE and flow-loss metrics.
        """
        predicted = self.predict_action_chunk(batch, sample_noise=False)
        target = batch[ACTION][:, : predicted.shape[1], : predicted.shape[2]].to(predicted)
        horizon_mask = batch.get("action_horizon_is_pad")
        if horizon_mask is not None:
            horizon_mask = horizon_mask[:, : predicted.shape[1]]
        dimension_mask = batch.get("action_dim_is_pad")
        if dimension_mask is not None:
            dimension_mask = dimension_mask[:, : predicted.shape[2]]
        action_mse = _masked_action_mse(
            predicted,
            target,
            action_horizon_is_pad=horizon_mask,
            action_dim_is_pad=dimension_mask,
        )
        flow_loss, _ = self.compute_loss(batch)
        return action_mse, {
            "loss": action_mse.detach(),
            "action_mse": action_mse.detach(),
            "action_flow_loss": flow_loss.detach(),
        }

    @property
    @override
    def reward_delta_indices(self) -> None:
        """Reward deltas are not model inputs."""
        return None

    @property
    @override
    def action_delta_indices(self) -> list[int]:
        """Future action indices in the configured chunk."""
        return list(range(self._chunk_size))

    @property
    @override
    def observation_delta_indices(self) -> None:
        """Observation deltas are not required by the architecture."""
        return None

    @classmethod
    def from_config(
        cls,
        config: MolmoAct2Config,
    ) -> Self:
        """Construct a model from a resolved MolmoAct2 configuration.

        Returns:
            A model initialized from the model-owned configuration fields.

        Raises:
            ValueError: If a required optional configuration field is unresolved.
        """
        if config.num_key_value_heads is None:
            msg = "num_key_value_heads must be resolved before building MolmoAct2Model."
            raise ValueError(msg)
        if config.image_patch_id is None:
            msg = "image_patch_id must be resolved before building MolmoAct2Model."
            raise ValueError(msg)

        action_dim = config.max_action_dim
        for feature in config.output_features or []:
            if feature.ftype == FeatureType.ACTION and feature.shape:
                action_dim = int(feature.shape[-1])
                break

        return cls(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            vocab_size=config.vocab_size,
            additional_vocab_size=config.additional_vocab_size,
            qkv_bias=config.qkv_bias,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rope_theta=config.rope_theta,
            use_qk_norm=config.use_qk_norm,
            layer_norm_eps=config.layer_norm_eps,
            norm_after=config.norm_after,
            vision_hidden_size=config.vision_hidden_size,
            vision_intermediate_size=config.vision_intermediate_size,
            vision_num_hidden_layers=config.vision_num_hidden_layers,
            vision_num_attention_heads=config.vision_num_attention_heads,
            vision_num_key_value_heads=config.vision_num_key_value_heads,
            vision_head_dim=config.vision_head_dim,
            vision_hidden_act=config.vision_hidden_act,
            vision_layer_norm_eps=config.vision_layer_norm_eps,
            image_default_input_size=config.image_default_input_size,
            image_patch_size=config.image_patch_size,
            image_num_pos=config.image_num_pos,
            vision_attention_dropout=config.vision_attention_dropout,
            vision_residual_dropout=config.vision_residual_dropout,
            adapter_vit_layers=config.adapter_vit_layers,
            adapter_pooling_attention_mask=config.adapter_pooling_attention_mask,
            adapter_hidden_size=config.adapter_hidden_size,
            adapter_num_attention_heads=config.adapter_num_attention_heads,
            adapter_num_key_value_heads=config.adapter_num_key_value_heads,
            adapter_head_dim=config.adapter_head_dim,
            adapter_attention_dropout=config.adapter_attention_dropout,
            adapter_residual_dropout=config.adapter_residual_dropout,
            adapter_hidden_act=config.adapter_hidden_act,
            adapter_intermediate_size=config.adapter_intermediate_size,
            adapter_text_hidden_size=config.adapter_text_hidden_size,
            image_feature_dropout=config.image_feature_dropout,
            add_action_expert=config.add_action_expert,
            action_expert_max_action_dim=config.action_expert_max_action_dim,
            action_expert_hidden_size=config.action_expert_hidden_size,
            action_expert_num_layers=config.action_expert_num_layers,
            action_expert_num_heads=config.action_expert_num_heads,
            action_expert_mlp_ratio=config.action_expert_mlp_ratio,
            action_expert_ffn_multiple_of=config.action_expert_ffn_multiple_of,
            action_expert_timestep_embed_dim=config.action_expert_timestep_embed_dim,
            action_expert_context_layer_norm=config.action_expert_context_layer_norm,
            action_expert_qk_norm=config.action_expert_qk_norm,
            action_expert_qk_norm_eps=config.action_expert_qk_norm_eps,
            action_expert_rope=config.action_expert_rope,
            action_expert_causal_attn=config.action_expert_causal_attn,
            image_patch_id=config.image_patch_id,
            mask_action_dim_padding=config.mask_action_dim_padding,
            flow_matching_num_steps=config.flow_matching_num_steps,
            max_action_dim=config.max_action_dim,
            num_flow_timesteps=config.num_flow_timesteps,
            flow_matching_cutoff=config.flow_matching_cutoff,
            flow_matching_time_offset=config.flow_matching_time_offset,
            flow_matching_time_scale=config.flow_matching_time_scale,
            flow_matching_beta_alpha=config.flow_matching_beta_alpha,
            flow_matching_beta_beta=config.flow_matching_beta_beta,
            chunk_size=config.chunk_size,
            n_action_steps=config.n_action_steps,
            action_dim=action_dim,
            use_random_input_noise=config.use_random_input_noise,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_bias=config.lora_bias,
        )
