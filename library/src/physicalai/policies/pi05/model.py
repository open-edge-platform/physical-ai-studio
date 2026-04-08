# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pi05 model implementation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers.cache_utils import DynamicCache

from physicalai.data.constants import IMAGE_MASKS, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK
from physicalai.export import ExportableModelMixin
from physicalai.export.backends import (
    ExportParameters,
    ONNXExportParameters,
    OpenVINOExportParameters,
    TorchExportParameters,
)
from physicalai.inference.manifest import ComponentSpec
from physicalai.policies.base import Model

from .pi_gemma import (
    PaliGemmaForConditionalGenerationWithPiGemma,
    PiGemmaForCausalLM,
    _gated_residual,
    layernorm_forward,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers.models.gemma import modeling_gemma

try:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
except ImportError as e:
    msg = "Pi05 requires the transformers library. Install with: uv pip install transformers"
    raise ImportError(msg) from e

logger = logging.getLogger(__name__)

# Attention mask value matching OpenPI convention
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38


def _get_safe_dtype(
    target_dtype: torch.dtype,
    device_type: str,
) -> torch.dtype:
    """Get a safe dtype for the given device type.

    Returns:
        Safe torch dtype for the target device.
    """
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def _create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> Tensor:
    """Compute sine-cosine positional embedding vectors for scalar positions.

    Returns:
        Tensor of sine-cosine positional embeddings.

    Raises:
        ValueError: If dimension is not divisible by 2 or time is not 1D.
    """
    if dimension % 2 != 0:
        msg = f"dimension ({dimension}) must be divisible by 2"
        raise ValueError(msg)

    if time.ndim != 1:
        msg = "The time tensor is expected to be of shape `(batch_size, )`."
        raise ValueError(msg)

    dtype = (
        torch.float32
        if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()
        else _get_safe_dtype(torch.float64, device.type)
    )
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def _sample_beta(
    alpha: float,
    beta: float,
    bsize: int,
    device: torch.device,
) -> Tensor:
    """Sample from Beta distribution (CPU-safe for MPS).

    Returns:
        Sampled tensor of shape (bsize,).
    """
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def _make_att_2d_masks(
    pad_masks: Tensor,
    att_masks: Tensor,
) -> Tensor:
    """Create 2D attention masks from padding and autoregressive masks.

    Tokens can attend to valid input tokens which have a cumulative mask_ar
    smaller or equal to theirs.

    Returns:
        2D boolean attention mask tensor.

    Raises:
        ValueError: If att_masks or pad_masks are not 2D.
    """
    if att_masks.ndim != 2:  # noqa: PLR2004
        msg = f"att_masks must be 2D, got {att_masks.ndim}D"
        raise ValueError(msg)
    if pad_masks.ndim != 2:  # noqa: PLR2004
        msg = f"pad_masks must be 2D, got {pad_masks.ndim}D"
        raise ValueError(msg)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def _clone_kv_cache(past_key_values: DynamicCache) -> DynamicCache:
    """Create a clone of a DynamicCache with cloned tensors.

    copy.deepcopy is not traceable by torch.jit / torch.onnx.export,
    so we manually clone the underlying key/value tensors instead.

    Returns:
        Cloned DynamicCache instance.
    """
    cloned = DynamicCache()
    for layer_idx, (key_states, value_states) in enumerate(past_key_values):
        cloned.update(key_states.clone(), value_states.clone(), layer_idx)
    return cloned


def _compute_layer_complete(  # noqa: PLR0914
    layer_idx: int,
    inputs_embeds: list[Tensor],
    attention_mask: Tensor,
    position_ids: Tensor,
    adarms_cond: list[Tensor | None],
    paligemma: PaliGemmaForConditionalGenerationWithPiGemma,
    gemma_expert: PiGemmaForCausalLM,
) -> list[Tensor]:
    """Complete layer computation function for gradient checkpointing.

    Returns:
        List of output embedding tensors for each model.
    """
    models = [paligemma.model.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])  # noqa: PLW2901
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        unsqueeze_dim=1,
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.model.language_model.layers[layer_idx].self_attn.scaling

    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()  # type: ignore[union-attr]
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


@dataclass
class GemmaVariantConfig:
    """Configuration for Gemma model variants."""

    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def get_gemma_config(variant: str) -> GemmaVariantConfig:
    """Return config for the specified Gemma variant.

    Returns:
        Configuration for the Gemma variant.

    Raises:
        ValueError: If variant is unknown.
    """
    if variant == "gemma_300m":
        return GemmaVariantConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return GemmaVariantConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    msg = f"Unknown variant: {variant}"
    raise ValueError(msg)


class PaliGemmaWithExpertModel(nn.Module):
    """PaliGemma model with action expert for Pi05."""

    def __init__(
        self,
        vlm_config: GemmaVariantConfig,
        action_expert_config: GemmaVariantConfig,
        use_adarms: list[bool] | None = None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = 224,
        freeze_vision_encoder: bool = False,  # noqa: FBT001, FBT002
        train_expert_only: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize PaliGemmaWithExpertModel."""
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(
        self,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ) -> None:
        """Convert selected parameters to bfloat16 precision.

        Raises:
            ValueError: If precision is not 'bfloat16' or 'float32'.
        """
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            msg = f"Invalid precision: {precision}"
            raise ValueError(msg)

        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self) -> None:
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True) -> None:  # noqa: FBT001, FBT002
        """Set training mode, keeping frozen modules in eval."""
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Embed image through vision tower and projector.

        Returns:
            Image embedding tensor.
        """
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)

        image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs * self.paligemma.config.text_config.hidden_size**0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens.

        Returns:
            Language token embedding tensor.
        """
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,  # noqa: FBT001
        adarms_cond: list[torch.Tensor] | None = None,
    ) -> tuple[list[Tensor | None], DynamicCache | None]:
        """Forward pass combining PaliGemma and action expert.

        Returns:
            Tuple of (output embeddings list, prefix past key values).

        Raises:
            ValueError: If inputs_embeds is None.
        """
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds is None:
            msg = "inputs_embeds must not be None"
            raise ValueError(msg)
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.model.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        _compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = _compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            def compute_final_norms(
                inputs_embeds: list[Tensor],
                adarms_cond: list[Tensor | None],
            ) -> list[Tensor]:
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(models[i].norm, hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class Pi05Model(ExportableModelMixin, Model):
    """Core Pi05 PyTorch model for flow matching VLA.

    This is the nn.Module that contains the actual model logic,
    separated from the Lightning wrapper.
    """

    def __init__(  # noqa: PLR0913
        self,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple[int, ...]]],
        *,
        paligemma_variant: Literal["gemma_300m", "gemma_2b"] = "gemma_2b",
        action_expert_variant: Literal["gemma_300m", "gemma_2b"] = "gemma_300m",
        dtype: Literal["bfloat16", "float32"] = "float32",
        chunk_size: int = 50,
        max_action_dim: int = 32,
        n_action_steps: int | None = None,
        num_inference_steps: int = 10,
        time_sampling_beta_alpha: float = 1.5,
        time_sampling_beta_beta: float = 1.0,
        time_sampling_scale: float = 0.999,
        time_sampling_offset: float = 0.001,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        image_resolution: tuple[int, int] = (224, 224),
        tokenizer_max_length: int = 200,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = True,
        gradient_checkpointing: bool = False,
        compile_model: bool = False,
        use_random_input_noise: bool = False,
    ) -> None:
        """Initialize Pi05Model.

        Args:
            dataset_stats: Dictionary containing dataset statistics for normalization
                and action dimension inference.
            paligemma_variant: Gemma variant for the VLM backbone.
            action_expert_variant: Gemma variant for the action expert.
            dtype: Precision for model weights.
            chunk_size: Number of action steps to predict.
            max_action_dim: Maximum dimension for action vectors.
            n_action_steps: Number of action steps to return. If None, returns all.
            num_inference_steps: Number of decoding steps for flow matching.
            time_sampling_beta_alpha: Alpha parameter for beta distribution time sampling.
            time_sampling_beta_beta: Beta parameter for beta distribution time sampling.
            time_sampling_scale: Scale factor for time sampling.
            time_sampling_offset: Offset for time sampling.
            min_period: Minimum period for sine-cosine positional encoding.
            max_period: Maximum period for sine-cosine positional encoding.
            image_resolution: Target image resolution (height, width). Must be square.
            tokenizer_max_length: Maximum token length for the tokenizer.
            freeze_vision_encoder: Whether to freeze the vision encoder during training.
            train_expert_only: Whether to train only the action expert.
            gradient_checkpointing: Whether to enable gradient checkpointing for memory optimization.
            compile_model: Whether to use torch.compile.
            use_random_input_noise: Whether to use random noise as the initial input for the denoising
                process during inference. If False, zeros are used instead.

        Raises:
            ValueError: If image resolution is not square.
        """
        super().__init__()
        self._chunk_size = chunk_size
        self._max_action_dim = max_action_dim
        self._n_action_steps = n_action_steps
        self._num_inference_steps = num_inference_steps
        self._dataset_stats = dataset_stats
        self._time_sampling_beta_alpha = time_sampling_beta_alpha
        self._time_sampling_beta_beta = time_sampling_beta_beta
        self._time_sampling_scale = time_sampling_scale
        self._time_sampling_offset = time_sampling_offset
        self._min_period = min_period
        self._max_period = max_period
        self._image_resolution = image_resolution
        self._tokenizer_max_length = tokenizer_max_length
        self._use_random_input_noise = use_random_input_noise

        paligemma_config = get_gemma_config(paligemma_variant)
        action_expert_config = get_gemma_config(action_expert_variant)

        if image_resolution[0] != image_resolution[1]:
            msg = f"PaliGemma expects square image resolution, invalid: {image_resolution}"
            raise ValueError(msg)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=dtype,
            image_size=image_resolution[0],
            freeze_vision_encoder=freeze_vision_encoder,
            train_expert_only=train_expert_only,
        )

        self.action_in_proj = nn.Linear(max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.gradient_checkpointing_enabled = False
        if gradient_checkpointing:
            self.gradient_checkpointing_enable()

        if compile_model:
            torch.set_float32_matmul_precision("high")
            # TODO(Eugene): max-autotune currently failed.  # noqa: TD003, FIX002
            # Set to default for now, need further investigation.
            compile_mode = "default"
            self.sample_actions = torch.compile(self.sample_actions, mode=compile_mode)  # type: ignore[method-assign]
            self.forward = torch.compile(self.forward, mode=compile_mode)  # type: ignore[method-assign]

    def set_dataset_stats(self, dataset_stats: dict) -> None:
        """Update dataset statistics used for normalization."""
        self._dataset_stats = dataset_stats

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Additional export arguments for model conversion.

        This property provides extra configuration parameters needed when exporting
        the model to different formats (ONNX, OpenVINO, and PyTorch).

        Returns:
            dict[str, ExportParameters]: A dictionary mapping format names to their export parameters.
            Supported formats: 'onnx', 'openvino', 'torch'.

        Example:
            >>> model = Pi05(input_features, output_features)
            >>> export_args = model.extra_export_args
            >>> onnx_args = export_args['onnx']
            >>> print(onnx_args.exporter_kwargs)
            {'output_names': ['action']}
        """
        preproc_specs = [
            ComponentSpec(type="pi05", image_resolution=self._image_resolution),
            ComponentSpec(
                type="hf_tokenizer",
                tokenizer_name="google/paligemma-3b-pt-224",
                revision="35e4f46485b4d07967e7e9935bc3786aad50687c",
                max_token_len=self._tokenizer_max_length,
            ),
        ]
        postproc_specs = [
            ComponentSpec(
                type="denormalize",
                stats={ACTION: self._dataset_stats[ACTION]},
                mode="mean_std",
            ),
        ]
        extra_args: dict[str, ExportParameters] = {}
        extra_args["onnx"] = ONNXExportParameters(
            exporter_kwargs={
                "output_names": ["action"],
            },
            export_tokenizer=False,
            preprocessors_specs=preproc_specs,
            postprocessors_specs=postproc_specs,
        )
        extra_args["openvino"] = OpenVINOExportParameters(
            outputs=["action"],
            compress_to_fp16=True,
            via_onnx=True,
            export_tokenizer=False,
            exporter_kwargs={},
            preprocessors_specs=preproc_specs,
            postprocessors_specs=postproc_specs,
        )
        extra_args["torch"] = TorchExportParameters()

        return extra_args

    @property
    def sample_input(self) -> dict[str, torch.Tensor | str]:
        """Generate a sample input dictionary for the model with random tensors.

        This method creates a dictionary containing sample input tensors that match the expected
        input format of the model. The tensors are randomly initialized and have shapes derived
        from the model's configuration.

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: A dictionary with two keys
                - 'state': A tensor representing the robot state with shape (1, *state_feature.shape).
                - 'images': Either a single tensor or a dictionary of tensors representing visual inputs,
                    depending on the number of image features configured.

        Note:
            The batch dimension (first dimension) is set to 1 for all tensors.
            The tensors are created on the same device as the model's parameters.
        """
        device = next(self.paligemma_with_expert.parameters()).device

        sample_input = {}

        num_image_features = sum(1 for key in self._dataset_stats if "image" in key)

        for feature_id in self._dataset_stats:
            if STATE in feature_id:
                state_feature = self._dataset_stats[feature_id]
                sample_input[STATE] = torch.randn(1, *cast("tuple", state_feature["shape"]), device=device)
            elif "image" in feature_id:
                image_feature = self._dataset_stats[feature_id]
                if num_image_features == 1:
                    sample_input[IMAGES] = torch.randn(1, *cast("tuple", image_feature["shape"]), device=device)
                else:
                    sample_input[IMAGES + "." + str(image_feature["name"])] = torch.randn(
                        1,
                        *cast("tuple", image_feature["shape"]),
                        device=device,
                    )

        sample_input[TASK] = "sample_task"

        return sample_input

    @property
    def reward_delta_indices(self) -> None:
        """Return reward indices.

        Currently returns `None` as rewards are not implemented.

        Returns:
            None
        """
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Get indices of actions relative to the current timestep.

        Returns:
            list[int]: A list of relative action indices.
        """
        return list(range(self._chunk_size))

    @property
    def observation_delta_indices(self) -> None:
        """Get indices of observations relative to the current timestep.

        Returns:
            None
        """
        return None

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        msg = "Enabled gradient checkpointing for Pi05Model"
        logger.info(msg)

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        msg = "Disabled gradient checkpointing for Pi05Model"
        logger.info(msg)

    def _apply_checkpoint(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Apply gradient checkpointing if enabled.

        Returns:
            Result of the function call.
        """
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func,
                *args,
                use_reentrant=False,
                preserve_rng_state=False,
                **kwargs,
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks: Tensor) -> Tensor:  # noqa: PLR6301
        """Prepare 4D attention masks for transformer.

        Returns:
            4D attention mask tensor.
        """
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape: tuple, device: torch.device) -> Tensor:
        """Sample noise for the model.

        Returns:
            Noise tensor.
        """
        if not self._use_random_input_noise or torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
            return torch.zeros(shape, dtype=torch.float32, device=device)
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            device=device,
        ).to(dtype=torch.float32)

    def sample_time(self, bsize: int, device: torch.device) -> Tensor:
        """Sample time values for the model.

        Returns:
            Time tensor.
        """
        if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
            alpha = self._time_sampling_beta_alpha
            beta = self._time_sampling_beta_beta
            time_beta = torch.full((bsize,), alpha / (alpha + beta), device=device, dtype=torch.float32)  # Beta mean
        else:
            time_beta = _sample_beta(
                self._time_sampling_beta_alpha,
                self._time_sampling_beta_beta,
                bsize,
                device,
            )
        time = time_beta * self._time_sampling_scale + self._time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self,
        images: Tensor,
        img_masks: Tensor,
        tokens: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer.

        Returns:
            Tuple of (embeddings, padding masks, attention masks).
        """
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img: Tensor) -> Tensor:
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        def lang_embed_func(tokens: Tensor) -> Tensor:
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(
        self,
        noisy_actions: Tensor,
        timestep: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embed noisy_actions and timestep for Expert Gemma processing.

        Returns:
            Tuple of (embeddings, padding masks, attention masks, adarms conditioning).
        """
        embs = []
        pad_masks = []
        att_masks = []

        time_emb = _create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self._min_period,
            max_period=self._max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        def action_proj_func(noisy_actions: Tensor) -> Tensor:
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb: Tensor) -> Tensor:
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        att_masks += [1] + ([0] * (self._chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(  # noqa: PLR0914
        self,
        batch: dict[str, Any],
    ) -> tuple[Tensor, dict[str, float]] | Tensor:
        """Training forward pass: compute flow matching loss.

        Args:
            batch: Preprocessed batch dict containing IMAGES, IMAGE_MASKS,
                TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK, and ACTION.

        Returns:
            Tuple of (mean loss tensor, loss dict with "loss" key).
        """
        if self.training:
            images = batch[IMAGES]
            img_masks = batch[IMAGE_MASKS]
            tokens = batch[TOKENIZED_PROMPT]
            masks = batch[TOKENIZED_PROMPT_MASK]
            actions = batch[ACTION]

            noise = self.sample_noise(actions.shape, actions.device)
            time = self.sample_time(actions.shape[0], actions.device)

            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

            if (
                self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

            pad_masks_combined = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks_combined = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

            att_2d_masks = _make_att_2d_masks(pad_masks_combined, att_masks_combined)
            position_ids = torch.cumsum(pad_masks_combined, dim=1) - 1

            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            def forward_func(
                prefix_embs: Tensor,
                suffix_embs: Tensor,
                att_2d_masks_4d: Tensor,
                position_ids: Tensor,
                adarms_cond: Tensor,
            ) -> Tensor:
                (_, suffix_out), _ = self.paligemma_with_expert.forward(
                    attention_mask=att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, suffix_embs],
                    use_cache=False,
                    adarms_cond=[None, adarms_cond],
                )
                return suffix_out

            suffix_out = self._apply_checkpoint(
                forward_func,
                prefix_embs,
                suffix_embs,
                att_2d_masks_4d,
                position_ids,
                adarms_cond,
            )

            suffix_out = suffix_out[:, -self._chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)

            def action_out_proj_func(suffix_out: Tensor) -> Tensor:
                return self.action_out_proj(suffix_out)

            v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

            losses = F.mse_loss(u_t, v_t, reduction="none")

            # Truncate losses to actual action dimensions to avoid dilution from padding
            original_action_dim = int(self._dataset_stats[ACTION]["shape"][-1])
            losses = losses[:, :, :original_action_dim]

            loss = losses.mean()
            return loss, {"loss": loss.item()}
        return self.predict_action_chunk(batch)

    def predict_action_chunk(self, batch: dict[str, Any]) -> Tensor:
        """Predict a chunk of actions from a preprocessed batch.

        Args:
            batch: Preprocessed batch dict containing IMAGES, IMAGE_MASKS,
                TOKENIZED_PROMPT, and TOKENIZED_PROMPT_MASK.

        Returns:
            Denoised action tensor, unpadded and clipped to n_action_steps.
        """
        images = batch[IMAGES]
        img_masks = batch[IMAGE_MASKS]
        tokens = batch[TOKENIZED_PROMPT]
        masks = batch[TOKENIZED_PROMPT_MASK]
        actions = self.sample_actions(images, img_masks, tokens, masks)

        # Unpad actions to actual action dimension
        original_action_dim = int(self._dataset_stats[ACTION]["shape"][-1])
        actions = actions[:, :, :original_action_dim]

        # Clip to n_action_steps so the action queue receives the first N actions,
        # not the last N (deque maxlen silently discards earlier items on extend).
        if self._n_action_steps is not None:
            actions = actions[:, : self._n_action_steps]

        return actions

    @torch.no_grad()
    def sample_actions(  # noqa: PLR0914
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """Inference forward pass: sample actions via iterative denoising.

        Returns:
            Denoised action tensor.
        """
        if num_steps is None:
            num_steps = self._num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            actions_shape = (bsize, self._chunk_size, self._max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = _make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )

            x_t += dt * v_t

        return x_t

    def denoise_step(  # noqa: PLR0914
        self,
        prefix_pad_masks: Tensor,
        past_key_values: DynamicCache | None,
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """Apply one denoising step of noise x_t at a given timestep.

        Returns:
            Velocity prediction tensor for this denoising step.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = _make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        # Deep-copy the KV cache so that HF's attention layers (which always call
        # past_key_values.update() in-place) don't corrupt the original prefix
        # cache across denoising steps.  copy.deepcopy is not traceable for ONNX,
        # so we clone tensors manually when exporting.
        past_key_values = _clone_kv_cache(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]  # type: ignore[index]
        suffix_out = suffix_out[:, -self._chunk_size :]  # type: ignore[index]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
