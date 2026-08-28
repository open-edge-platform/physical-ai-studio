# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)
# ruff: noqa: T201

"""VTC-Qwen3-VL backbone adapter."""

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from transformers import AutoConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.initialization import no_init_weights
from transformers.utils import is_torchdynamo_compiling

from physicalai.policies.rldx1.components.backbone.modeling_vtc import LayerWrapper, VTCQwen3Model


class VTCQwen3VLBackbone(nn.Module):
    """VTC-Qwen3-VL backbone adapter."""

    def __init__(  # noqa: C901, PLR0912, PLR0915
        self,
        model_name: str,
        tune_llm: bool = False,  # noqa: FBT001, FBT002
        tune_visual: bool = False,  # noqa: FBT001, FBT002
        select_layer: int = -1,
        load_bf16: bool = False,  # noqa: FBT001, FBT002
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,  # noqa: FBT001, FBT002
        use_cog_tokens: bool = False,  # noqa: FBT001, FBT002
        cog_mode: str = "cog_only",
        n_cog_tokens: int = 8,
        motion_config: dict | None = None,
        transformers_loading_kwargs: dict | None = None,
        gradient_checkpointing: bool = False,  # noqa: FBT001, FBT002
        skip_pretrained_weights: bool = True,  # noqa: FBT001, FBT002
        attn_implementation: str = "sdpa",
    ) -> None:
        """Initialize the VTC-Qwen3-VL backbone wrapper.

        Args:
            model_name: HuggingFace repo ID or local path of the VTC pretrained
                model or a base Qwen3-VL model to load weights from.
            tune_llm: Unfreeze the full LLM (language model + lm_head) for
                gradient updates.
            tune_visual: Unfreeze the full visual tower for gradient updates.
                The motion-module block remains trainable regardless of this flag.
            select_layer: Index (or list of indices) of LLM hidden states to
                return. Negative indices count from the last kept layer.
                Unused when ``use_cog_tokens`` is True (cog-only mode returns
                the last ``n_cog_tokens`` positions instead).
            load_bf16: Cast the full model to ``bfloat16`` after loading. When
                combined with ``trainable_params_fp32``, trainable parameters
                are subsequently upcast back to ``float32``.
            tune_top_llm_layers: Number of top LLM transformer layers to
                unfreeze in addition to ``tune_llm``. Ignored when zero.
            trainable_params_fp32: Cast trainable parameters to ``float32``
                after a ``load_bf16`` load for mixed-precision stability.
            use_cog_tokens: Append ``n_cog_tokens`` learnable query embeddings
                to the input sequence and use their hidden states as backbone
                features (VTC cog-token path).
            cog_mode: How to extract features when ``use_cog_tokens`` is True.
                ``"cog_only"`` returns only the last ``n_cog_tokens`` positions;
                ``"full"`` returns the entire sequence.
            n_cog_tokens: Number of learnable cognition-token queries to append.
                Must be positive when ``use_cog_tokens`` is True.
            motion_config: Optional dict configuring the motion-module block,
                e.g. ``{"use_motion": True, "motion_insert_layer": 9,
                "motion_injection_point": "vl_input"}``.
            transformers_loading_kwargs: Extra kwargs forwarded to every
                HuggingFace ``from_pretrained`` / ``AutoConfig.from_pretrained``
                call (e.g. ``revision``, ``cache_dir``, ``token``).
                ``trust_remote_code`` defaults to ``True`` if not supplied.
            gradient_checkpointing: Enable activation checkpointing on the LLM
                and visual tower for training memory savings.
            skip_pretrained_weights: When ``True``, builds the architecture
                from config only and skips downloading pretrained weights
                (weights are expected to arrive via a Lightning checkpoint).
            attn_implementation: Attention backend passed to the underlying
                HuggingFace model (e.g. ``"sdpa"``, ``"flash_attention_2"``).

        Raises:
            ValueError: If ``motion_injection_point`` is ``"vl_input"`` but
                ``use_cog_tokens`` is ``False``.
            ValueError: If any value in ``select_layer`` is outside the range
                ``[0, total_layers]``.
        """
        super().__init__()

        transformers_loading_kwargs = dict(transformers_loading_kwargs or {})
        transformers_loading_kwargs.setdefault("trust_remote_code", True)
        # Pop revision out so it's always passed as an explicit keyword (not hidden
        # inside **kwargs) — keeps the pin auditable and visible to security scanners.
        revision = transformers_loading_kwargs.pop("revision", None)

        if skip_pretrained_weights:
            print("[i] Creating VTC-Qwen3-VL architecture only (weights from checkpoint)")

            backbone_config = AutoConfig.from_pretrained(
                model_name,
                revision=revision,
                **transformers_loading_kwargs,
            )
            if motion_config is not None:
                for k, v in motion_config.items():
                    setattr(backbone_config.vision_config, k, v)
            backbone_config._attn_implementation = attn_implementation  # noqa: SLF001
            print("Attention implementation:", backbone_config._attn_implementation)  # noqa: SLF001
            # The checkpoint state dict below replaces every backbone tensor.
            # Avoid spending time initializing weights that are immediately discarded.
            with no_init_weights():
                self.qwen_model = VTCQwen3Model._from_config(backbone_config)  # noqa: SLF001

            for layer_idx in range(len(self.qwen_model.model.language_model.layers)):
                self.qwen_model.model.language_model.layers[layer_idx] = LayerWrapper(
                    self.qwen_model.model.language_model.layers[layer_idx],
                    layer_idx=layer_idx,
                    internal_projection=4,
                    img_pattern=[151652],
                    motion_token=1,
                )
            # Stock transformers never threads input_ids into decoder layers and
            # expects a bare-tensor layer return; the wrapped stack needs both.
            if load_bf16:
                self.qwen_model = self.qwen_model.to(torch.bfloat16)
        else:
            print(f"[i] Loading VTC-Qwen3-VL model from {model_name}")
            self.qwen_model = VTCQwen3Model.from_pretrained(  # type: ignore[assignment]
                model_name,
                motion_config=motion_config,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                revision=revision,
                **transformers_loading_kwargs,
            )

        if gradient_checkpointing:
            self.qwen_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.qwen_model.gradient_checkpointing_disable()

        # Keep BatchNorm running stats in float32 for bf16 compatibility
        motion_block = getattr(self.qwen_model.model.visual, "motion_block", None)
        if motion_block is not None:
            for m in motion_block.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                    m.float()
        self.qwen_linear = torch.nn.Identity()
        self.use_cog_tokens = bool(use_cog_tokens)
        self.cog_mode = cog_mode
        self.n_cog_tokens = int(n_cog_tokens)

        # motion module vl_input projection: vision_hidden_dim -> llm_hidden_dim
        self.motion_injection_point = (
            motion_config.get("motion_injection_point", "vision_encoder") if motion_config else "vision_encoder"
        )
        self.motion_pool_type = motion_config.get("motion_pool_type", "avg") if motion_config else "avg"
        self.motion_drop = motion_config.get("motion_drop", True) if motion_config else True
        self.moss_proj = None
        self.moss_spatial_conv = None
        if self.motion_injection_point == "vl_input" and motion_config and motion_config.get("use_motion", False):
            vit_hidden_size = self.qwen_model.model.visual.config.hidden_size  # 1280
            llm_hidden_size = self.qwen_model.model.language_model.config.hidden_size  # 3584
            self.moss_proj = nn.Sequential(
                nn.LayerNorm(vit_hidden_size),
                nn.Linear(vit_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
            # Zero-init last linear so motion module tokens start as no-ops
            last_linear = self.moss_proj[-1]  # type: ignore[index]
            nn.init.zeros_(last_linear.weight)  # type: ignore[arg-type]
            nn.init.zeros_(last_linear.bias)  # type: ignore[arg-type]

            if self.motion_pool_type == "conv":
                self.moss_spatial_conv = nn.Sequential(
                    nn.Conv2d(
                        vit_hidden_size,
                        vit_hidden_size,
                        kernel_size=4,
                        stride=4,
                        groups=vit_hidden_size,
                        bias=False,
                    ),
                    nn.Conv2d(vit_hidden_size, vit_hidden_size, kernel_size=1, bias=True),
                )
                print(
                    f"[motion module vl_input] Conv pooling: depthwise separable, "
                    f"params={sum(p.numel() for p in self.moss_spatial_conv.parameters()):,}",
                )

            print(
                f"[motion module] Created moss_proj: {vit_hidden_size} -> {llm_hidden_size},"
                f"pool_type={self.motion_pool_type}",
            )
            if not use_cog_tokens:
                msg = "motion module vl_input requires use_cog_tokens=True"
                raise ValueError(msg)

        if self.use_cog_tokens:
            feature_dim = self.qwen_model.model.language_model.config.hidden_size
            self._init_cog_token_modules(feature_dim)

        total_layers = len(self.qwen_model.model.language_model.layers)

        if isinstance(select_layer, (list, tuple)):
            hs_indices = sorted({int(i) for i in select_layer})
        else:
            hs_indices = [int(select_layer)]

        for k in hs_indices:
            if not (0 <= k <= total_layers):
                msg = f"select_layer {k} out of range 0..{total_layers}"
                raise ValueError(msg)

        max_k = hs_indices[-1]
        while len(self.qwen_model.model.language_model.layers) > max_k:
            self.qwen_model.model.language_model.layers.pop(-1)

        self.select_layers = hs_indices
        self._hs_idx = lambda k: k

        print(
            f"\n[i] Select layers (hs indices): {self.select_layers} "
            f"of total_layers={total_layers}; kept {max_k} blocks",
        )

        self.set_trainable_parameters(
            tune_llm,
            tune_visual,
            tune_top_llm_layers,
            print_params=False,
        )
        if load_bf16 and trainable_params_fp32:
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    print(f"Casting trainable parameter {n} to fp32")

    def _init_cog_token_modules(self, feature_dim: int) -> None:
        """Initialize the learnable cognition-token embedding.

        Creates ``self.cog_emb`` — an ``(n_cog_tokens, feature_dim)`` parameter
        initialised with small Gaussian noise (sigma=0.02) — and logs its statistics.
        Called once from ``__init__`` when ``use_cog_tokens`` is True.

        Args:
            feature_dim: Hidden dimension of the language model (must match the
                ``last_hidden_state`` width so the tokens can be concatenated).

        Raises:
            ValueError: If ``cog_mode`` is not ``"full"`` or ``"cog_only"``.
            ValueError: If ``n_cog_tokens`` is not positive.
            RuntimeError: If the statistics cannot be computed (e.g. if the parameter is a meta tensor).
        """
        if self.cog_mode not in {"full", "cog_only"}:
            msg = f"Unsupported cog_mode '{self.cog_mode}' for VTC backbone."
            raise ValueError(msg)
        if self.n_cog_tokens <= 0:
            msg = "`n_cog_tokens` must be > 0 when `use_cog_tokens` is enabled."
            raise ValueError(msg)
        self.cog_emb = nn.Parameter(torch.randn(self.n_cog_tokens, feature_dim) * 0.02)

        print("\n[i] cog_emb initialized in VTCQwen3VLBackbone:")
        print(f"  Shape: {self.cog_emb.shape}")
        print(f"  Dtype: {self.cog_emb.dtype}")
        try:
            print(f"  Min: {self.cog_emb.min().item():.6f}")
            print(f"  Max: {self.cog_emb.max().item():.6f}")
            print(f"  Mean: {self.cog_emb.mean().item():.6f}")
            print(f"  Std: {self.cog_emb.std().item():.6f}")
        except RuntimeError as e:
            if "meta tensors" in str(e):
                print("  (Statistics not available - meta tensor)")
            else:
                raise

    def set_trainable_parameters(  # noqa: PLR0912
        self,
        tune_llm: bool,  # noqa: FBT001
        tune_visual: bool,  # noqa: FBT001
        tune_top_llm_layers: int = 0,
        print_params: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Configure which backbone sub-modules are trainable.

        Freezes the LLM and / or visual tower according to the flags, then
        optionally unfreezes the top ``tune_top_llm_layers`` LLM transformer
        blocks. The motion-module block (if present) is always kept trainable
        even when the rest of the visual tower is frozen.

        Args:
            tune_llm: Unfreeze the full LLM (language model + lm_head).
            tune_visual: Unfreeze the full visual tower (except the motion
                block, which is always trainable).
            tune_top_llm_layers: Number of top LLM transformer layers to unfreeze
                in addition to the ``tune_llm`` flag.
            print_params: Log a parameter count summary and list individually
                trainable parameter names.
        """
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.qwen_model.model.language_model.requires_grad_(False)  # noqa: FBT003
            self.qwen_model.lm_head.requires_grad_(False)  # noqa: FBT003
        if not tune_visual:
            self.qwen_model.model.visual.requires_grad_(False)  # noqa: FBT003
            # Unfreeze motion module block even when visual is frozen
            if (
                hasattr(self.qwen_model.model.visual, "motion_block")
                and self.qwen_model.model.visual.motion_block is not None
            ):
                self.qwen_model.model.visual.motion_block.requires_grad_(True)  # noqa: FBT003

        if tune_top_llm_layers > 0:
            for layer in self.qwen_model.model.language_model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # Unfreeze lm_head if top layers are trainable, since it depends on the last layer
            self.qwen_model.lm_head.requires_grad_(True)  # noqa: FBT003

        if print_params:
            print("=" * 50)
            print(f"[i] Tune backbone llm: {self.tune_llm}")
            print(f"[i] Tune backbone vision tower: {self.tune_visual}")
            if tune_top_llm_layers > 0:
                print(f"[i] Tune top {tune_top_llm_layers} LLM layers")
            print("=" * 50 + "\n")

            trainable_params = 0
            total_params = 0
            trainable_param_names = []
            for name, p in self.named_parameters():
                total_params += p.numel()
                if p.requires_grad:
                    trainable_params += p.numel()
                    trainable_param_names.append(name)

            print(
                f"[i] Backbone trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)",
            )

            if not tune_llm and not tune_visual:
                for name in trainable_param_names:
                    print(f"[i] Backbone trainable parameter: {name}")
            if trainable_params == 0:
                print("[w] No backbone trainable parameters found.")

    def _process_moss_features(  # noqa: PLR0914
        self,
        moss_feats: torch.Tensor,
        moss_meta: tuple[int, int, int, int, int],
    ) -> torch.Tensor:
        """Process saved motion module features for vl_input injection: spatial pool + projection.

        Args:
            moss_feats: (B, T, V_moss, P, D) raw motion module output at vision encoder layer
            moss_meta: (true_batch, num_frames, num_moss_views, H, W)

        Returns:
            (B, num_moss_tokens, llm_hidden_dim) pooled and projected motion module tokens

        Raises:
            RuntimeError: If ``moss_proj`` is not initialized.
        """
        batch_size, tokens, v_moss, height, width = moss_meta
        depth = moss_feats.shape[-1]

        # (B, T, V, H, W, D) -> (B*V, D, T, H, W)
        moss_feats = moss_feats.reshape(batch_size, tokens, v_moss, height, width, depth)
        moss_feats = moss_feats.permute(0, 2, 5, 1, 3, 4).contiguous()
        moss_feats = moss_feats.reshape(batch_size * v_moss, depth, tokens, height, width)

        if self.moss_spatial_conv is not None:
            batch_view, d_moss, t_moss, h_moss, w_moss = moss_feats.shape
            moss_feats = moss_feats.permute(0, 2, 1, 3, 4).reshape(batch_view * t_moss, d_moss, h_moss, w_moss)
            moss_feats = self.moss_spatial_conv(moss_feats)
            h_out, w_out = moss_feats.shape[2], moss_feats.shape[3]
            moss_feats = moss_feats.reshape(batch_view, t_moss, d_moss, h_out, w_out).permute(0, 2, 1, 3, 4)
        else:
            pool_h = max(1, height // 4)
            pool_w = max(1, width // 4)
            moss_feats = F.adaptive_avg_pool3d(moss_feats, (tokens, pool_h, pool_w))

        # Flatten to tokens: (B*V, D, T*S) -> (B, V*T*S, D)
        moss_feats = moss_feats.reshape(batch_size * v_moss, depth, -1).permute(0, 2, 1).contiguous()
        moss_feats = moss_feats.reshape(batch_size, -1, depth)

        if self.moss_proj is None:
            msg = "moss_proj is not initialized"
            raise RuntimeError(msg)
        return self.moss_proj(moss_feats)

    @staticmethod
    def prepare_input(batch: dict[str, Any]) -> BatchFeature:
        """Wrap a raw dict of tensors in a :class:`~transformers.BatchFeature`.

        Args:
            batch: Dict mapping feature names to tensors (e.g. ``input_ids``,
                ``attention_mask``, ``pixel_values``).

        Returns:
            A :class:`~transformers.BatchFeature` wrapping the same data.
        """
        return BatchFeature(data=batch)

    def _forward_qwen_with_cog_tokens(  # noqa: C901, PLR0912, PLR0914, PLR0915
        self,
        qwen_input: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the VTC-Qwen3-VL LM forward pass with appended cognition tokens.

        Embeds ``input_ids``, patches in image / video tokens, optionally injects
        motion-module tokens before the image tokens (``moss_proj`` path), appends
        ``n_cog_tokens`` learnable query embeddings to the end of the sequence,
        recomputes 3-D RoPE position IDs for the extended sequence, and runs the
        language-model trunk.

        Args:
            qwen_input: Dict containing the preprocessed VLM inputs, e.g.
                ``input_ids``, ``attention_mask``, ``pixel_values``,
                ``image_grid_thw``, ``image_wise_encoding``, ``num_views``,
                ``num_frames``.  All keys are optional except ``input_ids`` or
                ``inputs_embeds`` (exactly one must be provided).

        Returns:
            Tuple ``(last_hidden_state, attention_mask)`` where
            ``last_hidden_state`` has shape ``(B, L_extended, hidden_dim)`` and
            ``attention_mask`` covers the full extended sequence.

        Raises:
            ValueError: If both or neither of ``input_ids`` / ``inputs_embeds``
                are provided.
            ValueError: If ``input_ids`` or ``attention_mask`` is missing when
                motion-module tokens need to be injected.
            ValueError: If ``attention_mask`` or ``input_ids`` is missing during
                the cognition-token forward pass.
        """
        # Unwrap the qwen_input dictionary
        input_ids = qwen_input.get("input_ids")
        attention_mask = qwen_input.get("attention_mask")
        position_ids = qwen_input.get("position_ids")
        past_key_values = qwen_input.get("past_key_values")
        inputs_embeds = qwen_input.get("inputs_embeds")
        pixel_values = qwen_input.get("pixel_values")
        image_grid_thw = qwen_input.get("image_grid_thw")
        video_grid_thw = qwen_input.get("video_grid_thw")
        cache_position = qwen_input.get("cache_position")
        image_wise_encoding = qwen_input.get("image_wise_encoding")
        num_views = qwen_input.get("num_views")
        num_frames = qwen_input.get("num_frames")

        if (input_ids is None) ^ (inputs_embeds is not None):
            msg = "You must specify exactly one of input_ids or inputs_embeds"
            raise ValueError(msg)

        if inputs_embeds is None:
            inputs_embeds = self.qwen_model.model.get_input_embeddings()(input_ids)

        device = inputs_embeds.device

        # Build motion module kwargs for get_image_features
        moss_kwargs = {}
        if num_frames is not None:
            moss_kwargs["num_frames"] = int(num_frames[0]) if isinstance(num_frames, torch.Tensor) else int(num_frames)
        if num_views is not None:
            moss_kwargs["num_views"] = int(num_views[0]) if isinstance(num_views, torch.Tensor) else int(num_views)

        image_embeds, deepstack_image_embeds = self.qwen_model.model.get_image_features(
            pixel_values,
            image_grid_thw,
            **moss_kwargs,
        )
        image_embeds = torch.cat(image_embeds, dim=0).to(device, inputs_embeds.dtype)
        image_mask, _ = self.qwen_model.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds

        # Extract and process motion module features for vl_input injection
        moss_tokens = None
        n_motion_tokens = 0
        motion_insert_pos = 0
        if self.moss_proj is not None:
            visual = self.qwen_model.model.visual
            if visual._moss_features is not None:  # noqa: SLF001
                moss_tokens = self._process_moss_features(
                    visual._moss_features,  # noqa: SLF001
                    visual._moss_meta,  # noqa: SLF001
                ).to(inputs_embeds.dtype)
                n_motion_tokens = moss_tokens.shape[1]
                visual._moss_features = None  # noqa: SLF001
                visual._moss_meta = None  # noqa: SLF001

        bsz = inputs_embeds.size(0)
        placeholder_token_id = 248068

        # Insert motion module tokens BEFORE image tokens (for causal attention benefit)
        # Sequence: [text | motion module | images | trailing] + [meta_queries]
        if moss_tokens is not None:
            if input_ids is None:
                msg = "input_ids is required for motion token injection"
                raise ValueError(msg)
            if attention_mask is None:
                msg = "attention_mask is required for motion token injection"
                raise ValueError(msg)
            image_token_id = self.qwen_model.model.config.image_token_id
            first_img_pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0][0].item()
            motion_insert_pos = first_img_pos

            text_emb = inputs_embeds[:, :first_img_pos, :]
            image_emb = inputs_embeds[:, first_img_pos:, :]
            inputs_embeds = torch.cat([text_emb, moss_tokens, image_emb], dim=1)

            text_att = attention_mask[:, :first_img_pos]
            image_att = attention_mask[:, first_img_pos:]
            moss_att = torch.ones(bsz, n_motion_tokens, dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([text_att, moss_att, image_att], dim=1)

            text_ids = input_ids[:, :first_img_pos]
            image_ids = input_ids[:, first_img_pos:]
            moss_ids = torch.full(
                (bsz, n_motion_tokens),
                placeholder_token_id,
                dtype=input_ids.dtype,
                device=device,
            )
            input_ids = torch.cat([text_ids, moss_ids, image_ids], dim=1)

            if visual_pos_masks is not None:
                text_vis = visual_pos_masks[:, :first_img_pos]
                image_vis = visual_pos_masks[:, first_img_pos:]
                moss_vis = torch.zeros(
                    bsz,
                    n_motion_tokens,
                    dtype=visual_pos_masks.dtype,
                    device=device,
                )
                visual_pos_masks = torch.cat([text_vis, moss_vis, image_vis], dim=1)

        # Append cognition token embeddings
        meta_raw = self.cog_emb.to(inputs_embeds.dtype).unsqueeze(0).expand(bsz, -1, -1)
        full_emb = torch.cat([inputs_embeds, meta_raw], dim=1)

        # Extend attention_mask for cognition tokens
        if attention_mask is None:
            msg = "attention_mask is required for cog-token forward pass"
            raise ValueError(msg)
        meta_ones = torch.ones(bsz, self.n_cog_tokens, dtype=attention_mask.dtype, device=device)
        full_att_mask = torch.cat([attention_mask, meta_ones], dim=1)

        # Extend visual_pos_masks for cognition tokens
        if visual_pos_masks is not None:
            vis_pad = torch.zeros(
                bsz,
                self.n_cog_tokens,
                dtype=visual_pos_masks.dtype,
                device=device,
            )
            visual_pos_masks = torch.cat([visual_pos_masks, vis_pad], dim=1)

        if input_ids is None:
            msg = "input_ids is required for cog-token forward pass"
            raise ValueError(msg)
        meta_ids = torch.full(
            (bsz, self.n_cog_tokens),
            placeholder_token_id,
            dtype=input_ids.dtype,
            device=device,
        )
        extended_input_ids = torch.cat([input_ids, meta_ids], dim=1)

        # ── 3D position_ids (mirrors Qwen3VLModel.forward) ──
        if position_ids is None:
            attention_mask_tensor = (
                full_att_mask if not isinstance(full_att_mask, dict) else full_att_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:  # noqa: PLR2004
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor /= torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do assisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (extended_input_ids is not None and extended_input_ids.shape[1] != 1)
                or (full_emb is not None and full_emb.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.qwen_model.model.rope_deltas is None:
                position_ids, rope_deltas = self.qwen_model.model.get_rope_index(
                    extended_input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.qwen_model.model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = full_emb.shape
                delta = (
                    (cache_position[0] + self.qwen_model.model.rope_deltas).to(device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None and isinstance(delta, torch.Tensor):
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # ── Language model forward ──
        # Pass motion_drop_info so LayerWrapper drops motion module tokens at internal_projection
        motion_drop_info = None
        if n_motion_tokens > 0 and self.motion_drop:
            motion_drop_info = {"start": motion_insert_pos, "count": n_motion_tokens}

        outputs = self.qwen_model.model.language_model(
            input_ids=extended_input_ids,
            position_ids=position_ids,
            attention_mask=full_att_mask,
            past_key_values=past_key_values,
            inputs_embeds=full_emb,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            output_hidden_states=True,
            image_wise_encoding=image_wise_encoding,
            num_views=num_views,
            motion_drop_info=motion_drop_info,
        )

        return outputs.last_hidden_state, full_att_mask

    def forward_qwen(
        self,
        vl_input: BatchFeature,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor, torch.Tensor | None]:
        """Run the Qwen3-VL backbone and return features, attention mask, and image mask.

        Dispatches to :meth:`_forward_qwen_with_cog_tokens` when
        ``use_cog_tokens`` is enabled (VTC path); otherwise runs the model
        directly and extracts hidden states at ``select_layers``.

        In ``cog_only`` mode only the last ``n_cog_tokens`` hidden states are
        returned; in ``full`` mode the whole sequence is returned.

        Args:
            vl_input: Preprocessed VLM feature dict (output of the RLDX
                data-collator or :meth:`prepare_input`).

        Returns:
            Tuple ``(features, attention_mask, image_mask)`` where

            - ``features``: ``(B, T_out, hidden_dim)`` backbone embeddings
              (or a list of per-layer tensors when multiple ``select_layers``
              are configured without cog tokens).
            - ``attention_mask``: ``(B, T_out)`` boolean / int64 mask.
            - ``image_mask``: ``(B, L)`` boolean mask of image-token positions,
              or ``None`` when ``input_ids`` is absent.
        """
        if "pixel_values" in vl_input and vl_input["pixel_values"].ndim == 3:  # noqa: PLR2004
            pv = vl_input["pixel_values"]
            vl_input["pixel_values"] = pv.reshape(-1, pv.shape[-1])

        if "image_grid_thw" in vl_input and vl_input["image_grid_thw"].ndim == 3:  # noqa: PLR2004
            grid = vl_input["image_grid_thw"]
            vl_input["image_grid_thw"] = grid.reshape(-1, 3)

        # Generate image_mask from input_ids
        image_mask = None
        if "input_ids" in vl_input:
            image_token_id = self.qwen_model.model.config.image_token_id
            image_mask = vl_input["input_ids"] == image_token_id  # [B, L]

        if self.use_cog_tokens:
            last_hs, attn_mask = self._forward_qwen_with_cog_tokens(dict(vl_input))

            if self.cog_mode == "cog_only":
                features = last_hs[:, -self.n_cog_tokens :, :]
                attn_mask = attn_mask[:, -self.n_cog_tokens :]
                if image_mask is not None:
                    image_mask = image_mask[:, -self.n_cog_tokens :]
            else:
                features = last_hs

            features = self.qwen_linear(features)
            return features, attn_mask, image_mask
        qwen_output = self.qwen_model(**vl_input, output_hidden_states=True, return_dict=True)
        attn_mask = vl_input["attention_mask"]

        hs = qwen_output.hidden_states
        selected_indices = [self._hs_idx(i) for i in self.select_layers]
        feats = [hs[idx] for idx in selected_indices]

        linearized = [self.qwen_linear(feat) for feat in feats]
        features = linearized[0] if len(linearized) == 1 else linearized
        return features, attn_mask, image_mask

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        """Extract backbone features from a VLM input batch.

        Filters the input to the keys consumed by the Qwen3-VL model, runs
        :meth:`forward_qwen`, and returns the result as a
        :class:`~transformers.BatchFeature`.

        Args:
            vl_input: Full VLM feature dict produced by the RLDX data-collator.
                Must contain at minimum ``input_ids``, ``attention_mask``,
                ``pixel_values``, ``image_grid_thw``, ``image_wise_encoding``,
                and ``num_views``; ``num_frames`` is passed through when present.

        Returns:
            A :class:`~transformers.BatchFeature` with three entries:

            - ``backbone_features``: ``(B, T_out, hidden_dim)`` float tensor.
            - ``backbone_attention_mask``: ``(B, T_out)`` attention mask.
            - ``image_mask``: ``(B, L)`` image-token boolean mask.
        """
        keys_to_use = [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "image_wise_encoding",
            "num_views",
        ]
        filtered = {k: vl_input[k] for k in keys_to_use}
        if "num_frames" in vl_input:
            filtered["num_frames"] = vl_input["num_frames"]
        vl_input = BatchFeature(data=filtered)
        outputs, attention_mask, image_mask = self.forward_qwen(vl_input)

        return BatchFeature(
            data={
                "backbone_features": outputs,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            },
        )
