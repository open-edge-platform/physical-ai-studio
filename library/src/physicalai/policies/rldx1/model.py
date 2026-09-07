# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)
# ruff: noqa: T201

"""RLDX model components: action head (RLDXActionModel) and full VLA model (RLDX)."""

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file
from torch import nn
from torch.distributions import Beta
from transformers.feature_extraction_utils import BatchFeature

from physicalai.policies.base import Model
from physicalai.policies.components.nn import CategorySpecificMLP, MultiEmbodimentActionEncoder
from physicalai.policies.rldx1.components.action_model.msat import MSAT
from physicalai.policies.rldx1.components.action_model.physics import init_physics_params_near_zero
from physicalai.policies.rldx1.components.action_model.physics_head import NoOpPhysicsHead, PhysicsHead
from physicalai.policies.rldx1.components.backbone.adapter import VTCQwen3VLBackbone


def compute_video_window_offsets(video_length: int, video_stride: int) -> list[int]:
    """Return the VTC video-window frame offsets relative to the current timestep.

    Returns:
        Integer frame offsets, e.g. ``[-6, -4, -2, 0]`` for ``video_length=4,
        video_stride=2``.
    """
    return [(i - (video_length - 1)) * video_stride for i in range(video_length)]


def compute_action_delta_indices(chunk_size: int) -> list[int]:
    """Return the action-chunk indices relative to the current timestep.

    Returns:
        ``list(range(chunk_size))``.
    """
    return list(range(chunk_size))


class RLDXActionModel(nn.Module):
    """Action head component for flow matching diffusion policy."""

    def __init__(  # noqa: PLR0915, PLR0913, PLR0917
        self,
        hidden_size: int,
        input_embedding_dim: int,
        backbone_embedding_dim: int,
        max_action_dim: int,
        action_horizon: int,
        num_inference_timesteps: int,
        max_num_embodiments: int,
        max_state_dim: int,
        use_vlln: bool,  # noqa: FBT001
        add_pos_embed: bool,  # noqa: FBT001
        max_seq_len: int,
        state_dropout_prob: float,
        state_additive_noise_scale: float,
        noise_beta_alpha: float,
        noise_beta_beta: float,
        noise_s: float,
        num_timestep_buckets: int,
        diffusion_model_cfg: dict,
        tune_projector: bool,  # noqa: FBT001
        tune_diffusion_model: bool,  # noqa: FBT001
        tune_vlln: bool,  # noqa: FBT001
        backbone_trainable_params_fp32: bool,  # noqa: FBT001
        action_model_use_lora: bool = False,  # noqa: FBT001, FBT002
        action_model_lora_rank: int = 64,
        action_model_lora_alpha: int = 64,
        action_model_lora_dropout: float = 0.0,
        action_model_lora_target_modules: list[str] | None = None,
        use_physics: bool = False,  # noqa: FBT001, FBT002
        physics_dim: int = 0,
        physics_use_flow_matching: bool = True,  # noqa: FBT001, FBT002
        physics_delta_indices: list[int] | None = None,
        physics_loss_weight: float = 0.1,
        physics_dropout_prob: float = 0.0,
        gradient_checkpointing: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize action model components (encoders, diffusion model, physics head)."""
        super().__init__()
        self.hidden_size = hidden_size
        self.input_embedding_dim = input_embedding_dim
        self.add_pos_embed = add_pos_embed
        self.noise_s = noise_s
        self.backbone_trainable_params_fp32 = backbone_trainable_params_fp32
        self.action_model_use_lora = action_model_use_lora
        self.action_model_lora_rank = action_model_lora_rank
        self.action_model_lora_alpha = action_model_lora_alpha
        self.action_model_lora_dropout = action_model_lora_dropout
        self.action_model_lora_target_modules = action_model_lora_target_modules or [
            "vl_qkv",
            "vl_proj",
            "sa_qkv",
            "sa_proj",
            "p_qkv",
            "p_proj",
            "linear1",
            "linear2",
        ]

        # Initialize MSAT from config
        diffusion_model_cfg.setdefault("attention_head_dim", 64)
        diffusion_model_cfg.setdefault("depth_multi_stream", 4)
        diffusion_model_cfg.setdefault("depth_single_stream", 8)
        diffusion_model_cfg.setdefault("dropout", 0.2)
        diffusion_model_cfg.setdefault("num_attention_heads", 24)
        diffusion_model_cfg.setdefault("output_dim", 1024)
        diffusion_model_cfg.setdefault("positional_embeddings", "rope_sa_only")
        diffusion_model_cfg.setdefault("rope_theta", 10000.0)
        diffusion_model_cfg.setdefault("temb_type", "input_token")
        diffusion_model_cfg.setdefault("gradient_checkpointing", gradient_checkpointing)
        diffusion_model_cfg.setdefault("action_model_max_seq_len", 1024)
        diffusion_model_cfg.setdefault("pre_norm", "layer_norm")
        diffusion_model_cfg.setdefault("qk_norm", "rms_norm")
        diffusion_model_cfg.setdefault("sa_dim", input_embedding_dim)
        diffusion_model_cfg.setdefault("vl_dim", backbone_embedding_dim)

        # Strip unsupported triple-stream config keys before model construction.
        for _key in (
            "set_triple_stream_for_mq",
            "set_triple_stream_for_state",
            "state_dim",
            "action_dim",
            "mq_dim",
            "state_mlp_ratio",
            "action_mlp_ratio",
            "mq_mlp_ratio",
        ):
            diffusion_model_cfg.pop(_key, None)

        # Inject physics config
        diffusion_model_cfg["use_physics"] = use_physics
        diffusion_model_cfg["physics_dim"] = physics_dim
        self.model = MSAT(
            **diffusion_model_cfg,
        )
        self.action_dim = max_action_dim
        self.action_horizon = action_horizon
        self.num_inference_timesteps = num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = nn.LayerNorm(backbone_embedding_dim) if use_vlln else nn.Identity()

        if self.add_pos_embed:
            self.position_embedding = nn.Embedding(max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim)) if self.state_dropout_prob > 0 else None
        )

        # State noise parameters
        self.state_additive_noise_scale = state_additive_noise_scale
        self.beta_dist = Beta(
            noise_beta_alpha,
            noise_beta_beta,
            validate_args=False,
        )
        self.num_timestep_buckets = num_timestep_buckets

        # Physics (tactile/torque) stream
        self.use_physics = use_physics

        if self.use_physics and physics_dim > 0:
            embed_dim = self.input_embedding_dim
            msat_output_dim = diffusion_model_cfg.get("output_dim", 1024)
            physics_delta = physics_delta_indices or []
            effective_flow_matching = physics_use_flow_matching and sum(1 for d in physics_delta if d > 0) > 0

            self.physics: PhysicsHead | NoOpPhysicsHead = PhysicsHead(
                physics_dim=physics_dim,
                embed_dim=embed_dim,
                msat_output_dim=msat_output_dim,
                physics_delta_indices=physics_delta,
                physics_use_flow_matching=effective_flow_matching,
                physics_loss_weight=physics_loss_weight,
                action_horizon=self.action_horizon,
                physics_dropout_prob=physics_dropout_prob,
            )
            print("[Physics] Applying near-zero (exit-zero) initialization...")
            init_physics_params_near_zero(self)
        else:
            self.physics = NoOpPhysicsHead()

        self.set_trainable_parameters(
            tune_projector,
            tune_diffusion_model,
            tune_vlln,
        )

    def set_trainable_parameters(  # noqa: PLR0912
        self,
        tune_projector: bool,  # noqa: FBT001
        tune_diffusion_model: bool,  # noqa: FBT001
        tune_vlln: bool,  # noqa: FBT001
    ) -> None:
        """Set which submodules are trainable based on tuning flags."""
        # When LoRA is on, the diffusion model is no longer full-tuned —
        # LoRA adapters are the only trainable surface inside ``self.model``.
        # Override the flag here so callers passing
        # ``tune_diffusion_model=True`` from the config default still get
        # LoRA-only behaviour with a single source of truth.
        use_lora = self.action_model_use_lora
        if use_lora:
            tune_diffusion_model = False

        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)  # noqa: FBT003
            self.action_encoder.requires_grad_(False)  # noqa: FBT003
            self.action_decoder.requires_grad_(False)  # noqa: FBT003
            self.physics.requires_grad_(False)  # noqa: FBT003
            if self.add_pos_embed:
                self.position_embedding.requires_grad_(False)  # noqa: FBT003
            if self.state_dropout_prob > 0 and self.mask_token is not None:
                self.mask_token.requires_grad_(False)  # noqa: FBT003

        if use_lora:
            # Replaces the unconditional ``self.model.requires_grad_(False)``:
            # _apply_action_model_lora freezes the DiT first and then PEFT
            # marks only the injected LoRA params trainable.
            self._apply_action_model_lora()
        elif not tune_diffusion_model:
            self.model.requires_grad_(False)  # noqa: FBT003

        if not tune_vlln:
            self.vlln.requires_grad_(False)  # noqa: FBT003
        elif self.backbone_trainable_params_fp32:
            # Keep the trainable vlln in fp32. Its weight inits to 1.0, where the
            # bf16 ULP is 2**-7 ~= 0.0078 -- larger than a typical AdamW step
            # (~lr), so in-place bf16 updates round back to 1.0 and the gain
            # never moves. fp32 storage lets small updates accumulate.
            self.vlln.to(torch.float32)

        print(f"[MSAT] Tune action model projector: {self.tune_projector}")
        print(f"[MSAT] Tune action model diffusion model: {self.tune_diffusion_model}")
        print(f"[MSAT] Tune action model vlln: {self.tune_vlln}")
        print(f"[MSAT] Action model LoRA: {use_lora}")

        # Check if any parameters are still trainable. If not, _print a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln and not use_lora:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action model trainable parameters found.")

    def _apply_action_model_lora(self) -> None:
        """Inject PEFT LoRA adapters into the MSAT diffusion model.

        Freezes the entire MSAT first, then wraps the target Linear layers
        listed in ``self.action_model_lora_target_modules`` (PEFT marks the
        injected LoRA weights ``requires_grad=True``). Target names that don't
        exist in the current MSAT (e.g. ``p_qkv``/``p_proj`` when
        ``use_physics=False``) are filtered before the PEFT call so PEFT
        doesn't raise on a missing target.

        Raises:
            ImportError: If peft is not installed.
            ValueError: If none of the requested target modules exist in the MSAT.
        """
        try:
            from peft import LoraConfig, inject_adapter_in_model  # noqa: PLC0415
        except ImportError as e:
            msg = "LoRA requires peft. Install with: pip install peft"
            raise ImportError(msg) from e

        target_modules = list(self.action_model_lora_target_modules)

        # Keep only target names that actually appear in the MSAT. PEFT
        # matches by exact name OR ".{name}" suffix of the fully qualified
        # module path — mirror that here so we don't pass dead targets
        # (e.g. p_qkv when physics is disabled).
        module_names = [name for name, _ in self.model.named_modules()]

        def _present(target: str) -> bool:
            dot_target = f".{target}"
            return any(n == target or n.endswith(dot_target) for n in module_names)

        filtered = [t for t in target_modules if _present(t)]
        skipped = [t for t in target_modules if t not in filtered]
        if skipped:
            print(f"[ActionModel LoRA] Skipping absent target modules: {skipped}")
        if not filtered:
            msg = f"[ActionModel LoRA] None of the requested target modules {target_modules} exist in the MSAT."
            raise ValueError(msg)

        # Freeze the entire MSAT; LoRA weights will be marked trainable by PEFT.
        self.model.requires_grad_(False)  # noqa: FBT003

        lora_config = LoraConfig(
            r=int(self.action_model_lora_rank),
            lora_alpha=int(self.action_model_lora_alpha),
            lora_dropout=float(self.action_model_lora_dropout),
            bias="none",
            target_modules=filtered,
        )
        inject_adapter_in_model(lora_config, self.model)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        ratio = (100.0 * trainable / total) if total > 0 else 0.0
        print(
            f"[ActionModel LoRA] target_modules={filtered}, "
            f"r={lora_config.r}, alpha={lora_config.lora_alpha}, "
            f"dropout={lora_config.lora_dropout}",
        )
        print(f"[ActionModel LoRA] trainable params: {trainable} / {total} ({ratio:.2f}%)")

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sample random flow-matching timesteps from the Beta distribution.

        Returns:
            Tensor of shape ``(batch_size,)`` with values in ``[0, noise_s]``.
        """
        return (1 - self.beta_dist.sample([batch_size]).to(device, dtype=dtype)) * self.noise_s

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        """Apply vision-language layer norm to backbone features.

        Returns:
            Updated :class:`~transformers.feature_extraction_utils.BatchFeature`
            with normed ``backbone_features``.
        """
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(  # noqa: PLR0915, PLR0914
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
    ) -> BatchFeature:
        """Forward pass through the action model.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss

        Raises:
            RuntimeError: If ``mask_token`` is ``None`` when ``state_dropout_prob > 0``.
        """
        # Set frozen modules to eval

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Per-sample embodiment id (shape [B]). It is carried through the data
        # batch, not stored on the model, because RLDX-1 is a multi-embodiment
        # model: a single training batch mixes samples from different robots, so
        # each row may route through a different CategorySpecificLinear projector
        # slot (W[embodiment_id]). PAS v1 fine-tunes one embodiment, so every row
        # holds the same constant, but the per-sample contract is kept to match
        # upstream verbatim (and to support mixed-embodiment batches unchanged).
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features.
        if self.state_dropout_prob > 0:
            do_dropout = torch.rand(state_features.shape[0], device=state_features.device) < self.state_dropout_prob
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            if self.mask_token is None:
                msg = "mask_token must be initialized when state_dropout_prob > 0"
                raise RuntimeError(msg)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features.
        if self.training and self.state_additive_noise_scale > 0:
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise  # noqa: PLR6104

        # Embed noised action trajectory.
        actions = action_input.action
        batch_size = actions.shape[0]
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t_raw = self.sample_time(batch_size, device=actions.device, dtype=actions.dtype)  # (B,)

        prefix_mask = None
        t_tok = t_raw.unsqueeze(1).expand(-1, self.action_horizon).contiguous()
        t = t_raw[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions

        velocity = actions - noise
        action_features = self.action_encoder(noisy_trajectory, t_tok, embodiment_id)

        # Maybe add position embedding.
        if self.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs  # noqa: PLR6104

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)

        encoder_attention_mask = backbone_output.get("backbone_attention_mask", None)
        # Only pass mask to MSAT when there are actually masked positions
        if encoder_attention_mask is not None and encoder_attention_mask.all():
            encoder_attention_mask = None

        # Encode physics signal
        physics_embs, physics_attn_mask, physics_velocity = self.physics.prepare_train(
            action_input,
            t_raw,
        )

        # MSAT global temb uses the scalar postfix τ per sample. Per-token
        # time has already been threaded through action_encoder above.
        model_output, _ = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            timestep=t_raw,
            return_all_hidden_states=True,
            encoder_attention_mask=encoder_attention_mask,
            physics_embs=physics_embs,
            physics_attention_mask=physics_attn_mask,
        )

        # When physics is enabled, model_output is a dict {"action": ..., "physics": ...}
        if isinstance(model_output, dict):
            action_model_output = model_output["action"]
            physics_model_output = model_output["physics"]
        else:
            action_model_output = model_output
            physics_model_output = None

        pred = self.action_decoder(action_model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        action_mask = action_input.action_mask
        loss_mask = action_mask
        if prefix_mask is not None:
            postfix = (~prefix_mask).to(dtype=action_mask.dtype).unsqueeze(-1)
            loss_mask = action_mask * postfix
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * loss_mask
        loss = action_loss.sum() / (loss_mask.sum() + 1e-6)

        results = {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

        # Physics prediction loss (flow matching only; conditioning-only mode has no physics loss)
        physics_loss = self.physics.compute_loss(
            physics_model_output,
            physics_velocity,
            action_mask,
            physics_attn_mask,
        )
        if physics_loss is not None:
            if not isinstance(self.physics, PhysicsHead):
                msg = "Physics loss is not None but self.physics is not a PhysicsHead."
                raise RuntimeError(msg)
            loss = loss + self.physics.physics_loss_weight * physics_loss  # noqa: PLR6104
            results["loss"] = loss
            results["physics_loss"] = physics_loss

        return BatchFeature(data=results)

    def _encode_features(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
    ) -> BatchFeature:
        """Encode features for the action model.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, state_horizon, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        # Per-sample embodiment id (shape [B]); see forward() above for why this
        # rides the data batch instead of living on the model.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    def get_action_with_features(  # noqa: PLR0914
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature | None = None,
    ) -> BatchFeature:
        """Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
            action_input: Optional, used for physics conditioning.

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, state_horizon, input_embedding_dim]
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        dtype = vl_embeds.dtype
        horizon = self.action_horizon

        # ─── Physics ────────────────────────────────────────────────────────
        with torch.no_grad():
            phys_state = self.physics.prepare_inference(action_input, batch_size, device, dtype)

        # ─── Initial noise ──────────────────────────────────────────────────
        actions = torch.randn(
            size=(batch_size, horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )

        # Use custom denoising timesteps if set, otherwise uniform spacing.
        if hasattr(self, "denoising_timesteps") and self.denoising_timesteps is not None:
            timesteps_list = [*list(self.denoising_timesteps), 1.0]  # type: ignore[arg-type]
        else:
            n = self.num_inference_timesteps
            timesteps_list = [*[t / float(n) for t in range(n)], 1.0]

        encoder_attention_mask = backbone_output.get("backbone_attention_mask", None)
        if encoder_attention_mask is not None and encoder_attention_mask.all():
            encoder_attention_mask = None

        def _dit_forward(
            x_tau: torch.Tensor,
            t_scalar: torch.Tensor,
            t_tok: torch.Tensor,
        ) -> dict | torch.Tensor:
            """One MSAT forward at the current Euler step.

            Returns:
                MSAT output: dict with ``'action'``/``'physics'`` when physics is on,
                raw tensor otherwise.
            """
            af = self.action_encoder(x_tau, t_tok, embodiment_id)
            if self.add_pos_embed:
                pos_ids = torch.arange(af.shape[1], dtype=torch.long, device=device)
                af += self.position_embedding(pos_ids).unsqueeze(0)
            sa = torch.cat((state_features, af), dim=1)
            phy_embs = self.physics.build_tokens(phys_state, t_scalar)
            return self.model(
                hidden_states=sa,
                encoder_hidden_states=vl_embeds,
                timestep=t_scalar,
                encoder_attention_mask=encoder_attention_mask,
                physics_embs=phy_embs,
                physics_attention_mask=phys_state.attn_mask,
            )

        # Run denoising steps.
        for i in range(len(timesteps_list) - 1):
            t_cont = float(timesteps_list[i])
            dt = float(timesteps_list[i + 1] - timesteps_list[i])

            t_scalar = torch.full((batch_size,), t_cont, device=device, dtype=dtype)
            t_tok = t_scalar.unsqueeze(1).expand(-1, horizon).clone()

            with torch.no_grad():
                mo = _dit_forward(actions, t_scalar, t_tok)
                ao = mo["action"] if isinstance(mo, dict) else mo
                pred_velocity = self.action_decoder(ao, embodiment_id)[:, -horizon:]
                model_output = mo

            # Euler step.
            with torch.no_grad():
                actions += dt * pred_velocity
                # Re-lock prefix between Euler steps (trained mode only).
                phys_state = self.physics.update_state(phys_state, model_output, dt)

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            },
        )

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
        )

    @property
    def device(self) -> torch.device:
        """Device of the model parameters."""
        return next(iter(self.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the model parameters."""
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:  # noqa: PLR6301
        """Prepare input batch for the action model.

        Returns:
            :class:`~transformers.feature_extraction_utils.BatchFeature` wrapping the batch dict.
        """
        return BatchFeature(data=batch)


class Rldx1Model(Model):
    """RLDX: Vision-Language-Action model with backbone."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        base_model_path: str | None = "RLWRLD/RLDX-1-PT",
        attn_implementation: Literal["sdpa", "flash_attention_2"] = "sdpa",
        # action model configs
        hidden_size: int = 1024,
        input_embedding_dim: int = 1536,
        backbone_embedding_dim: int = 4096,
        max_action_dim: int = 64,
        max_state_dim: int = 64,
        action_horizon: int = 16,
        num_inference_timesteps: int = 4,
        max_num_embodiments: int = 36,
        # VTC video window (see observation_delta_indices)
        video_length: int = 4,
        video_stride: int = 2,
        use_vlln: bool = False,  # noqa: FBT001, FBT002
        add_pos_embed: bool = True,  # noqa: FBT001, FBT002
        max_seq_len: int = 1024,
        state_dropout_prob: float = 0.0,
        state_additive_noise_scale: float = 0.0,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        num_timestep_buckets: int = 1000,
        diffusion_model_cfg: dict | None = None,
        tune_projector: bool = True,  # noqa: FBT001, FBT002
        tune_diffusion_model: bool = True,  # noqa: FBT001, FBT002
        tune_vlln: bool = True,  # noqa: FBT001, FBT002
        gradient_checkpointing: bool = False,  # noqa: FBT001, FBT002
        # action lora config
        action_model_use_lora: bool = False,  # noqa: FBT001, FBT002
        action_model_lora_rank: int = 64,
        action_model_lora_alpha: int = 64,
        action_model_lora_dropout: float = 0.0,
        action_model_lora_target_modules: list[str] | None = None,
        # physics config
        use_physics: bool = False,  # noqa: FBT001, FBT002
        physics_dim: int = 0,
        physics_use_flow_matching: bool = True,  # noqa: FBT001, FBT002
        physics_delta_indices: list[int] | None = None,
        physics_loss_weight: float = 0.1,
        physics_dropout_prob: float = 0.0,
        # backbone configs
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        n_cog_tokens: int = 64,
        tune_llm: bool = False,  # noqa: FBT001, FBT002
        tune_visual: bool = False,  # noqa: FBT001, FBT002
        select_layer: int = 18,
        load_bf16: bool = True,  # noqa: FBT001, FBT002
        tune_top_llm_layers: int = 4,
        backbone_trainable_params_fp32: bool = False,  # noqa: FBT001, FBT002
        backbone_use_lora: bool = False,  # noqa: FBT001, FBT002
        backbone_lora_rank: int = 64,
        backbone_lora_alpha: int = 64,
        backbone_lora_dropout: float = 0.0,
        backbone_lora_num_layers: int = -1,
        backbone_lora_target_modules: list[str] | None = None,
        # motion module configs
        use_motion: bool = False,  # noqa: FBT001, FBT002
        motion_insert_layer: int = 9,
        motion_d_hid: int = 512,
        motion_window: tuple[int, int, int] = (5, 9, 9),
        motion_ext_chnls: tuple[int] = (256,),
        motion_int_chnls: tuple[int, int, int] = (256, 256, 512),
        motion_corr_func: str = "cosine",
        motion_n_encoders: int = 1,
        motion_use_layerscale: bool = False,  # noqa: FBT001, FBT002
        motion_layerscale_init: float = 1e-5,
        motion_use_layernorm: bool = False,  # noqa: FBT001, FBT002
        motion_use_syncbn: bool = False,  # noqa: FBT001, FBT002
        motion_injection_point: str = "vision_encoder",
        motion_pool_type: str = "avg",
        motion_drop: bool = True,  # noqa: FBT001, FBT002
        motion_gradient_check: bool = False,  # noqa: FBT001, FBT002
        cog_mode: str = "cog_only",
    ) -> None:
        """Build the RLDX backbone, action model, and optional LoRA adapters."""
        transformers_loading_kwargs = {"trust_remote_code": True}
        super().__init__()

        kwargs: dict[str, Any] = {}
        kwargs["use_cog_tokens"] = True
        kwargs["cog_mode"] = cog_mode
        kwargs["n_cog_tokens"] = n_cog_tokens
        print(f"\n[MSAT Configs] n_cog_tokens: {n_cog_tokens}")

        # Build motion module config if enabled
        if use_motion:
            kwargs["motion_config"] = {
                "use_motion": True,
                "motion_insert_layer": motion_insert_layer,
                "motion_d_hid": motion_d_hid,
                "motion_window": motion_window,
                "motion_ext_chnls": motion_ext_chnls,
                "motion_int_chnls": motion_int_chnls,
                "motion_corr_func": motion_corr_func,
                "motion_n_encoders": motion_n_encoders,
                "motion_use_layerscale": motion_use_layerscale,
                "motion_layerscale_init": motion_layerscale_init,
                "motion_use_layernorm": motion_use_layernorm,
                "motion_use_syncbn": motion_use_syncbn,
                "motion_injection_point": motion_injection_point,
                "motion_pool_type": motion_pool_type,
                "motion_drop": motion_drop,
                "motion_gradient_check": motion_gradient_check,
            }
            print(f"[motion module] Enabled with config: {kwargs['motion_config']}")

        if diffusion_model_cfg is None:
            diffusion_model_cfg = {}
        if backbone_lora_target_modules is None:
            backbone_lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        skip_pretrained_weights = base_model_path is not None
        self.backbone = VTCQwen3VLBackbone(
            model_name=model_name,
            tune_llm=tune_llm,
            tune_visual=tune_visual,
            select_layer=select_layer,
            load_bf16=load_bf16,
            tune_top_llm_layers=tune_top_llm_layers,
            trainable_params_fp32=backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
            skip_pretrained_weights=skip_pretrained_weights,
            attn_implementation=attn_implementation,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs,
        )

        # Initialize action model
        self.action_model = RLDXActionModel(
            hidden_size=hidden_size,
            input_embedding_dim=input_embedding_dim,
            backbone_embedding_dim=backbone_embedding_dim,
            max_action_dim=max_action_dim,
            action_horizon=action_horizon,
            num_inference_timesteps=num_inference_timesteps,
            max_num_embodiments=max_num_embodiments,
            max_state_dim=max_state_dim,
            use_vlln=use_vlln,
            add_pos_embed=add_pos_embed,
            max_seq_len=max_seq_len,
            state_dropout_prob=state_dropout_prob,
            state_additive_noise_scale=state_additive_noise_scale,
            noise_beta_alpha=noise_beta_alpha,
            noise_beta_beta=noise_beta_beta,
            noise_s=noise_s,
            num_timestep_buckets=num_timestep_buckets,
            diffusion_model_cfg=diffusion_model_cfg,
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
            tune_vlln=tune_vlln,
            backbone_trainable_params_fp32=backbone_trainable_params_fp32,
            action_model_use_lora=action_model_use_lora,
            action_model_lora_rank=action_model_lora_rank,
            action_model_lora_alpha=action_model_lora_alpha,
            action_model_lora_dropout=action_model_lora_dropout,
            action_model_lora_target_modules=action_model_lora_target_modules,
            use_physics=use_physics,
            physics_dim=physics_dim,
            physics_use_flow_matching=physics_use_flow_matching,
            physics_delta_indices=physics_delta_indices,
            physics_loss_weight=physics_loss_weight,
            physics_dropout_prob=physics_dropout_prob,
            gradient_checkpointing=gradient_checkpointing,
        )

        # Backbone (Qwen3 LLM) LoRA. Runs AFTER the action model is built
        # so its requires_grad bookkeeping (top-N freeze) is already done;
        # the LoRA injection then freezes the entire backbone, lets PEFT
        # mark only the adapter params trainable, and casts those to fp32
        # for NaN safety on the first optimizer step.
        if backbone_use_lora:
            self._apply_backbone_lora(
                num_layers=backbone_lora_num_layers,
                rank=backbone_lora_rank,
                alpha=backbone_lora_alpha,
                dropout=backbone_lora_dropout,
                target_modules=backbone_lora_target_modules,
            )

        self._chunk_size = action_horizon
        self._video_length = video_length
        self._video_stride = video_stride

    def _get_backbone_hidden_size(self) -> int:
        """Return the backbone LLM hidden size.

        Returns:
            Hidden size of the Qwen language model, or 4096 as a fallback.
        """
        if hasattr(self.backbone, "qwen_model"):
            return self.backbone.qwen_model.model.language_model.config.hidden_size
        return 4096

    def _apply_backbone_lora(
        self,
        *,
        num_layers: int,
        rank: int,
        alpha: int,
        dropout: float,
        target_modules: list[str],
    ) -> None:
        """Inject PEFT LoRA adapters into the backbone LLM layers (top-N or all).

        Sibling of ``RLDXActionModel._apply_action_model_lora``: mirrors the
        same plumbing but targets the Qwen3 LLM layers held under
        ``self.backbone.qwen_model.model.language_model.layers``. ``num_layers``
        picks the suffix to adapt: ``-1`` (or any negative) and any value
        ``>= total`` ⇒ all layers, ``0`` ⇒ no-op skip (logged), ``N > 0`` ⇒
        last ``N`` layers only.

        The function freezes the entire backbone first, then PEFT marks only
        the injected LoRA params trainable. Adapter params are immediately
        cast bf16 → fp32 to avoid NaN losses on the first optimizer step
        (mirrors VTC's ``trainable_params_fp32`` policy).

        Raises:
            ImportError: If peft is not installed.
        """
        try:
            from peft import LoraConfig, inject_adapter_in_model  # noqa: PLC0415
        except ImportError as e:
            msg = "LoRA requires peft. Install with: pip install peft"
            raise ImportError(msg) from e

        layers = self.backbone.qwen_model.model.language_model.layers
        total = len(layers)

        if num_layers == 0:
            print("[Backbone LoRA] backbone_lora_num_layers=0, skipping injection")
            return

        layers_to_transform = (
            list(range(total)) if num_layers < 0 or num_layers >= total else list(range(total - num_layers, total))
        )

        self.backbone.requires_grad_(False)  # noqa: FBT003

        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=target_modules,
            layers_to_transform=layers_to_transform,
            layers_pattern="layers",
        )
        inject_adapter_in_model(lora_cfg, self.backbone)

        # fp32 contract for LoRA adapter params (NaN safety: bf16 AdamW state
        # underflows on the first step). Filter on the ``lora_`` name segment
        # so a future non-LoRA trainable backbone param is not silently
        # promoted. Cast unconditionally — owning the contract here decouples
        # it from ``adapter.py:186-190``'s top-N pre-cast.
        n_cast = 0
        for pname, p in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            if not any(seg.startswith("lora_") for seg in pname.split(".")):
                continue
            p.data = p.data.to(torch.float32)
            n_cast += 1
        print(f"[Backbone LoRA] Ensured fp32 dtype on {n_cast} LoRA parameter tensors")

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in self.backbone.parameters())
        ratio = (100.0 * trainable / total_p) if total_p > 0 else 0.0
        print(
            f"[Backbone LoRA] layers_to_transform={layers_to_transform} (total={total}), "
            f"r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, "
            f"dropout={lora_cfg.lora_dropout}, target_modules={target_modules}",
        )
        print(f"[Backbone LoRA] trainable params: {trainable} / {total_p} ({ratio:.3f}%)")

    def prepare_input(self, inputs: dict) -> tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action model.

        Studio feeds pre-collated tensors via :class:`Rldx1Preprocessor`. Raw
        upstream ``vlm_content`` (the un-collated ``RLDXProcessor`` output) is
        not supported here -- collate it with the preprocessor first.

        Returns:
            Tuple of ``(backbone_inputs, action_inputs)`` :class:`~transformers.feature_extraction_utils.BatchFeature`.
        """
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_model.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x: object) -> object:
            if not torch.is_tensor(x):
                return x
            if torch.is_floating_point(x):  # type: ignore[arg-type]
                return x.to(self.device, dtype=self.dtype)  # type: ignore[attr-defined]
            return x.to(self.device)  # type: ignore[attr-defined]

        backbone_inputs = BatchFeature(data={k: to_device_with_dtype(v) for k, v in backbone_inputs.items()})
        action_inputs = BatchFeature(data={k: to_device_with_dtype(v) for k, v in action_inputs.items()})

        return backbone_inputs, action_inputs

    def get_action(self, inputs: dict | None = None, **_kwargs: object) -> torch.Tensor:
        """Run inference to predict an action chunk.

        Returns:
            Predicted action tensor of shape ``[B, action_horizon, action_dim]``.
        """
        if inputs is None:
            inputs = {}

        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)

        outputs = self.action_model.get_action(backbone_outputs, action_inputs)
        return outputs["action_pred"]

    @property
    def device(self) -> torch.device:
        """Device of the model parameters."""
        return next(iter(self.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the model parameters."""
        return next(iter(self.parameters())).dtype

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]] | torch.Tensor:
        """Dispatch between training loss and action prediction.

        Args:
            batch: Preprocessed batch dict.

        Returns:
            Training: ``(loss, loss_dict)``. Eval: predicted action tensor.
        """
        if self.training:
            return self.compute_loss(batch)
        return self.get_action(batch)

    def compute_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Run backbone + action model in training mode.

        Returns:
            :class:`~transformers.feature_extraction_utils.BatchFeature` with ``loss`` and auxiliary keys.
        """
        backbone_inputs, action_inputs = self.prepare_input(batch)
        backbone_outputs = self.backbone(backbone_inputs)

        outputs = self.action_model(backbone_outputs, action_inputs)
        loss = outputs["loss"]
        loss_dict: dict[str, torch.Tensor | float] = {
            key: float(value.detach())
            for key, value in outputs.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        }
        return loss, loss_dict

    @torch.no_grad()
    def compute_val_loss(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute deterministic action prediction MSE for validation.

        Runs the full denoising path via :meth:`get_action` and compares it to
        ground-truth actions from the preprocessed batch.

        Args:
            batch: Collated RLDX inputs dict (must include ``action`` during
                validation).

        Returns:
            ``(loss, loss_dict)`` where ``loss`` is action-space MSE.

        Raises:
            ValueError: If the batch does not contain ground-truth ``action``.
        """
        if "action" not in batch:
            msg = "Validation batch must include 'action' to compute action MSE."
            raise ValueError(msg)

        predicted = self.get_action(batch)
        target = batch["action"]

        min_len = min(predicted.shape[1], target.shape[1])
        min_dim = min(predicted.shape[2], target.shape[2])
        pred_trimmed = predicted[:, :min_len, :min_dim]
        target_trimmed = target[:, :min_len, :min_dim]

        if "action_mask" in batch:
            mask = batch["action_mask"][:, :min_len, :min_dim].to(dtype=pred_trimmed.dtype)
            sq_err = (pred_trimmed - target_trimmed).pow(2)
            denom = mask.sum().clamp_min(1.0)
            loss = (sq_err * mask).sum() / denom
        else:
            loss = F.mse_loss(pred_trimmed, target_trimmed)

        return loss, {"loss": float(loss.detach()), "action_mse": float(loss.detach())}

    @property
    def reward_delta_indices(self) -> None:
        """Reward indices (rewards not implemented)."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Indices of actions relative to the current timestep."""
        return compute_action_delta_indices(self._chunk_size)

    @property
    def observation_delta_indices(self) -> list[int]:
        """VTC video-window frame offsets relative to the current timestep.

        Returns the integer frame offsets for the multi-frame video window used
        by the RLDX-1 backbone, e.g. ``[-6, -4, -2, 0]`` for
        ``video_length=4, video_stride=2``. These are applied to every
        observation key (camera views and state) by
        ``reformat_dataset_to_match_policy``; the RLDX-1 transform slices state
        to the current (last) frame automatically.
        """
        return compute_video_window_offsets(self._video_length, self._video_stride)

    def load_sharded_weights(self, shard_files: list[Path]) -> None:
        """Load merged ``safetensors`` shards onto this already-built network in place.

        Call after ``__init__`` (backbone + action model, with LoRA already
        injected if requested) so the checkpoint fills in the real pretrained
        weights. Handles two mismatches between the checkpoint and the live
        module tree:

        - **LoRA base-layer renaming**: ``_apply_backbone_lora`` /
          ``_apply_action_model_lora`` (run inside ``__init__``) wrap target
          ``nn.Linear`` modules with PEFT, renaming ``foo.weight`` to
          ``foo.base_layer.weight``. The checkpoint predates that injection, so
          those keys are remapped onto the renamed params -- otherwise the
          pretrained weights silently fail to load and the LoRA base stays at
          its random init.
        - **``max_state_dim`` / ``max_action_dim`` shrinkage**: checkpoint
          tensors sized for the pretrained dims are sliced down to this
          instance's (smaller) dims, keeping the first-N pretrained weights.

        Args:
            shard_files: Local ``safetensors`` shard paths (e.g. from
                ``retrieve_safetensors_shards``).

        Raises:
            ValueError: If the checkpoint contains parameters this model does
                not expect (architecture mismatch).
        """
        state_dict: dict[str, torch.Tensor] = {}
        for shard in shard_files:
            state_dict.update(load_file(str(shard)))

        model_params = dict(self.named_parameters())

        for key in list(state_dict.keys()):
            if key in model_params:
                continue
            for suffix in (".weight", ".bias"):
                if key.endswith(suffix):
                    remapped_key = key[: -len(suffix)] + ".base_layer" + suffix
                    if remapped_key in model_params:
                        state_dict[remapped_key] = state_dict.pop(key)
                    break

        # Slice checkpoint tensors whose shape exceeds the model's (e.g. when
        # max_state_dim / max_action_dim is smaller than the pretrained 64).
        # Each mismatched dim is trimmed independently so the first-N weights
        # from the pretrained model are reused rather than discarded.
        for key, ckpt_tensor in list(state_dict.items()):
            if key in model_params:
                model_shape = model_params[key].shape
                ckpt_shape = ckpt_tensor.shape
                if model_shape != ckpt_shape and len(model_shape) == len(ckpt_shape):
                    slices = tuple(slice(0, s) for s in model_shape)
                    state_dict[key] = ckpt_tensor[slices].contiguous()
                    print(
                        f"[load_sharded_weights] Sliced checkpoint param {key} "
                        f"from {list(ckpt_shape)} to {list(model_shape)}.",
                    )

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if unexpected:
            msg = (
                f"Checkpoint shards contain {len(unexpected)} unexpected "
                f"parameter(s) (architecture mismatch), e.g. {unexpected[:5]}"
            )
            raise ValueError(msg)

        if missing:
            print(
                f"[load_sharded_weights] {len(missing)} parameter(s) not found in checkpoint "
                f"(expected to be non-persistent buffers or fresh LoRA adapters): {missing[:5]}",
            )
