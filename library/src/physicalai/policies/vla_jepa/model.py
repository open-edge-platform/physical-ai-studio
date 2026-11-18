# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native Qwen3-VL and DiT action policy; no LeRobot policy dependency."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Self

import torch
from safetensors import safe_open
from torch import Tensor
from transformers import Qwen3VLConfig, Qwen3VLModel, VJEPA2Config, VJEPA2Model

from physicalai.policies.base import Model
from physicalai.policies.mixins.peft import PeftModelMixin
from physicalai.policies.utils.loss import in_episode_bound, reduce_losses

from .action_head import VLAJEPAActionHead
from .world_model import VideoPredictor

if TYPE_CHECKING:
    from pathlib import Path

    from .config import VLAJEPAConfig

logger = logging.getLogger(__name__)
_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
_PROJECTIONS = {
    "action_model.action_encoder.layer1.weight",
    "action_model.action_decoder.layer2.weight",
    "action_model.action_decoder.layer2.bias",
    "action_model.state_encoder.layer1.weight",
}


class VLAJEPAModel(PeftModelMixin, Model):
    """Network-only VLA-JEPA with joint JEPA/action training and full-chunk inference."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        qwen_config: dict[str, Any],
        dtype: str,
        chunk_size: int,
        action_dim: int,
        state_dim: int,
        action_hidden_size: int,
        action_num_layers: int,
        action_num_heads: int,
        action_attention_head_dim: int,
        action_dropout: float,
        num_target_vision_tokens: int,
        action_max_seq_len: int,
        num_inference_steps: int,
        num_timestep_buckets: int,
        noise_beta_alpha: float,
        noise_beta_beta: float,
        noise_s: float,
        repeated_diffusion_steps: int,
        enable_world_model: bool,
        world_model_config: dict[str, Any],
        num_video_frames: int,
        num_action_tokens_per_timestep: int,
        world_model_num_views: int,
        world_model_loss_weight: float,
        predictor_depth: int,
        predictor_num_heads: int,
        predictor_mlp_ratio: float,
    ) -> None:
        """Initialize Qwen and the checkpoint-compatible action head."""
        super().__init__()
        self.qwen = Qwen3VLModel(Qwen3VLConfig.from_dict(qwen_config)).to(dtype=_DTYPES[dtype])
        self.qwen.config._attn_implementation = "sdpa"  # noqa: SLF001
        # The released policy conditions on the final decoder block BEFORE its RMSNorm.
        self.qwen.language_model.norm.requires_grad_(requires_grad=False)
        self.action_model = VLAJEPAActionHead(
            context_dim=qwen_config["text_config"]["hidden_size"],
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            hidden_size=action_hidden_size,
            num_layers=action_num_layers,
            num_heads=action_num_heads,
            head_dim=action_attention_head_dim,
            dropout=action_dropout,
            num_future_tokens=num_target_vision_tokens,
            max_seq_len=action_max_seq_len,
            num_inference_steps=num_inference_steps,
            num_timestep_buckets=num_timestep_buckets,
            noise_beta_alpha=noise_beta_alpha,
            noise_beta_beta=noise_beta_beta,
            noise_s=noise_s,
        )
        self.repeated_diffusion_steps = repeated_diffusion_steps
        self._backbone_frozen = False
        self.num_video_frames = num_video_frames
        self.world_model_num_views = world_model_num_views
        self.world_model_loss_weight = world_model_loss_weight
        self.video_encoder: VJEPA2Model | None = None
        self.video_predictor: VideoPredictor | None = None
        if enable_world_model:
            encoder_config = VJEPA2Config.from_dict(world_model_config)
            self.video_encoder = VJEPA2Model(encoder_config).to(dtype=_DTYPES[dtype])
            self.video_encoder.requires_grad_(requires_grad=False).eval()
            self.video_predictor = VideoPredictor(
                encoder_dim=encoder_config.hidden_size * world_model_num_views,
                action_dim=qwen_config["text_config"]["hidden_size"],
                hidden_size=encoder_config.hidden_size,
                num_layers=predictor_depth,
                num_heads=predictor_num_heads,
                mlp_ratio=predictor_mlp_ratio,
                context_frames=num_video_frames // encoder_config.tubelet_size - 1,
                grid_size=world_model_config["crop_size"] // world_model_config["patch_size"],
                action_tokens_per_frame=num_action_tokens_per_timestep,
            )

    @classmethod
    def from_config(cls, config: VLAJEPAConfig) -> Self:
        """Return a network materialized from every resolved architecture argument."""
        return cls(
            qwen_config=config.qwen_config,
            dtype=config.dtype,
            chunk_size=config.chunk_size,
            action_dim=config.action_dim,
            state_dim=config.state_dim,
            action_hidden_size=config.action_hidden_size,
            action_num_layers=config.action_num_layers,
            action_num_heads=config.action_num_heads,
            action_attention_head_dim=config.action_attention_head_dim,
            action_dropout=config.action_dropout,
            num_target_vision_tokens=config.num_target_vision_tokens,
            action_max_seq_len=config.action_max_seq_len,
            num_inference_steps=config.num_inference_steps,
            num_timestep_buckets=config.num_timestep_buckets,
            noise_beta_alpha=config.noise_beta_alpha,
            noise_beta_beta=config.noise_beta_beta,
            noise_s=config.noise_s,
            repeated_diffusion_steps=config.repeated_diffusion_steps,
            enable_world_model=config.enable_world_model,
            world_model_config=config.world_model_config,
            num_video_frames=config.num_video_frames,
            num_action_tokens_per_timestep=config.num_action_tokens_per_timestep,
            world_model_num_views=config.world_model_num_views,
            world_model_loss_weight=config.world_model_loss_weight,
            predictor_depth=config.predictor_depth,
            predictor_num_heads=config.predictor_num_heads,
            predictor_mlp_ratio=config.predictor_mlp_ratio,
        )

    @classmethod
    def get_default_peft_targets(cls) -> str:
        """Return targets for language attention/MLPs, action projections/DiT, and the JEPA predictor."""
        return (
            r"qwen\.language_model\.layers\.\d+\."
            r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
            r"|action_model\.((action_encoder|action_decoder|state_encoder)\.layer\d+"
            r"|model\.(proj_out_[12]|timestep_encoder\.timestep_embedder\.linear_[12]"
            r"|transformer_blocks\.\d+\.(norm1\.linear|attn1\.(to_q|to_k|to_v|to_out\.0)"
            r"|ff\.net\.(0\.proj|2))))"
            r"|video_predictor\.(predictor_embed|action_encoder|predictor_proj"
            r"|predictor_blocks\.\d+\.(attn\.(qkv|proj)|mlp\.fc[12]))"
        )

    def load_weights(self, path: Path, *, reinitialize_projections: bool = False) -> None:  # noqa: PLR0912
        """Load converted safetensors, allowing only explicit robot I/O adaptation.

        Raises:
            ValueError: If checkpoint keys or shapes do not match the final architecture.
        """
        expected = self.state_dict()
        weights: dict[str, Tensor] = {}
        skipped: set[str] = set()
        source_has_state = False
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for source_key in checkpoint.keys():  # noqa: SIM118 - safe_open is not iterable.
                if self.video_encoder is None and source_key.startswith((
                    "model.video_encoder.",
                    "model.video_predictor.",
                    "video_encoder.",
                    "video_predictor.",
                )):
                    continue
                if source_key == "model.qwen.model.lm_head.weight":
                    continue
                key = source_key
                if key.startswith("model.qwen.model.model."):
                    key = "qwen." + key.removeprefix("model.qwen.model.model.")
                elif key.startswith(("model.action_model.", "model.video_encoder.", "model.video_predictor.")):
                    key = key.removeprefix("model.")
                source_has_state |= key.startswith("action_model.state_encoder.")
                if key not in expected:
                    if (
                        reinitialize_projections
                        and self.action_model.state_encoder is None
                        and key.startswith("action_model.state_encoder.")
                    ):
                        continue
                    msg = f"Unexpected VLA-JEPA checkpoint tensor: {source_key}"
                    raise ValueError(msg)
                tensor = checkpoint.get_tensor(source_key)
                if tensor.shape != expected[key].shape:
                    if reinitialize_projections and key in _PROJECTIONS:
                        skipped.add(key)
                        continue
                    msg = f"Checkpoint shape mismatch for {key}: {tuple(tensor.shape)} != {tuple(expected[key].shape)}"
                    raise ValueError(msg)
                if key in weights:
                    msg = f"Duplicate checkpoint tensor: {key}"
                    raise ValueError(msg)
                weights[key] = tensor
        missing = set(expected) - weights.keys() - skipped
        if reinitialize_projections and not source_has_state and self.action_model.state_encoder is not None:
            missing_state = {key for key in missing if key.startswith("action_model.state_encoder.")}
            skipped.update(missing_state)
            missing -= missing_state
        if missing:
            msg = f"VLA-JEPA checkpoint is missing tensors: {sorted(missing)[:8]}"
            raise ValueError(msg)
        self.load_state_dict(weights, strict=False)
        if skipped:
            logger.warning("Reinitialized VLA-JEPA robot projections for fine-tuning: %s", sorted(skipped))

    def freeze_backbone(self) -> None:
        """Freeze Qwen while keeping the complete action head trainable."""
        self.qwen.requires_grad_(requires_grad=False)
        self._backbone_frozen = True
        self.qwen.eval()

    def enable_gradient_checkpointing(self) -> None:
        """Checkpoint Qwen decoder and DiT layers without changing checkpoint keys."""
        self.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.action_model.model.gradient_checkpointing = True
        if self.video_predictor is not None:
            self.video_predictor.gradient_checkpointing = True

    def train(self, mode: bool = True) -> Self:  # noqa: FBT001, FBT002
        """Return this model, keeping a frozen backbone in evaluation mode."""
        super().train(mode)
        if self._backbone_frozen:
            self.qwen.eval()
        if self.video_encoder is not None:
            self.video_encoder.eval()
        return self

    def _context(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        captured: list[Tensor] = []

        def capture(_module: torch.nn.Module, inputs: tuple[Tensor, ...]) -> None:
            captured.append(inputs[0])

        handle = self.qwen.language_model.norm.register_forward_pre_hook(capture)
        try:
            inputs = {
                name: batch[name]
                for name in (
                    "input_ids",
                    "attention_mask",
                    "pixel_values",
                    "image_grid_thw",
                    "mm_token_type_ids",
                )
                if name in batch
            }
            self.qwen(**inputs, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        hidden = captured[0]
        positions = batch["embodied_positions"]
        tokens = hidden.gather(1, positions[:, :, None].expand(-1, -1, hidden.shape[-1]))
        dtype = next(self.action_model.parameters()).dtype
        context = {"tokens": tokens.to(dtype)}
        if "latent_positions" in batch:
            latent_positions = batch["latent_positions"]
            context["latent_tokens"] = hidden.gather(1, latent_positions[:, :, None].expand(-1, -1, hidden.shape[-1]))
        if "state" in batch:
            context["state"] = batch["state"].to(dtype)
        return context

    def forward(self, batch: dict[str, Tensor]) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        """Return flow loss during training, otherwise normalized actions."""
        return self.compute_loss(batch) if self.training else self.predict_action_chunk(batch)

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Return flow loss and metrics, excluding episode-end padding."""
        context = self._context(batch)
        world_loss = context["tokens"].new_zeros(())
        if self.video_predictor is not None:
            world_loss = self._world_model_loss(batch, context.pop("latent_tokens"))
        repeats = self.repeated_diffusion_steps
        actions = batch["action"].to(context["tokens"]).repeat(repeats, 1, 1)
        context = {name: tensor.repeat(repeats, 1, 1) for name, tensor in context.items()}
        bound = in_episode_bound(batch)
        if bound is not None:
            bound = bound.repeat(repeats, 1)
        with torch.autocast(device_type=actions.device.type, enabled=False):
            action_loss = reduce_losses(self.action_model.compute_loss(actions, context), bound)
        # Adding a constant zero world loss makes inductor drop the loss's grad_fn.
        loss = action_loss if self.video_predictor is None else action_loss + self.world_model_loss_weight * world_loss
        return loss, {
            "loss": loss.detach(),
            "action_flow_loss": action_loss.detach(),
            "world_model_loss": world_loss.detach(),
        }

    def _world_model_loss(self, batch: dict[str, Tensor], latent_tokens: Tensor) -> Tensor:
        if self.video_encoder is None or self.video_predictor is None:
            msg = "The JEPA training modules are not initialized."
            raise RuntimeError(msg)
        videos = batch["videos"]
        batch_size, views, frames, channels, height, width = videos.shape
        with torch.no_grad():
            encoded = self.video_encoder.get_vision_features(
                videos.reshape(batch_size * views, frames, channels, height, width).to(self.video_encoder.dtype),
            )
        temporal = frames // self.video_encoder.config.tubelet_size
        patches = encoded.shape[1] // temporal
        # Keep samples separate while concatenating the views at each spatial/temporal position.
        encoded = encoded.view(batch_size, views, temporal, patches, -1).permute(0, 2, 3, 1, 4)
        encoded = encoded.reshape(batch_size, temporal * patches, -1)
        dtype = self.video_predictor.predictor_embed.weight.dtype
        with torch.autocast(device_type=videos.device.type, enabled=False):
            predicted = self.video_predictor(encoded[:, :-patches].to(dtype), latent_tokens.to(dtype))
            error = (predicted.float() - encoded[:, patches:].float()).abs()
            error = error.view(batch_size, temporal - 1, patches, -1)
            valid = batch["video_valid"].to(error)[:, :, None, None]
            return (error * valid).sum() / (valid.sum() * patches * error.shape[-1]).clamp_min(1)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        """Return one complete normalized action chunk."""
        context = self._context(batch)
        if noise is not None:
            noise = noise.to(context["tokens"])
        with torch.autocast(device_type=context["tokens"].device.type, enabled=False):
            return self.action_model.sample(context, noise=noise).float()

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Return fully denoised action MSE and validation metrics."""
        reference = batch["action"]
        prediction = self.predict_action_chunk(batch, noise=torch.zeros_like(reference))
        mse = reduce_losses((prediction - reference.to(prediction)).square(), in_episode_bound(batch))
        return mse, {"loss": mse.detach(), "action_mse": mse.detach()}

    @property
    def action_delta_indices(self) -> list[int]:
        """Future action offsets consumed during training."""
        return list(range(self.action_model.chunk_size))

    @property
    def observation_delta_indices(self) -> list[int] | None:
        """Future video offsets, present only when JEPA supervision is enabled."""
        return list(range(self.num_video_frames)) if self.video_encoder is not None else None

    @property
    def reward_delta_indices(self) -> None:
        """Rewards are not model inputs."""
        return None
