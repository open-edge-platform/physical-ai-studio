# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA Policy - Lightning wrapper for training and inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from physicalai.data.observation import ACTION
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.policies.base import Policy
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler
from physicalai.train.utils import reformat_dataset_to_match_policy

from .config import SmolVLAConfig
from .model import SmolVLAModel

if TYPE_CHECKING:
    from physicalai.data import Observation

    from .preprocessor import SmolVLAPostprocessor, SmolVLAPreprocessor


class SmolVLA(ExportablePolicyMixin, Policy):
    """SmolVLA Policy - Hugging Face's flow matching VLA model.

    Lightning wrapper for training and inference with SmolVLA model.

    Uses dual-path initialization:
    - **Lazy path**: `SmolVLA()` + `trainer.fit()` - model built in setup()
    - **Eager path**: `SmolVLA.load_from_checkpoint()` - model built immediately

    Args:
        n_obs_steps: Number of observation steps to use. Default: 1.
        chunk_size: Size of action chunks for prediction. Default: 50.
        n_action_steps: Number of action steps to execute. Default: 50.
        max_state_dim: Maximum state dimension (shorter vectors are padded). Default: 32.
        max_action_dim: Maximum action dimension (shorter vectors are padded). Default: 32.
        resize_imgs_with_padding: Target image resolution (height, width). Default: (512, 512).
        tokenizer_max_length: Maximum length for tokenizer. Default: 48.
        vlm_model_name: VLM backbone model name. Default: "HuggingFaceTB/SmolVLM2-500M-Video-Instruct".
        load_vlm_weights: Whether to load pretrained VLM weights. Default: False.
        add_image_special_tokens: Whether to use special image tokens around image features. Default: False.
        attention_mode: Attention mode for the model. Default: "cross_attn".
        prefix_length: Prefix length for attention. Default: -1.
        pad_language_to: Padding strategy for language tokens. Default: "longest".
        num_expert_layers: Number of expert layers (-1 matches VLM layers). Default: -1.
        num_vlm_layers: Number of layers used in the VLM. Default: 16.
        self_attn_every_n_layers: Interleave self-attention layers frequency. Default: 2.
        expert_width_multiplier: Action expert hidden size ratio to VLM. Default: 0.75.
        min_period: Minimum period for sine-cosine positional encoding. Default: 4e-3.
        max_period: Maximum period for sine-cosine positional encoding. Default: 4.0.
        use_random_input_noise: Whether to use random noise as the initial input for the denoising
            process during inference. If False, zeros are used instead. Default: True.
        num_steps: Number of decoding steps. Default: 10.
        use_cache: Whether to use attention cache. Default: True.
        freeze_vision_encoder: Whether to freeze vision encoder during training. Default: True.
        train_expert_only: Whether to train only the expert layers. Default: True.
        train_state_proj: Whether to train state projection layers. Default: True.
        optimizer_lr: Learning rate for optimizer. Default: 1e-4.
        optimizer_betas: Beta parameters for AdamW optimizer. Default: (0.9, 0.95).
        optimizer_eps: Epsilon for optimizer numerical stability. Default: 1e-8.
        optimizer_weight_decay: Weight decay for optimizer. Default: 1e-10.
        optimizer_grad_clip_norm: Gradient clipping norm value. Default: 10.
        scheduler_warmup_steps: Number of warmup steps for scheduler. Default: 1_000.
        scheduler_decay_steps: Number of steps between learning rate decays. Default: 30_000.
        scheduler_decay_lr: Learning rate decay factor. Default: 2.5e-6.
        dataset_stats: Dataset normalization statistics for eager initialization. Default: None.

    Example:
        Training:

        >>> policy = SmolVLA(learning_rate=2.5e-5)
        >>> trainer = physicalai.Trainer(max_epochs=100)
        >>> trainer.fit(policy, datamodule)

        Inference:

        >>> policy = SmolVLA.load_from_checkpoint("checkpoint.ckpt")
        >>> action = policy.select_action(obs)
    """

    model: Any
    _preprocessor: Any

    def __init__(  # noqa: PLR0913
        self,
        # Input / output structure.
        n_obs_steps: int = 1,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        # Shorter state and action vectors will be padded
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        # Image preprocessing
        resize_imgs_with_padding: tuple[int, int] = (512, 512),
        *,
        # Architecture
        tokenizer_max_length: int = 48,
        vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",  # Select the VLM backbone.
        load_vlm_weights: bool = False,  # Set to True in case of training the expert from scratch.
        # True when init from pretrained SmolVLA weights
        add_image_special_tokens: bool = False,  # Whether to use special image tokens around image features.
        attention_mode: str = "cross_attn",
        prefix_length: int = -1,
        pad_language_to: str = "max_length",  # "longest"
        num_expert_layers: int = -1,  # Less or equal to 0 is the default where the action expert has the same
        # number of layers of VLM. Otherwise, the expert have less layers.
        num_vlm_layers: int = 16,  # Number of layers used in the VLM (first num_vlm_layers layers)
        self_attn_every_n_layers: int = 2,  # Interleave SA layers each self_attn_every_n_layers
        expert_width_multiplier: float = 0.75,  # The action expert hidden size (wrt to the VLM)
        min_period: float = 4e-3,  # sensitivity range for the timestep used in sine-cosine positional encoding
        max_period: float = 4.0,
        use_random_input_noise: bool = False,
        snapflow_enabled: bool = False,
        snapflow_alpha: float = 0.5,
        snapflow_lambda: float = 1.0,
        snapflow_num_inference_steps: int = 1,
        # Decoding
        num_steps: int = 10,
        # Attention utils
        use_cache: bool = True,
        # Finetuning settings
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        train_state_proj: bool = True,
        # Training presets
        optimizer_lr: float = 1e-4,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 1e-10,
        optimizer_grad_clip_norm: float = 10,
        scheduler_warmup_steps: int = 1_000,
        scheduler_decay_steps: int = 30_000,
        scheduler_decay_lr: float = 2.5e-6,
        # Eager initialization (for checkpoint loading)
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
    ) -> None:
        """Initialize SmolVLA policy.

        Creates SmolVLAConfig from explicit args and saves it as hyperparameters.
        """
        super().__init__(n_action_steps=n_action_steps)

        # Create config from explicit args (policy-level config)
        self.config = SmolVLAConfig(
            n_obs_steps=n_obs_steps,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            resize_imgs_with_padding=resize_imgs_with_padding,
            tokenizer_max_length=tokenizer_max_length,
            vlm_model_name=vlm_model_name,
            load_vlm_weights=load_vlm_weights,
            add_image_special_tokens=add_image_special_tokens,
            attention_mode=attention_mode,
            prefix_length=prefix_length,
            pad_language_to=pad_language_to,
            num_expert_layers=num_expert_layers,
            num_vlm_layers=num_vlm_layers,
            self_attn_every_n_layers=self_attn_every_n_layers,
            expert_width_multiplier=expert_width_multiplier,
            min_period=min_period,
            max_period=max_period,
            use_random_input_noise=use_random_input_noise,
            snapflow_enabled=snapflow_enabled,
            snapflow_alpha=snapflow_alpha,
            snapflow_lambda=snapflow_lambda,
            snapflow_num_inference_steps=snapflow_num_inference_steps,
            num_steps=num_steps,
            use_cache=use_cache,
            freeze_vision_encoder=freeze_vision_encoder,
            train_expert_only=train_expert_only,
            train_state_proj=train_state_proj,
            optimizer_lr=optimizer_lr,
            optimizer_betas=optimizer_betas,
            optimizer_eps=optimizer_eps,
            optimizer_weight_decay=optimizer_weight_decay,
            optimizer_grad_clip_norm=optimizer_grad_clip_norm,
            scheduler_warmup_steps=scheduler_warmup_steps,
            scheduler_decay_steps=scheduler_decay_steps,
            scheduler_decay_lr=scheduler_decay_lr,
        )

        # Save config as hyperparameters for checkpoint restoration
        self.save_hyperparameters(ignore=["config"])  # Save individual args, not config object
        # Also save config dict for compatibility
        self.hparams["config"] = self.config.to_dict()

        # Model will be built in setup() or immediately if env_action_dim provided
        self._smolvla_model: SmolVLAModel | None = None

        # Preprocessor/postprocessor set in setup() or _initialize_model()
        self._smolvla_preprocessor: SmolVLAPreprocessor | None = None
        self._smolvla_postprocessor: SmolVLAPostprocessor | None = None

        # Eager initialization if dataset_stats is provided
        if dataset_stats is not None:
            self._initialize_model(dataset_stats)

        self._dataset_stats = dataset_stats

    def _initialize_model(
        self,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]],
    ) -> None:
        """Initialize model and preprocessors.

        Called by both lazy (setup) and eager (checkpoint) paths.

        Args:
            dataset_stats: Dataset normalization statistics.
        """
        from .preprocessor import make_smolvla_preprocessors  # noqa: PLC0415

        self._smolvla_model = SmolVLAModel(
            dataset_stats,
            chunk_size=self.config.chunk_size,
            max_state_dim=self.config.max_state_dim,
            max_action_dim=self.config.max_action_dim,
            resize_imgs_with_padding=self.config.resize_imgs_with_padding,
            adapt_to_pi_aloha=self.config.adapt_to_pi_aloha,
            num_steps=self.config.num_steps,
            use_cache=self.config.use_cache,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            train_state_proj=self.config.train_state_proj,
            vlm_model_name=self.config.vlm_model_name,
            load_vlm_weights=self.config.load_vlm_weights,
            add_image_special_tokens=self.config.add_image_special_tokens,
            attention_mode=self.config.attention_mode,
            prefix_length=self.config.prefix_length,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            use_random_input_noise=self.config.use_random_input_noise,
            tokenizer_max_length=self.config.tokenizer_max_length,
            snapflow_enabled=self.config.snapflow_enabled,
            snapflow_alpha=self.config.snapflow_alpha,
            snapflow_lambda=self.config.snapflow_lambda,
            snapflow_num_inference_steps=self.config.snapflow_num_inference_steps,
        )

        self._smolvla_preprocessor, self._smolvla_postprocessor = make_smolvla_preprocessors(
            max_state_dim=self.config.max_state_dim,
            max_action_dim=self.config.max_action_dim,
            stats=dataset_stats,
            image_resolution=self.config.resize_imgs_with_padding,
            max_token_len=self.config.tokenizer_max_length,
            token_pad_type=self.config.pad_language_to,
            tokenizer_name=self.config.vlm_model_name,
        )
        self.model = self._smolvla_model
        self._preprocessor = self._smolvla_preprocessor

    def setup(self, stage: str) -> None:
        """Set up model from datamodule (lazy initialization path).

        Called by Lightning before fit/validate/test/predict.

        Args:
            stage: Lightning stage (unused, required by Lightning API).

        Raises:
            TypeError: If train dataset is not a physicalai Dataset.
        """
        del stage  # Unused argument

        if self._smolvla_model is not None:
            return

        from physicalai.data.dataset import Dataset  # noqa: PLC0415

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        stats_dict = train_dataset.stats

        # Save to hparams for checkpoint
        self.hparams["dataset_stats"] = stats_dict

        self._initialize_model(stats_dict)

        reformat_dataset_to_match_policy(self, datamodule)

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Forward pass through the model.

        Processes the input batch and either trains the model or predicts actions
        depending on the current mode.

        Args:
            batch: An Observation object containing the input data for the model.

        Returns:
            If training: Returns the model output, either a tensor or a tuple
                containing a tensor and a dictionary of loss metrics.
            If not training: Returns the predicted action chunk as a tensor.

        Raises:
            ValueError: If the model is not initialized during training mode.
        """
        if self.training:
            if self._smolvla_model is None or self._smolvla_preprocessor is None:
                msg = "Model is not initialized"
                raise ValueError(msg)

            preprocessor = self._smolvla_preprocessor
            model = self._smolvla_model
            processed_batch = preprocessor(batch.to_dict())
            return model(processed_batch)
        return self.predict_action_chunk(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict a chunk of actions from the given observation batch.

        Args:
            batch: An Observation object containing the input data for action prediction.

        Returns:
            torch.Tensor: The predicted action chunk after post-processing.

        Raises:
            ValueError: If the model has not been initialized.
        """
        if self._smolvla_model is None or self._smolvla_preprocessor is None or self._smolvla_postprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)

        preprocessor = self._smolvla_preprocessor
        postprocessor = self._smolvla_postprocessor
        model = self._smolvla_model
        processed_batch = preprocessor(batch.to(self.device).to_dict())
        chunk = model.predict_action_chunk(processed_batch)
        return postprocessor({ACTION: chunk})[ACTION]

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Args:
            batch: Input batch.
            batch_idx: Batch index (unused, required by Lightning API).

        Returns:
            Loss tensor for backpropagation.
        """
        del batch_idx
        loss, loss_dict = self(batch)

        # Log metrics
        self.log("train/loss", loss_dict["loss"], prog_bar=True)

        return loss

    def compute_val_loss(self, batch: Observation) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute validation loss on a batch.

        Delegates to the model's ``compute_val_loss`` without toggling
        train mode.

        Args:
            batch: Observation batch (must contain ground-truth actions).

        Returns:
            Tuple of (loss tensor, loss dict).

        Raises:
            ValueError: If the model is not initialized.
        """
        if self.model is None or self._preprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)
        processed_batch = self._preprocessor(batch.to_dict())
        return self.model.compute_val_loss(processed_batch)

    def configure_optimizers(self) -> Any:  # noqa: ANN401
        """Configure optimizer and scheduler.

        Returns:
            Optimizer configuration dict.
        """
        # Get trainable parameters
        params = [p for p in self.parameters() if p.requires_grad]

        # Create optimizer (use config values)
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.optimizer_lr,
            weight_decay=self.config.optimizer_weight_decay,
            betas=self.config.optimizer_betas,
        )

        num_decay_steps = self.config.scheduler_decay_steps
        scheduler = cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=self.config.optimizer_lr,
            decay_lr=self.config.scheduler_decay_lr,
            num_warmup_steps=self.config.scheduler_warmup_steps,
            num_decay_steps=num_decay_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def to_onnx(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Export model to ONNX format.

        Raises:
            ValueError: If output_path is not provided.
        """
        output_path = kwargs.pop("output_path", None)
        input_sample = kwargs.pop("input_sample", None)
        if output_path is None and args:
            output_path = args[0]
        if input_sample is None and len(args) > 1:
            input_sample = args[1]
        if output_path is None:
            msg = "output_path must be provided"
            raise ValueError(msg)
        ExportablePolicyMixin.to_onnx(self, output_path=output_path, input_sample=input_sample, **kwargs)

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Configure gradient clipping from policy config.

        This overrides Lightning's default gradient clipping to use
        the policy's grad_clip_norm config value.

        Args:
            optimizer: The optimizer being used.
            gradient_clip_val: Ignored (uses config value instead).
            gradient_clip_algorithm: Ignored (always uses 'norm').
        """
        # Use Trainer's value if set, otherwise fall back to policy config
        clip_val = gradient_clip_val if gradient_clip_val is not None else self.config.optimizer_grad_clip_norm

        if clip_val and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Get a list of export backends supported by policy.

        This method returns a list of supported export backends as strings.

        Returns:
            list[str | ExportBackend]: A list of supported export backends.
        """
        return [ExportBackend.TORCH, ExportBackend.OPENVINO]
