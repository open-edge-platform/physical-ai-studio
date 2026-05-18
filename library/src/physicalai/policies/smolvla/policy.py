# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA Policy - Lightning wrapper for training and inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub import hf_hub_download
from physicalai.inference.manifest import ComponentSpec
from safetensors.torch import load_file

from physicalai.data.observation import ACTION, STATE
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import (
    ExportParameters,
    ONNXExportParameters,
    OpenVINOExportParameters,
    TorchExportParameters,
)
from physicalai.policies.base import Policy
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler
from physicalai.train.utils import reformat_dataset_to_match_policy

from .config import SmolVLAConfig
from .model import SmolVLAModel
from .pretrained_utils import extract_dataset_stats, fix_state_dict_keys

if TYPE_CHECKING:
    from physicalai.data import Observation

    from .preprocessor import SmolVLAPostprocessor, SmolVLAPreprocessor

logger = logging.getLogger(__name__)


class SmolVLA(ExportablePolicyMixin, Policy):
    """SmolVLA Policy - Hugging Face's flow matching VLA model.

    Lightning wrapper for training and inference with SmolVLA model.

    Uses dual-path initialization:
    - **Lazy path**: `SmolVLA()` + `trainer.fit()` - model built in setup()
    - **Eager path**: `SmolVLA.load_from_checkpoint()` - model built immediately

    Args:
        pretrained_name_or_path: HuggingFace repo ID or local path for pretrained weights and config.
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

    def __init__(  # noqa: PLR0913
        self,
        # Pretrained model id
        pretrained_name_or_path: str | Path | None = None,
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
        # Compilation
        compile_model: bool = False,
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

        weights_file = None
        if pretrained_name_or_path is not None:
            self.config, dataset_stats, weights_file = self._from_hf(
                pretrained_name_or_path,
                tokenizer_max_length=tokenizer_max_length,
                pad_language_to=pad_language_to,
                use_random_input_noise=use_random_input_noise,
                compile_model=compile_model,
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
        else:
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
                compile_model=compile_model,
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
        self.save_hyperparameters(
            ignore=["config", "pretrained_name_or_path", "compile_model"],
        )
        # Also save config dict for compatibility
        self.hparams["config"] = self.config.to_dict()

        # Model will be built in setup() or immediately if env_action_dim provided
        self.model: SmolVLAModel | None = None

        # Preprocessor/postprocessor set in setup() or _initialize_model()
        self._preprocessor: SmolVLAPreprocessor | None = None
        self._postprocessor: SmolVLAPostprocessor | None = None

        # Eager initialization if dataset_stats is provided
        if dataset_stats is not None:
            self._initialize_model(dataset_stats, weights_file)

        self._dataset_stats = dataset_stats

    def _initialize_model(
        self,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]],
        weights_file: Path | None = None,
    ) -> None:
        """Initialize model and preprocessors.

        Called by both lazy (setup) and eager (checkpoint) paths.

        Args:
            dataset_stats: Dataset normalization statistics.
            weights_file: Optional pretrained weights file.
        """
        self.model = SmolVLAModel(
            dataset_stats,
            chunk_size=self.config.chunk_size,
            max_state_dim=self.config.max_state_dim,
            max_action_dim=self.config.max_action_dim,
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
            compile_model=self.config.compile_model,
        )

        if weights_file is not None:
            original_sd = load_file(str(weights_file))

            fixed_sd = fix_state_dict_keys(original_sd)

            missing, unexpected = self.model.load_state_dict(fixed_sd, strict=False, assign=False)
            if missing:
                msg = f"Missing keys when loading pretrained weights: {len(missing)} keys"
                logger.warning(msg)
                for k in missing[:10]:
                    msg = f"  - {k}"
                    logger.warning(msg)
            if unexpected:
                msg = f"Unexpected keys when loading pretrained weights: {len(unexpected)} keys"
                logger.warning(msg)
                for k in unexpected[:10]:
                    msg = f"  - {k}"
                    logger.warning(msg)

            # Apply requires_grad
            self.model._model.set_requires_grad()  # noqa: SLF001
            self.model._model.vlm_with_expert.set_requires_grad()  # noqa: SLF001

        self._update_preprocessor_stats(dataset_stats)

        self._dataset_stats = dataset_stats

    def _from_hf(  # noqa: PLR6301, PLR0913
        self,
        pretrained_name_or_path: str | Path,
        *,
        tokenizer_max_length: int = 48,
        pad_language_to: str = "max_length",
        use_random_input_noise: bool = False,
        compile_model: bool = False,
        num_steps: int = 10,
        use_cache: bool = True,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        train_state_proj: bool = True,
        optimizer_lr: float = 1e-4,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 1e-10,
        optimizer_grad_clip_norm: float = 10,
        scheduler_warmup_steps: int = 1_000,
        scheduler_decay_steps: int = 30_000,
        scheduler_decay_lr: float = 2.5e-6,
    ) -> tuple[SmolVLAConfig, dict[str, dict[str, list[float] | str | tuple]] | None, Path | None]:
        """Template loader for SmolVLA pretrained config/weights from local path or HF Hub.

        This mirrors Pi05's structure and is intentionally incomplete. It resolves
        files, applies caller overrides to config, and returns placeholders for
        dataset stats / weight-loading integration.

        Returns:
            Tuple of (config, dataset_stats, weights_file).
        """
        path = Path(pretrained_name_or_path)
        is_local = path.is_dir()

        if is_local:
            config_file = path / "config.json"
            weights_file = path / "model.safetensors"
            preprocessor_file = path / "policy_preprocessor.json"
            preprocessor_dir = path
        else:
            config_file = Path(hf_hub_download(str(pretrained_name_or_path), "config.json"))  # nosec B615
            weights_file = Path(hf_hub_download(str(pretrained_name_or_path), "model.safetensors"))  # nosec B615
            try:
                preprocessor_file = Path(
                    hf_hub_download(str(pretrained_name_or_path), "policy_preprocessor.json"),  # nosec B615
                )
                preprocessor_dir = preprocessor_file.parent

                # Also download referenced state files
                with Path(preprocessor_file).open(encoding="utf-8") as f:
                    preproc_data = json.load(f)
                for step in preproc_data.get("steps", []):
                    sf = step.get("state_file")
                    if sf:
                        hf_hub_download(str(pretrained_name_or_path), sf)  # nosec B615
            except Exception:  # noqa: BLE001
                preprocessor_file = None
                preprocessor_dir = None

        with Path(config_file).open(encoding="utf-8") as f:
            hf_config = json.load(f)

        # Apply only safe overrides
        hf_config["tokenizer_max_length"] = tokenizer_max_length
        hf_config["pad_language_to"] = pad_language_to
        hf_config["use_random_input_noise"] = use_random_input_noise
        hf_config["compile_model"] = compile_model
        hf_config["num_steps"] = num_steps
        hf_config["use_cache"] = use_cache
        hf_config["freeze_vision_encoder"] = freeze_vision_encoder
        hf_config["train_expert_only"] = train_expert_only
        hf_config["train_state_proj"] = train_state_proj
        hf_config["optimizer_lr"] = optimizer_lr
        hf_config["optimizer_betas"] = optimizer_betas
        hf_config["optimizer_eps"] = optimizer_eps
        hf_config["optimizer_weight_decay"] = optimizer_weight_decay
        hf_config["optimizer_grad_clip_norm"] = optimizer_grad_clip_norm
        hf_config["scheduler_warmup_steps"] = scheduler_warmup_steps
        hf_config["scheduler_decay_steps"] = scheduler_decay_steps
        hf_config["scheduler_decay_lr"] = scheduler_decay_lr

        config = SmolVLAConfig.from_dict(hf_config)

        dataset_stats = extract_dataset_stats(hf_config, preprocessor_file, preprocessor_dir)

        return config, dataset_stats, weights_file

    def _update_preprocessor_stats(
        self,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]],
    ) -> None:
        """Rebuild pre- and postprocessors from dataset_stats.

        Used on the fine-tuning path to replace pretrained normalization with
        training-data statistics, and by _initialize_model on the lazy path.

        Args:
            dataset_stats: Dataset normalization statistics.
        """
        from .preprocessor import make_smolvla_preprocessors  # noqa: PLC0415

        self._preprocessor, self._postprocessor = make_smolvla_preprocessors(
            max_state_dim=self.config.max_state_dim,
            max_action_dim=self.config.max_action_dim,
            stats=dataset_stats,
            image_resolution=self.config.resize_imgs_with_padding,
            max_token_len=self.config.tokenizer_max_length,
            token_pad_type=self.config.pad_language_to,
            tokenizer_name=self.config.vlm_model_name,
        )
        self._dataset_stats = dataset_stats
        self.hparams["dataset_stats"] = dataset_stats
        if self.model is not None:
            self.model.set_dataset_stats(dataset_stats)

    def setup(self, stage: str) -> None:
        """Set up model from datamodule (lazy initialization path).

        Called by Lightning before fit/validate/test/predict.

        Args:
            stage: Lightning stage (unused, required by Lightning API).

        Raises:
            TypeError: If train dataset is not a physicalai Dataset.
        """
        del stage  # Unused argument

        from physicalai.data.dataset import Dataset  # noqa: PLC0415

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        stats_dict = train_dataset.stats

        if self.model is not None:
            self._update_preprocessor_stats(stats_dict)
            reformat_dataset_to_match_policy(self, datamodule)
            return

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
            if self.model is None or self._preprocessor is None:
                msg = "Model is not initialized"
                raise ValueError(msg)

            processed_batch = self._preprocessor(batch.to_dict())
            return self.model(processed_batch)
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
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)

        processed_batch = self._preprocessor(batch.to(self.device).to_dict())
        chunk = self.model.predict_action_chunk(processed_batch)
        return self._postprocessor({ACTION: chunk})[ACTION]

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

    def configure_optimizers(self) -> dict[str, Any]:
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

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Additional export arguments for model conversion.

        Returns:
            dict[str, ExportParameters]: A dictionary mapping format names to their export parameters.

        Raises:
            ValueError: If dataset_stats is not available for export argument construction.
        """
        extra_args: dict[str, ExportParameters] = {}
        if self._dataset_stats is None:
            msg = (
                "Dataset stats are required for export. Initialize the policy with dataset_stats"
                " or train for at least one epoch to populate them."
            )
            raise ValueError(msg)

        base_preproc_specs = [
            ComponentSpec(type="smolvla_resize", image_resolution=self.config.resize_imgs_with_padding),
            ComponentSpec(type="new_line"),
            ComponentSpec(
                type="normalize",
                stats={STATE: self._dataset_stats[f"observation.{STATE}"]},
                mode="mean_std",
            ),
        ]
        postproc_specs = [
            ComponentSpec(
                type="denormalize",
                stats={ACTION: self._dataset_stats[ACTION]},
                mode="mean_std",
            ),
        ]
        if self.config.chunk_size != self.config.n_action_steps:
            postproc_specs.append(
                ComponentSpec(
                    type="action_chunk_trimmer",
                    n_action_steps=self.config.n_action_steps,
                ),
            )

        extra_args["onnx"] = ONNXExportParameters(
            exporter_kwargs={
                "output_names": [ACTION],
            },
            preprocessors_specs=[
                *base_preproc_specs,
                ComponentSpec(
                    type="hf_tokenizer",
                    tokenizer_name=self.config.vlm_model_name,
                    revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467",
                    max_token_len=self.config.tokenizer_max_length,
                ),
            ],
            postprocessors_specs=postproc_specs,
            export_tokenizer=False,
        )
        extra_args["openvino"] = OpenVINOExportParameters(
            outputs=[ACTION],
            compress_to_fp16=False,
            export_tokenizer=True,
            exporter_kwargs={},
            preprocessors_specs=[
                *base_preproc_specs,
                ComponentSpec(
                    type="ov_tokenizer",
                    artifact="tokenizer.xml",
                ),
            ],
            postprocessors_specs=postproc_specs,
        )
        extra_args["torch"] = TorchExportParameters()

        return extra_args
