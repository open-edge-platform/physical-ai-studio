# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning module for ACT policy."""

from typing import Any, cast

import torch

from physicalai.data import Dataset, Feature, FeatureType, NormalizationParameters, Observation
from physicalai.export.mixin_export import Export, ExportBackend
from physicalai.gyms import Gym
from physicalai.policies.act.config import ACTConfig
from physicalai.policies.act.model import ACT as ACTModel  # noqa: N811
from physicalai.policies.base import Policy
from physicalai.train.utils import reformat_dataset_to_match_policy


class ACT(Export, Policy):
    """Action Chunking with Transformers (ACT) policy implementation.

    This class implements the ACT policy for imitation learning, which uses a transformer-based
    architecture to predict sequences of actions given observations.
    Policy contains model and other related modules and methods that are required
    to start training in a Lightning Trainer.

    The model is lazily initialized during ``setup()`` from the datamodule, or eagerly
    when ``dataset_stats`` is provided (e.g. when restoring from a checkpoint).

    Args:
        n_obs_steps: Number of observation steps to pass to the policy.
        chunk_size: Size of the action prediction chunk in environment steps.
        n_action_steps: Number of action steps to execute per policy invocation.
            Should be no greater than ``chunk_size``.
        vision_backbone: Name of the torchvision ResNet backbone for encoding images.
        pretrained_backbone_weights: Pretrained weights for the vision backbone.
            ``None`` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final
            2x2 stride with a dilated convolution.
        pre_norm: Whether to use pre-norm in transformer blocks.
        dim_model: Main hidden dimension of the transformer blocks.
        n_heads: Number of attention heads in transformer blocks.
        dim_feedforward: Feedforward expansion dimension in transformer blocks.
        feedforward_activation: Activation function in transformer feedforward layers.
        n_encoder_layers: Number of transformer encoder layers.
        n_decoder_layers: Number of transformer decoder layers.
        use_vae: Whether to use a variational objective during training.
        latent_dim: Latent dimension for the VAE.
        n_vae_encoder_layers: Number of transformer layers in the VAE encoder.
        temporal_ensemble_coeff: Coefficient for temporal ensembling. ``None`` disables it.
            When enabled, ``n_action_steps`` must be 1.
        dropout: Dropout rate in transformer layers.
        kl_weight: Weight for the KL-divergence loss term when ``use_vae`` is True.
        optimizer_lr: Learning rate for the optimizer.
        optimizer_weight_decay: Weight decay for the optimizer.
        optimizer_grad_clip_norm: Maximum gradient norm for gradient clipping.
        dataset_stats: Dataset normalization statistics for eager model initialization
            (used when restoring from a checkpoint).

    Examples:
        Create a policy with default parameters (model built lazily during training):

            >>> policy = ACT()
            >>> trainer.fit(policy, datamodule=dm)

        Create a policy with custom architecture parameters:

            >>> policy = ACT(
            ...     chunk_size=50,
            ...     dim_model=256,
            ...     n_heads=4,
            ...     n_encoder_layers=6,
            ... )

        Export a trained policy:

            >>> policy.export("./exports", backend="openvino")
            >>> policy.export("./exports", backend="onnx")
    """

    def __init__(  # noqa: PLR0913
        self,
        n_obs_steps: int = 1,
        chunk_size: int = 100,
        n_action_steps: int = 100,
        *,
        vision_backbone: str = "resnet18",
        pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1",
        replace_final_stride_with_dilation: bool = False,
        pre_norm: bool = False,
        dim_model: int = 512,
        n_heads: int = 8,
        dim_feedforward: int = 3200,
        feedforward_activation: str = "relu",
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 1,
        use_vae: bool = True,
        latent_dim: int = 32,
        n_vae_encoder_layers: int = 4,
        temporal_ensemble_coeff: float | None = None,
        dropout: float = 0.1,
        kl_weight: float = 10.0,
        optimizer_lr: float = 1e-5,
        optimizer_weight_decay: float = 1e-4,
        optimizer_grad_clip_norm: float = 10.0,
        # Eager initialization (for checkpoint loading)
        dataset_stats: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ACT policy.

        Creates ACTConfig from explicit args and saves it as hyperparameters.
        """
        super().__init__(n_action_steps=n_action_steps)

        # Create config from explicit args (policy-level config)
        self.config = ACTConfig(
            input_features={},
            output_features={},
            n_obs_steps=n_obs_steps,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            vision_backbone=vision_backbone,
            pretrained_backbone_weights=pretrained_backbone_weights,
            replace_final_stride_with_dilation=replace_final_stride_with_dilation,
            pre_norm=pre_norm,
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            feedforward_activation=feedforward_activation,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            use_vae=use_vae,
            latent_dim=latent_dim,
            n_vae_encoder_layers=n_vae_encoder_layers,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
            dropout=dropout,
            kl_weight=kl_weight,
            optimizer_lr=optimizer_lr,
            optimizer_weight_decay=optimizer_weight_decay,
            optimizer_grad_clip_norm=optimizer_grad_clip_norm,
        )

        # Save config as hyperparameters for checkpoint restoration
        self.save_hyperparameters(ignore=["config"])  # Save individual args, not config object
        # Also save config dict for compatibility
        self.hparams["config"] = self.config.to_dict()

        # Model will be built in setup() or immediately if env_action_dim provided
        self.model: ACTModel | None = None

        # Preprocessor/postprocessor set in setup() or _initialize_model()
        self._preprocessor = None
        self._postprocessor = None

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
            env_action_dim: Environment action dimension.
            dataset_stats: Dataset normalization statistics.
        """
        features: dict[str, Feature] = {}
        for stat in dataset_stats.values():
            features[str(stat["name"])] = Feature(
                name=str(stat["name"]),
                ftype=cast("FeatureType", stat["type"]),
                shape=cast("tuple[int, ...]", stat["shape"]),
                normalization_data=NormalizationParameters(
                    mean=cast("list[float]", stat["mean"]),
                    std=cast("list[float]", stat["std"]),
                ),
            )

        self.model = ACTModel(
            input_features={
                name: feature
                for name, feature in features.items()
                if feature.ftype in {FeatureType.STATE, FeatureType.VISUAL}
            },
            output_features={
                name: feature for name, feature in features.items() if feature.ftype == FeatureType.ACTION
            },
            n_obs_steps=self.config.n_obs_steps,
            chunk_size=self.config.chunk_size,
            n_action_steps=self.config.n_action_steps,
            vision_backbone=self.config.vision_backbone,
            pretrained_backbone_weights=self.config.pretrained_backbone_weights,
            replace_final_stride_with_dilation=self.config.replace_final_stride_with_dilation,
            pre_norm=self.config.pre_norm,
            dim_model=self.config.dim_model,
            n_heads=self.config.n_heads,
            dim_feedforward=self.config.dim_feedforward,
            feedforward_activation=self.config.feedforward_activation,
            n_encoder_layers=self.config.n_encoder_layers,
            n_decoder_layers=self.config.n_decoder_layers,
            use_vae=self.config.use_vae,
            latent_dim=self.config.latent_dim,
            n_vae_encoder_layers=self.config.n_vae_encoder_layers,
            temporal_ensemble_coeff=self.config.temporal_ensemble_coeff,
            dropout=self.config.dropout,
            kl_weight=self.config.kl_weight,
        )

    def setup(self, stage: str) -> None:
        """Set up the policy from datamodule if not already initialized.

        This method is called by Lightning before fit/validate/test/predict.
        It extracts features from the datamodule's training dataset and
        initializes the policy if it wasn't already created in __init__.

        Args:
            stage: The stage of training ('fit', 'validate', 'test', or 'predict')

        Raises:
            TypeError: If the train_dataset is not a physicalai.data.Dataset.
        """
        del stage  # Unused argument

        datamodule = self.trainer.datamodule  # type: ignore[union-attr]
        train_dataset = datamodule.train_dataset

        # Get the underlying LeRobot dataset - handle both data formats
        if not isinstance(train_dataset, Dataset):
            msg = f"Expected train_dataset to be physicalai.data.Dataset, got {type(train_dataset)}."
            raise TypeError(msg)

        if self.model is not None:
            reformat_dataset_to_match_policy(self, datamodule)
            return

        self.model = ACTModel(
            input_features=train_dataset.observation_features,
            output_features=train_dataset.action_features,
            n_obs_steps=self.config.n_obs_steps,
            chunk_size=self.config.chunk_size,
            n_action_steps=self.config.n_action_steps,
            vision_backbone=self.config.vision_backbone,
            pretrained_backbone_weights=self.config.pretrained_backbone_weights,
            replace_final_stride_with_dilation=self.config.replace_final_stride_with_dilation,
            pre_norm=self.config.pre_norm,
            dim_model=self.config.dim_model,
            n_heads=self.config.n_heads,
            dim_feedforward=self.config.dim_feedforward,
            feedforward_activation=self.config.feedforward_activation,
            n_encoder_layers=self.config.n_encoder_layers,
            n_decoder_layers=self.config.n_decoder_layers,
            use_vae=self.config.use_vae,
            latent_dim=self.config.latent_dim,
            n_vae_encoder_layers=self.config.n_vae_encoder_layers,
            temporal_ensemble_coeff=self.config.temporal_ensemble_coeff,
            dropout=self.config.dropout,
            kl_weight=self.config.kl_weight,
        )

        reformat_dataset_to_match_policy(self, datamodule)

    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict a chunk of actions from observation.

        Implements the abstract method from Policy base class.
        Returns the full action chunk predicted by the model.

        Args:
            batch: Input batch of observations.

        Returns:
            Action chunk tensor from the model.

        Raises:
            RuntimeError: If the ACT model is not initialized.
        """
        # Move batch to device (observations from gym are on CPU)
        inference_batch = batch.to(self.device).to_dict()
        if self.model is None:
            msg = "ACT model is not initialized."
            raise RuntimeError(msg)
        return self.model.predict_action_chunk(inference_batch)

    # select_action() is inherited from Policy base class - uses queue with predict_action_chunk()

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Perform forward pass of the ACT policy.

        The return value depends on the model's training mode:
        - In training mode: Returns (loss, loss_dict) from the model's forward method
        - In evaluation mode: Returns action chunk predictions via predict_action_chunk

        Args:
            batch (Observation): Input batch of observations

        Returns:
            torch.Tensor | tuple[torch.Tensor, dict[str, float]]: In training mode, returns
                tuple of (loss, loss_dict). In eval mode, returns action chunk tensor.

        Raises:
            RuntimeError: If the ACT model is not initialized.
        """
        if self.training:
            # During training, return loss information for backpropagation
            if self.model is None:
                msg = "ACT model is not initialized."
                raise RuntimeError(msg)
            return self.model(batch.to_dict())

        # During evaluation, return action chunk predictions
        return self.predict_action_chunk(batch)

    def training_step(self, batch: Observation, batch_idx: int) -> dict[str, torch.Tensor]:
        """Training step for the policy.

        Args:
            batch (Observation): The training batch.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss.

        Raises:
            RuntimeError: If the ACT model is not initialized.
        """
        del batch_idx
        if self.model is None:
            msg = "ACT model is not initialized."
            raise RuntimeError(msg)
        loss, loss_dict = self.model(batch.to_dict())  # noqa: RUF059
        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return {"loss": loss}

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer.

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
        )

        return {
            "optimizer": optimizer,
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

    def evaluation_step(self, batch: Observation, stage: str) -> None:  # noqa: PLR6301
        """Evaluation step (no-op by default).

        Args:
            batch (Observation): Input batch.
            stage (str): Evaluation stage, e.g., "val" or "test".
        """
        del batch, stage

    def validation_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Validation step.

        Runs gym-based validation via rollout evaluation. The DataModule's val_dataloader
        returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Index of the batch.

        Returns:
            Metrics dict from gym rollout.
        """
        return self.evaluate_gym(batch, batch_idx, stage="val")

    def test_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Test step.

        Runs gym-based testing via rollout evaluation. The DataModule's test_dataloader
        returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Index of the batch.

        Returns:
            Metrics dict from gym rollout.
        """
        return self.evaluate_gym(batch, batch_idx, stage="test")

    def reset(self) -> None:
        """Reset the policy state for a new episode.

        Clears internal state like action queues or observation history.
        For ACT, this also delegates to the model's reset method if available.
        """
        super().reset()  # Clear action queue
        if self.model is None:
            return
        if hasattr(self.model, "reset") and callable(self.model.reset):
            self.model.reset()

    @property
    def supported_export_backends(self) -> list[str | ExportBackend]:
        """Get a list of export backends supported by policy.

        This method returns a list of supported export backends as strings.

        Returns:
            list[str | ExportBackend]: A list of supported export backends.
        """
        return [ExportBackend.TORCH, ExportBackend.OPENVINO, ExportBackend.ONNX, ExportBackend.TORCH_EXPORT_IR]
