# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA model implementation."""

from __future__ import annotations

import copy
import logging
import math
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from getiaction.data.observation import ACTION, EXTRA, IMAGES, STATE, Observation

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import SmolVLAConfig


def _lazy_import_transformers() -> tuple:
    """Lazy import transformers classes to reduce initial load time.

    Returns:
        Tuple containing (AutoConfig, AutoModel, AutoModelForImageTextToText,
            AutoProcessor, SmolVLMForConditionalGeneration).

    Raises:
        ImportError: If diffusers is not installed.
    """
    try:
        from transformers import (  # noqa: PLC0415
            AutoConfig,
            AutoModel,
            AutoModelForImageTextToText,
            AutoProcessor,
            SmolVLMForConditionalGeneration,
        )
    except ImportError as e:
        msg = "SmolVLA requires transformers library.\n\nInstall with:\n    pip install transformers"
        raise ImportError(msg) from e
    else:
        return AutoConfig, AutoModel, AutoModelForImageTextToText, AutoProcessor, SmolVLMForConditionalGeneration


logger = logging.getLogger(__name__)


class SmolVLAModel(nn.Module):
    """SmolVLA flow matching vision-language-action model."""

    def __init__(
        self,
        config: SmolVLAConfig,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple[int, ...]]],
    ) -> None:
        """Initialize the SmolVLA model.

        Args:
            config: Configuration object containing model hyperparameters and settings.
            dataset_stats: Dictionary containing dataset statistics with keys mapping to
                dictionaries that hold statistics values (lists of floats), string metadata,
                or tuple information used for normalization and preprocessing.
        """
        super().__init__()
        self._config = config
        self._model = VLAFlowMatching(config)
        self._dataset_stats = dataset_stats

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Forward pass for the SmolVLA model.

        During training, processes the input batch to compute the loss for action prediction.
        Handles optional adaptation for PI-ALOHA format, prepares images, state, and actions,
        and applies masking for padded actions.
        During inference, delegates to predict_action_chunk for action generation.

        Args:
            batch: Dictionary containing input tensors with keys:
                - STATE: Robot state tensor
                - ACTION: Ground truth action tensor (training only)
                - "tokenized_prompt": Language instruction tokens
                - "tokenized_prompt_mask": Attention mask for language tokens
                - EXTRA + ".actions_id_pad": Optional padding mask for actions
                - Image-related keys processed by _prepare_images

        Returns:
            If training: A tuple containing:
                - loss: Mean loss value as a tensor
                - loss_dict: Dictionary with intermediate loss values for debugging
            If inference: Output from predict_action_chunk (action predictions)
        """
        if self.training:
            if self._config.adapt_to_pi_aloha:
                batch[STATE] = self._pi_aloha_decode_state(batch[STATE])
                batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

            images, img_masks = self._prepare_images(batch)
            state = self._prepare_state(batch)
            actions = self._prepare_action(batch)

            lang_tokens = batch["tokenized_prompt"]
            lang_masks = batch["tokenized_prompt_mask"]
            actions_is_pad = batch.get(EXTRA + ".actions_id_pad")
            loss_dict = {}
            losses = self._model.forward(images, img_masks, lang_tokens, lang_masks, state, actions)
            loss_dict["losses_after_forward"] = losses.clone()

            if actions_is_pad is not None:
                in_episode_bound = ~actions_is_pad
                losses *= in_episode_bound.unsqueeze(-1)
                loss_dict["losses_after_in_ep_bound"] = losses.clone()

            # Remove padding
            losses = losses[:, :, : self._config.max_action_dim]
            loss_dict["losses_after_rm_padding"] = losses.clone()

            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict
        return self.predict_action_chunk(batch)

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict a chunk of actions from input batch.

        This method processes the input batch, prepares images, state, and language tokens,
        then uses the model to sample actions. The resulting actions are unpadded to match
        the original action dimension and optionally encoded for Pi Aloha compatibility.

        Args:
            batch: A dictionary containing input tensors including images, state information,
                and tokenized prompts with their masks.

        Returns:
            torch.Tensor: A tensor of predicted actions with shape matching the original
                action dimensions from the dataset statistics.
        """
        processed_batch = self._prepare_batch(batch)

        images, img_masks = self._prepare_images(processed_batch)
        state = self._prepare_state(processed_batch)
        lang_tokens = processed_batch["tokenized_prompt"].to(processed_batch[STATE].device)
        lang_masks = processed_batch["tokenized_prompt_mask"].to(processed_batch[STATE].device)

        actions = self._model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        )

        # Unpad actions
        original_action_dim = int(self._dataset_stats[ACTION]["shape"][-1])
        actions = actions[:, :, :original_action_dim]

        if self._config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    @property
    def sample_input(self) -> dict[str, torch.Tensor]:
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
        device = next(self._model.parameters()).device

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
        return list(range(self._config.chunk_size))

    @property
    def observation_delta_indices(self) -> list[int]:
        """Get indices of observations relative to the current timestep.

        Returns:
            list[int]: A list of relative observation indices.
        """
        return [0]

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._config.adapt_to_pi_aloha:
            batch[STATE] = self._pi_aloha_decode_state(batch[STATE])
        return batch

    def _prepare_images(self, batch: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Apply SmolVLA preprocessing to the images.

        This method processes image tensors from a batch by:
        1. Extracting the last frame if the input is a 5D tensor (video sequence)
        2. Optionally resizing images with padding to maintain aspect ratio
        3. Converting pixel values from [0.0, 1.0] range to [-1.0, 1.0] range as required by SigLIP
        4. Extracting or creating padding masks for each image
        Args:
            batch: A dictionary containing image tensors and optional padding masks.
                Image tensors should be 4D (B, C, H, W) or 5D (B, T, C, H, W).
                Optional padding masks are stored with keys prefixed by EXTRA.

        Returns:
            A tuple containing:
                - images: List of preprocessed image tensors, each with shape (B, C, H, W)
                    and pixel values in range [-1.0, 1.0]
                - img_masks: List of boolean mask tensors indicating valid (non-padded)
                    images in each batch position

        Raises:
            ValueError: If the batch is missing expected image features based on
                the dataset statistics configuration.
        """
        images = []
        img_masks = []

        batch_img_keys = Observation.get_flattened_keys(batch, IMAGES)
        all_keys = [key for key in self._dataset_stats if "image" in key]

        if len(batch_img_keys) != len(all_keys):
            msg = f"Some of the image features are missing from the batch. \
                    (batch: {batch.keys()}) (image_features:{all_keys})"
            raise ValueError(msg)
        # Preprocess image features present in the batch
        max_image_dim = 5
        for key in batch_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == max_image_dim else batch[key]
            if self._config.resize_imgs_with_padding is not None:
                img = _resize_with_pad(img, *self._config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if EXTRA + f".{key}_padding_mask" in batch:
                mask = batch[EXTRA + f".{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    @staticmethod
    def _pi_aloha_decode_state(state: torch.Tensor) -> torch.Tensor:
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = _aloha_gripper_to_angular(state[:, motor_idx])
        return state

    @staticmethod
    def _pi_aloha_encode_actions(actions: torch.Tensor) -> torch.Tensor:
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = _aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    @staticmethod
    def _pi_aloha_encode_actions_inv(actions: torch.Tensor) -> torch.Tensor:
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = _aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def _prepare_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare and normalize the state tensor from the input batch.

        Extracts the state tensor from the batch dictionary, selecting the last
        timestep if the tensor has more than 2 dimensions. The state is then
        padded to match the maximum state dimension specified in the configuration.

        Args:
            batch: A dictionary containing tensors, must include a STATE key
                with a tensor of shape (batch_size, seq_len, state_dim) or
                (batch_size, state_dim).

        Returns:
            A tensor of shape (batch_size, max_state_dim) containing the
            padded state representation.
        """
        state_dim = 2
        state = batch[STATE][:, -1, :] if batch[STATE].ndim > state_dim else batch[STATE]
        return _pad_vector(state, self._config.max_state_dim)

    def _prepare_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare and pad the action tensor from the batch.

        Args:
            batch: A dictionary containing tensors, must include the ACTION key
                with the action tensor to be padded.

        Returns:
            A dictionary containing the padded action tensor with dimensions
            extended to match the configured maximum action dimension.
        """
        return _pad_vector(batch[ACTION], self._config.max_action_dim)


def _resize_with_pad(img: torch.Tensor, width: int, height: int, pad_value: float = -1) -> torch.Tensor:
    # assume no-op when width height fits already
    img_dim = 4
    if img.ndim != img_dim:
        msg = f"(b,c,h,w) expected, but {img.shape}"
        raise ValueError(msg)

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    return F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)


def _pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    """Pad a tensor along its last dimension with zeros to reach a new dimension size.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension).

    Args:
        vector: Input tensor to pad. Can be of shape (batch_size x sequence_length x features_dimension)
            or (batch_size x features_dimension).
        new_dim: The target size for the last dimension after padding.

    Returns:
        A new tensor with the last dimension padded to `new_dim` with zeros.
        If the input tensor's last dimension already equals `new_dim`,
        the original tensor is returned unchanged.

    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def _normalize(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return x * (max_val - min_val) + min_val


def _safe_arcsin(value: torch.Tensor) -> torch.Tensor:
    # This ensures that the input stays within
    # [-1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def _aloha_gripper_to_angular(value: torch.Tensor) -> torch.Tensor:
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position: torch.Tensor, arm_length: float, horn_radius: float) -> torch.Tensor:
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return _safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return _normalize(value, min_val=0.4, max_val=1.5)


def _aloha_gripper_from_angular(value: torch.Tensor) -> torch.Tensor:
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = _unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _aloha_gripper_from_angular_inv(value: torch.Tensor) -> torch.Tensor:
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return _normalize(value, min_val=0.4, max_val=1.5)


def _get_safe_dtype(dtype: torch.dtype, device: str | torch.device) -> torch.dtype:
    """Get a safe dtype for the given device, handling compatibility issues.

    This function checks if the requested dtype is compatible with the given device
    and returns a safe alternative if necessary.
    Mps is currently not compatible with float64

    Args:
        dtype: The requested torch dtype.
        device: The device (as string or torch.device) where the tensor will be used.

    Returns:
        The original dtype if compatible with the device, or torch.float32 as a
        fallback when float64 is not supported (e.g., on MPS devices or Intel XPU
        devices without FP64 support).
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    if device == "xpu" and dtype == torch.float64:
        if hasattr(torch.xpu, "get_device_capability"):
            device_capability = torch.xpu.get_device_capability()
            # NOTE: Some Intel XPU devices do not support double precision (FP64).
            # The `has_fp64` flag is returned by `torch.xpu.get_device_capability()`
            # when available; if False, we fall back to float32 for compatibility.
            if not device_capability.get("has_fp64", False):
                logger.warning("Device %s does not support float64, using float32 instead.", device)
                return torch.float32
        else:
            logger.warning(
                "Device %s capability check failed. Assuming no support for float64, using float32 instead.",
                device,
            )
            return torch.float32
        return dtype
    return dtype


def _create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> torch.Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions.

    This function generates sinusoidal positional embeddings using a combination
    of sine and cosine functions with varying frequencies, commonly used in
    transformer-based models for encoding temporal information.

    Args:
        time: A 1D tensor of shape `(batch_size,)` containing time values.
        dimension: The dimension of the output embeddings. Must be divisible by 2.
        min_period: The minimum period for the sinusoidal functions.
        max_period: The maximum period for the sinusoidal functions.
        device: The device on which to create the embedding tensor. Defaults to CPU.

    Returns:
        A tensor of shape `(batch_size, dimension)` containing the sinusoidal
        positional embeddings, where the first half contains sine values and
        the second half contains cosine values.

    Raises:
        ValueError: If `dimension` is not divisible by 2.
        ValueError: If `time` tensor is not 1-dimensional.
    """
    if dimension % 2 != 0:
        msg = f"Dimension ({dimension}) must be divisible by 2"
        raise ValueError(msg)

    if time.ndim != 1:
        msg = "The time tensor is expected to be of shape `(batch_size, )`."
        raise ValueError(msg)

    dtype = _get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def _make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behavior.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      pad_masks: bool[B, N] true if its part of the input, false if padding.
      att_masks: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.

    Returns:
        bool[B, N, N] attention masks.

    Raises:
        ValueError: If input masks do not have the expected number of dimensions.
    """
    required_mask_dim = 2
    if att_masks.ndim != required_mask_dim:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != required_mask_dim:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def _pad_tensor(tensor: torch.Tensor, max_len: int, pad_value: float = 0) -> torch.Tensor:
    """Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """SmolVLA internal model.

    [Paper](https://arxiv.org/abs/2506.01844)

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig) -> None:
        """Initialize the SmolVLA model.

        Args:
            config: Configuration object containing model hyperparameters including:
                - vlm_model_name: Name/ID of the vision-language model to use
                - freeze_vision_encoder: Whether to freeze the vision encoder weights
                - train_expert_only: Whether to only train the expert layers
                - load_vlm_weights: Whether to load pretrained VLM weights
                - attention_mode: Attention mechanism configuration
                - num_expert_layers: Number of expert layers
                - num_vlm_layers: Number of VLM layers
                - self_attn_every_n_layers: Frequency of self-attention layers
                - expert_width_multiplier: Multiplier for expert layer width
                - max_state_dim: Maximum state dimension for state projection
                - max_action_dim: Maximum action dimension for action projections
                - add_image_special_tokens: Whether to add special image tokens
                - prefix_length: Length of the prefix sequence
        """
        super().__init__()
        self.config = config

        self.vlm_with_expert = _SmolVLMWithExpertModel(
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

    def set_requires_grad(self) -> None:
        """Set the requires_grad attribute for state projection parameters.

        Configures the gradient computation for the state projection module's
        parameters based on the train_state_proj setting in the model configuration.
        """
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    @staticmethod
    def _sample_noise(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    @staticmethod
    def _sample_time(bsize: int, device: torch.device) -> torch.Tensor:
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        return time_beta * 0.999 + 0.001

    def embed_prefix(  # noqa: PLR0914, PLR0915
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens to prepare for SmolVLM transformer processing.

        Args:
            images: List of image tensors to be embedded.
            img_masks: List of boolean masks for each image indicating valid regions.
            lang_tokens: Token IDs for language input to be embedded.
            lang_masks: Boolean mask for language tokens indicating valid tokens.
            state: Optional state tensor to be projected and included in the prefix.
                If None, state embedding is still computed.

        Returns:
            A tuple containing:
                - embs: Concatenated embeddings tensor of shape (batch_size, seq_len, embed_dim)
                    containing image, language, and state embeddings.
                - pad_masks: Boolean tensor of shape (batch_size, seq_len) indicating
                    valid (non-padded) positions.
                - att_masks: Boolean tensor of shape (batch_size, seq_len) for attention
                    masking, where True indicates positions that should be masked
                    (state tokens are masked from image/language attention).

        Note:
            If the total sequence length is less than `self.prefix_length`, the outputs
            are padded to match the required prefix length.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device),
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0],
                    dtype=torch.bool,
                    device=image_start_token.device,
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb *= torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask_ = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask_)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device),
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0],
                    dtype=torch.bool,
                    device=image_end_token.device,
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb *= math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        emb_dim = 2
        state_emb = state_emb[:, None, :] if state_emb.ndim == emb_dim else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = _pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = _pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = _pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed noisy actions and timestep to prepare for Expert Gemma processing.

        This method creates embeddings for the action prediction head by:
        1. Projecting noisy actions through an input projection layer
        2. Creating sinusoidal positional embeddings for the timestep
        3. Concatenating action and time embeddings and processing through an MLP
        4. Generating appropriate padding and attention masks
        Args:
            noisy_actions: Tensor of shape (batch_size, chunk_size, action_dim) containing
                noisy action samples from the diffusion process.
            timestep: Tensor of shape (batch_size,) containing the diffusion timestep
                values for each sample in the batch.

        Returns:
            A tuple containing:
                - embs: Tensor of shape (batch_size, chunk_size, hidden_size) containing
                    the fused action-time embeddings.
                - pad_masks: Boolean tensor of shape (batch_size, chunk_size) indicating
                    valid positions (all True for action tokens).
                - att_masks: Tensor of shape (batch_size, chunk_size) containing attention
                    mask values (all 1s) indicating that action tokens should not be
                    attended to by image, language, and state inputs.
        """
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = _create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(  # noqa: PLR0914
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors).

        This method implements the flow matching training objective for the diffusion model.
        It generates noisy action trajectories and trains the model to predict the noise
        direction (velocity field) that would denoise them.

        Args:
            images: List of image tensors from different camera views.
            img_masks: List of attention masks for each image tensor.
            lang_tokens: Tokenized language instruction tensor of shape (batch_size, seq_len).
            lang_masks: Attention masks for language tokens of shape (batch_size, seq_len).
            state: Robot state tensor containing proprioceptive information.
            actions: Ground truth action sequence tensor of shape (batch_size, chunk_size, action_dim).
            noise: Optional pre-sampled noise tensor. If None, noise is sampled internally.
            time: Optional time step tensor for the diffusion process. If None, time is sampled uniformly.

        Returns:
            torch.Tensor: Per-element MSE loss between predicted and target velocity fields,
                with shape (batch_size, chunk_size, action_dim)
        """
        if noise is None:
            noise = self._sample_noise(actions.shape, actions.device)

        if time is None:
            time = self._sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = _make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    def sample_actions(  # noqa: PLR0914
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform full inference forward pass to compute actions using a diffusion-based sampling process.

        This method generates actions by iteratively denoising a noise sample using the learned
        velocity field. It first computes embeddings for the prefix (images, language tokens, and state),
        then uses a cached key-value mechanism for efficient inference, and finally performs
        multi-step denoising to produce the final action sequence.

        Args:
            images: Input images tensor for visual conditioning.
            img_masks: Mask tensor indicating valid image regions.
            lang_tokens: Tokenized language instructions.
            lang_masks: Mask tensor indicating valid language tokens.
            state: Current state tensor of shape (batch_size, state_dim).
            noise: Optional pre-sampled noise tensor. If None, noise will be sampled
                with shape (batch_size, chunk_size, max_action_dim).

        Returns:
            Tensor: Predicted actions of shape (batch_size, chunk_size, max_action_dim),
                representing the denoised action sequence.
        """
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self._sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
        )
        prefix_att_2d_masks = _make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(
                input_x_t: torch.Tensor,
                current_timestep: torch.Tensor = time_tensor,
            ) -> torch.Tensor:
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                )

            v_t = denoise_step_partial_call(x_t)

            x_t += dt * v_t

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values: dict,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one de-noising step of the noise `x_t` at a given timestep.

        This method performs a single de-noising iteration in the diffusion process,
        transforming noisy action representations towards cleaner predictions.

        Args:
            prefix_pad_masks: Boolean tensor of shape (batch_size, prefix_len) indicating
                valid positions in the prefix sequence (e.g., vision/language tokens).
            past_key_values: Cached key-value pairs from the prefix forward pass,
                used to avoid recomputing prefix representations.
            x_t: Noisy action tensor at the current timestep to be denoised.
            timestep: Current diffusion timestep indicating the noise level.

        Returns:
            Tensor of shape (batch_size, chunk_size, action_dim) containing the
            predicted denoised action output after projection.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = _make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


def _apply_rope(x: torch.Tensor, positions: torch.Tensor, max_wavelength: int = 10_000) -> torch.Tensor:
    """Applies Rotary Position Embedding (RoPE) to the input tensor.

    RoPE encodes positional information by rotating the input features based on
    their position, allowing the model to learn relative positional relationships.

    Args:
        x: Input tensor of shape [B, L, H, D] where B is batch size, L is sequence
            length, H is number of heads, and D is the head dimension.
        positions: Position indices tensor of shape [B, L] containing the position
            of each token in the sequence.
        max_wavelength: Maximum wavelength for the sinusoidal position encoding.
            Controls the range of frequencies used. Defaults to 10,000.

    Returns:
        torch.Tensor: The input tensor with rotary position embeddings applied,
            same shape as input [B, L, H, D] and same dtype as input.
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


def _get_intermediate_size(hidden_dim: int, ffn_dim_multiplier: int = 4, multiple_of: int = 256) -> int:
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class _SmolVLMWithExpertModel(nn.Module):
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        *,
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
        device: str = "auto",
    ) -> None:
        super().__init__()

        (
            auto_config_cls,
            auto_model_cls,
            auto_model_for_image_text_to_text_cls,
            auto_processor_cls,
            smol_vlm_for_conditional_generation_cls,
        ) = _lazy_import_transformers()

        if load_vlm_weights:
            logger.info("Loading  %s weights ...", model_id)
            self.vlm = auto_model_for_image_text_to_text_cls.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            config = auto_config_cls.from_pretrained(model_id)
            self.vlm = smol_vlm_for_conditional_generation_cls(config=config)
        self.processor = auto_processor_cls.from_pretrained(model_id)
        if num_vlm_layers > 0:
            logger.info("Reducing the number of VLM layers to %s ...", num_vlm_layers)
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config
        # Smaller lm expert
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)  # hidden_size // 2
        lm_expert_config.intermediate_size = _get_intermediate_size(int(hidden_size * expert_width_multiplier))
        lm_expert_config.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers > 0 and len(self.get_vlm_model().text_model.layers) % num_expert_layers == 0:
            lm_expert_config.num_hidden_layers = num_expert_layers
            msg = (
                f"Number of layers in the VLM {len(self.get_vlm_model().text_model.layers)} are "
                f"not multiple of num_expert_layers {num_expert_layers}"
            )
            raise RuntimeError(msg)
        self.lm_expert = auto_model_cls.from_config(lm_expert_config)

        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = self_attn_every_n_layers
        if "cross" in attention_mode:
            # Reshape qkv projections to have the same input dimension as the vlm
            for layer_idx in range(len(self.lm_expert.layers)):
                if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                    continue
                self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
                self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
        # Remove unused embed_tokens
        self.lm_expert.embed_tokens = None

        self.num_attention_heads = self.config.text_config.num_attention_heads
        self.num_key_value_heads = self.config.text_config.num_key_value_heads

        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_mode = attention_mode
        self.expert_hidden_size = lm_expert_config.hidden_size
        self.set_requires_grad()

    def get_vlm_model(self) -> torch.nn.Module:
        return self.vlm.model

    def set_requires_grad(self) -> None:
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            # To avoid unused params issue with distributed training
            last_layers = [self.num_vlm_layers - 1]
            if self.num_vlm_layers != self.num_expert_layers and self.num_vlm_layers % self.num_expert_layers == 0:
                last_layers.append(self.num_vlm_layers - 2)
            frozen_layers = [
                "lm_head",
                "text_model.model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")  # noqa: PERF401

            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False
        # To avoid unused params issue with distributed training
        for name, params in self.lm_expert.named_parameters():
            if "lm_head" in name:
                params.requires_grad = False

    def train(self, mode: bool = True) -> None:  # noqa: FBT002, FBT001 : torch is not compatible with the fix
        super().train(mode)

        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()

        if self.train_expert_only:
            self.vlm.eval()

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        patch_attention_mask = None
        # Get sequence from the vision encoder
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=patch_attention_mask,
            )
            .last_hidden_state
        )
        # Modality projection & resampling
        return self.get_vlm_model().connector(image_hidden_states)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def forward_attn_layer(  # noqa: PLR0914
        self,
        model_layers: list,
        inputs_embeds: list,
        layer_idx: int,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        *,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values: dict | None = None,
    ) -> tuple[list[torch.Tensor], dict | None]:
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            hidden_states_ = layer.input_layernorm(hidden_states)

            input_shape = hidden_states_.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states_ = hidden_states_.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states_).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states_).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states_).view(hidden_shape)

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
            indexed_position_ids = position_ids[:, :seq_len]
            indexed_attention_mask = attention_mask[:, :seq_len, :seq_len]
        else:
            indexed_position_ids = position_ids
            indexed_attention_mask = attention_mask

        attention_mask_ = indexed_attention_mask
        position_ids_ = indexed_position_ids

        query_states = _apply_rope(query_states, position_ids_)
        key_states = _apply_rope(key_states, position_ids_)

        if use_cache:
            if past_key_values is None:
                past_key_values = {}

            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # to-do here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len`
                # before. so we create an empty cache, with just one cuda malloc, and if (in autoregressive case)
                # we reach  the max len, then we (for instance) double the cache size. This implementation
                #  already exists in `transformers`. (molbap)
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)

        attention_interface = self.get_attention_interface()

        att_output = attention_interface(
            attention_mask_,
            batch_size,
            head_dim,
            query_states,
            key_states,
            value_states,
        )
        return [att_output], past_key_values

    def forward_cross_attn_layer(  # noqa: PLR0914
        self,
        model_layers: list[nn.Module],
        inputs_embeds: list[torch.Tensor],
        layer_idx: int,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        *,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values: dict | None = None,
    ) -> tuple[list[torch.Tensor], dict | None]:
        attention_interface = self.get_attention_interface()

        att_outputs = []
        required_embeds_num = 2

        if not (
            len(inputs_embeds) == required_embeds_num
            or (use_cache and past_key_values is not None and not fill_kv_cache)
        ):
            msg = f"Both len(inputs_embeds) == {len(inputs_embeds)} and past_key_values is {past_key_values}"
            raise ValueError(msg)

        if len(inputs_embeds) == required_embeds_num and not past_key_values:
            # Prefix attention
            seq_len = inputs_embeds[0].shape[1]
            position_id, expert_position_id = position_ids[:, :seq_len], position_ids[:, seq_len:]
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

            layer = model_layers[0][layer_idx]

            hidden_states = layer.input_layernorm(inputs_embeds[0])

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            query_states = _apply_rope(query_state, position_id)
            key_states = _apply_rope(key_state, position_id)

            att_output = attention_interface(
                prefix_attention_mask,
                batch_size,
                head_dim,
                query_states,
                key_states,
                value_states,
            )
            att_outputs.append(att_output)
        else:
            expert_position_id = position_ids

        if use_cache:
            if past_key_values is None:
                past_key_values = {}
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # to-do here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len`
                # before. so we create an empty cache, with just one cuda malloc, and if (in autoregressive case)
                # we reach the max len, then we (for instance) double the cache size. This implementation
                # already exists in `transformers`. (molbap)
                key_states = past_key_values[layer_idx]["key_states"]
                value_states = past_key_values[layer_idx]["value_states"]

        # Expert
        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])

            expert_input_shape = expert_hidden_states.shape[:-1]
            expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)

            expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)

            casted_key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(
                *key_states.shape[:2],
                -1,
            )
            expert_key_states = expert_layer.self_attn.k_proj(casted_key_states).view(
                *casted_key_states.shape[:-1],
                -1,
                expert_layer.self_attn.head_dim,
            )  # k_proj should have same dim as kv

            casted_value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(
                *value_states.shape[:2],
                -1,
            )
            expert_value_states = expert_layer.self_attn.v_proj(casted_value_states).view(
                *casted_value_states.shape[:-1],
                -1,
                expert_layer.self_attn.head_dim,
            )

            expert_position_id -= torch.min(expert_position_id, dim=1, keepdim=True).values  # start from 0
            expert_attention_mask = attention_mask[
                :,
                -inputs_embeds[1].shape[1] :,
                : expert_key_states.shape[1] :,
            ]  # take into account kv

            expert_query_states = _apply_rope(expert_query_state, expert_position_id)

            att_output = attention_interface(
                expert_attention_mask,
                batch_size,
                head_dim,
                expert_query_states,
                expert_key_states,
                expert_value_states,
            )
            att_outputs.append(att_output)
        else:
            att_outputs.append(None)

        return att_outputs, past_key_values

    def get_model_layers(self, models: list) -> list:
        vlm_layers = []
        expert_layers = []
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layer = None
            else:
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                expert_layer = models[1].layers[expert_layer_index]
            vlm_layers.append(models[0].layers[i])
            expert_layers.append(expert_layer)
        return [vlm_layers, expert_layers]

    def forward(  # noqa: PLR0914, PLR0912
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: dict | None,
        inputs_embeds: list[torch.Tensor | None],
        *,
        use_cache: bool = False,
        fill_kv_cache: bool = False,
    ) -> tuple:
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)

        for hidden_states in inputs_embeds:
            # to-do this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        # RMSNorm
        num_layers = self.num_vlm_layers
        head_dim = self.vlm.config.text_config.head_dim
        for layer_idx in range(num_layers):
            if (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
            ):
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]  # in case of self_attn
                if hidden_states is not None:
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    att_out = att_output[:, start:end]
                    out_emb = layer.self_attn.o_proj(att_out)

                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values

    def get_attention_interface(self) -> Callable:
        return self.eager_attention_forward

    def eager_attention_forward(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
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

        # Attention here is upcasted to float32 to match the original eager implementation.
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
        return att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)
