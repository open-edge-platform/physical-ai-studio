# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dummy policy for testing usage."""

from collections import deque
from collections.abc import Iterable

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from getiaction.config import FromConfig
from getiaction.data.utils import infer_batch_size
from getiaction.export import FromCheckpoint
from getiaction.policies.dummy.config import DummyConfig


class Dummy(nn.Module, FromConfig, FromCheckpoint):
    """A dummy model for testing training and evaluation loops.

    This model simulates behavior of an action-predicting model by returning
    random actions and optionally managing temporal ensembles or action queues.
    """

    def __init__(
        self,
        action_shape: list | tuple,
        action_dtype: torch.dtype | str = torch.float32,
        action_min: float | None = None,
        action_max: float | None = None,
        n_action_steps: int = 1,
        temporal_ensemble_coeff: float | None = None,
        n_obs_steps: int = 1,
        horizon: int | None = None,
    ) -> None:
        """Initialize the DummyModel.

        Args:
            action_shape (list | tuple): The shape of a single action.
            action_dtype (torch.dtype | str): The dtype to put the action in.
            action_min (float | None): Minimum value of action given dtype.
            action_max (float | None): Maximum value of action given dtype.
            n_action_steps (int, optional): Number of action steps per chunk.
                Defaults to 1.
            temporal_ensemble_coeff (float | None, optional): Coefficient for
                temporal ensembling. If `None`, an action queue is used instead.
                Defaults to `None`.
            n_obs_steps (int, optional): Number of observation steps.
                Defaults to 1.
            horizon (int | None, optional): Prediction horizon. If `None`,
                defaults to `n_action_steps`.
        """
        super().__init__()

        self.n_action_steps = n_action_steps
        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        self.action_shape = self._validate_action_shape(action_shape)
        self.action_dtype = self._validate_dtype(action_dtype)
        self.action_min, self.action_max = self._validate_min_max(
            min_=action_min,
            max_=action_max,
        )

        # default horizon = number of action steps
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon if horizon is not None else n_action_steps

        if self.temporal_ensemble_coeff is not None:
            # simple placeholder for temporal ensemble
            self.temporal_buffer: None = None
        else:
            self._action_queue: deque = deque(maxlen=self.n_action_steps)

        # dummy parameter for optimizer and backward
        self.dummy_param = nn.Parameter(torch.zeros(1))

    @property
    def config(self) -> DummyConfig:
        """Get the configuration of the Dummy model.

        Returns:
            DummyConfig: The configuration dataclass instance.
        """
        return DummyConfig(action_shape=self.action_shape)

    @property
    def sample_input(self) -> dict[str, torch.Tensor]:
        """Generate a sample input dictionary for the model.

        This method creates a dictionary containing random tensor data that matches
        the expected input format and shape for the model. Useful for testing and
        validation purposes.

        Returns:
            dict[str, torch.Tensor]: A dictionary with a single key "action" containing
                a randomly initialized tensor of shape (1, action_shape).
        """
        return {
            "action": torch.randn((1, *self.action_shape)),
        }

    @property
    def extra_export_args(self) -> dict:
        """Additional export arguments for model conversion.

        This property provides extra configuration parameters needed when exporting
        the model to different formats, particularly ONNX format.

        Returns:
            dict: A dictionary containing format-specific export arguments.

        Example:
            >>> extra_args = model.extra_export_args()
            >>> print(extra_args)
            {'onnx': {'output_names': ['action']}}
        """
        extra_args = {}
        extra_args["onnx"] = {
            "output_names": ["action"],
        }

        return extra_args

    @property
    def observation_delta_indices(self) -> list[int]:
        """Get indices of observations relative to the current timestep.

        Returns:
            list[int]: A list of relative observation indices.
        """
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        """Get indices of actions relative to the current timestep.

        Returns:
            list[int]: A list of relative action indices.
        """
        return list(range(0 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        """Return reward indices.

        Currently returns `None` as rewards are not implemented.

        Returns:
            None
        """
        return None

    def reset(self) -> None:
        """Reset internal buffers.

        Clears the temporal buffer (if using temporal ensemble) or the
        action queue (otherwise).
        """
        if self.temporal_ensemble_coeff is not None:
            self.temporal_buffer = None
        else:
            self._action_queue.clear()

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select a single action from the model.

        If temporal ensembling is enabled, returns the mean over predicted
        actions. Otherwise, actions are queued and returned sequentially.

        Args:
            batch (dict[str, torch.Tensor]): A batch of input observations.

        Returns:
            torch.Tensor: A tensor representing the selected action.
        """
        self.eval()

        if self.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            # simple stand-in for ensembler: just take mean
            return actions.mean(dim=1)

        # Handle action queue logic
        if len(self._action_queue) == 0:
            # chunk gets [B, n, dim]
            actions = self.predict_action_chunk(batch)
            for t in range(self.n_action_steps):
                self._action_queue.append(actions[:, t])  # queue [B, dim]
        return self._action_queue.popleft()

    @staticmethod
    def _sample(
        action_shape: tuple[int, ...] | list[int],
        dtype: torch.dtype | str,
        min_: float | None,
        max_: float | None,
    ) -> torch.Tensor:
        """Generate a random tensor for the given action shape.

        Sampling behavior depends on the provided bounds:

        - If both ``min_`` and ``max_`` are ``None``, a default random
          distribution is used (``torch.rand`` for floating-point dtypes or
          ``torch.randint`` across the full integer dtype range).

        - If both ``min_`` and ``max_`` are provided, values are sampled
          uniformly within the specified range.

        - If exactly one of ``min_`` or ``max_`` is provided, the tensor is
          first generated using the default distribution and then clamped to
          the provided bound.

        Args:
            action_shape: Shape of the generated tensor.
            dtype: Output dtype, given as a ``torch.dtype`` or string
                identifier.
            min_: Optional lower bound for values.
            max_: Optional upper bound for values.

        Returns:
            torch.Tensor: A tensor of shape ``action_shape`` containing random
            values sampled according to the rules above.
        """
        is_float = torch.is_floating_point(torch.tensor([], dtype=dtype))

        # Neither bound given -> default distribution
        if min_ is None and max_ is None:
            if is_float:
                return torch.rand(action_shape, dtype=dtype)
            info = torch.iinfo(dtype)
            return torch.randint(info.min, info.max, action_shape, dtype=dtype)

        # Both bounds given -> sample within [min_, max_]
        if min_ is not None and max_ is not None:
            if is_float:
                return min_ + (max_ - min_) * torch.rand(action_shape, dtype=dtype)
            return torch.randint(min_, max_ + 1, action_shape, dtype=dtype)

        # Only one bound given -> clamp
        if is_float:
            out = torch.rand(action_shape, dtype=dtype)
        else:
            info = torch.iinfo(dtype)
            out = torch.randint(info.min, info.max, action_shape, dtype=dtype)

        if min_ is not None:
            out = out.clamp(min=min_)
        if max_ is not None:
            out = out.clamp(max=max_)

        return out

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict a chunk of random actions.

        Args:
            batch (dict[str, torch.Tensor]): A batch of input observations.

        Returns:
            torch.Tensor: A tensor of shape
                `(batch_size, n_action_steps, *action_shape)`.
        """
        batch_size = infer_batch_size(batch)
        full_shape = (batch_size, self.n_action_steps, *tuple(self.action_shape))
        return self._sample(
            action_shape=full_shape,
            dtype=self.action_dtype,
            min_=self.action_min,
            max_=self.action_max,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """Forward pass through the model.

        If in training mode, returns an MSE loss between predictions and
        zeros. If in evaluation mode, returns a chunk of random actions.

        Args:
            batch (dict[str, torch.Tensor]): A batch of input observations.

        Returns:
            tuple[torch.Tensor, dict]:
                - If training: a scalar loss tensor and dictionary of loss.
                - If evaluating: predicted actions tensor.
        """
        if self.training:
            batch_size = infer_batch_size(batch)
            # pred now depends on a parameter so it has grad_fn
            pred = (
                torch.randn(
                    (batch_size, self.n_action_steps, *self.action_shape),
                    device=self.dummy_param.device,
                )
                + self.dummy_param
            )
            target = torch.zeros_like(pred)
            loss = F.mse_loss(pred, target)
            return loss, {"loss_mse": loss}
        return self.predict_action_chunk(batch)

    @staticmethod
    def _validate_action_shape(shape: list | tuple) -> list | tuple:
        """Validate and normalize the action shape.

        Args:
            shape (list | tuple): The input shape to validate.

        Returns:
            list | tuple: A validated list or tuple object.

        Raises:
            ValueError: If `shape` is `None`.
            TypeError: If `shape` is not a valid type (e.g., string).
        """
        if shape is None:
            msg = "Action is missing a 'shape' key in its features dictionary."
            raise ValueError(msg)

        if isinstance(shape, torch.Size):
            return shape

        if isinstance(shape, str):
            msg = f"Shape for action '{shape}' must be a sequence of numbers, but received a string."
            raise TypeError(msg)

        if isinstance(shape, Iterable):
            return list(shape)

        msg = f"The 'action_shape' argument must be a list or tuple, but received type {type(shape).__name__}."
        raise TypeError(msg)

    @staticmethod
    def _validate_dtype(dtype: torch.dtype | str | None) -> torch.dtype:
        """Validate and resolve dtype.

        Args:
            dtype: The dtype to validate. May be a ``torch.dtype`` instance,
                a string representing a dtype (e.g., ``"float32"``,
                ``"double"``, ``"int"``), or ``None``.

        Returns:
            torch.dtype: A fully-resolved PyTorch dtype.

        Raises:
            ValueError: If the provided string cannot be resolved to a valid
                ``torch.dtype``.
        """
        # if None, assume float32
        if dtype is None:
            return torch.float32

        # if already a dtype then return
        if isinstance(dtype, torch.dtype):
            return dtype

        # ensure string and lower
        key = str(dtype).lower()

        # common aliases for dtypes
        alias_map = {
            "float": "float32",
            "fp32": "float32",
            "double": "float64",
            "fp64": "float64",
            "half": "float16",
            "long": "int64",
            "int": "int32",
            "short": "int16",
            "byte": "uint8",
            "bf16": "bfloat16",
        }
        key = alias_map.get(key, key)

        attr = getattr(torch, key, None)
        if isinstance(attr, torch.dtype):
            return attr

        msg = f"Unknown dtype string: {dtype}"
        raise ValueError(msg)

    @staticmethod
    def _validate_min_max(
        min_: float | None = None,
        max_: float | None = None,
    ) -> tuple[float | None, float | None]:
        """Validate range for action space.

        Args:
            min_: The lower bound of the range, or ``None``.
            max_: The upper bound of the range, or ``None``.

        Returns:
            tuple[float | None, float | None]: A tuple ``(min_, max_)`` where
                both values are either the validated inputs or ``(None, None)``
                if the range is unspecified.

        Raises:
            ValueError: If both bounds are provided and ``max_`` is smaller
                than ``min_``.
        """
        if (min_ is None) or (max_ is None):
            return (min_, max_)
        # only assumption is that min is smaller than max
        if max_ < min_:
            msg = f"Max cannot be smaller than min: {max_} < {min_}"
            raise ValueError(msg)
        return (min_, max_)
