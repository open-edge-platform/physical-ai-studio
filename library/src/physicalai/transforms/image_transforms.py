# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Image augmentation transforms adapted from LeRobot (Apache-2.0)."""

from __future__ import annotations

import collections.abc
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2 import functional as F  # noqa: N812

_EXPECTED_SHARPNESS_LEN = 2


class RandomChoiceApply(Transform):
    """Apply a random subset of N transforms from a list of transforms.

    Similar to :class:`torchvision.transforms.v2.RandomChoice`, but samples
    *multiple* transforms per forward pass using weighted multinomial sampling
    *without replacement*. This produces more diverse augmentations than
    applying all transforms every time.

    Args:
        transforms: Sequence of callable transforms.
        p: Per-transform sampling weights. Weights are normalised internally
            so they need not sum to 1. If ``None`` (default), uniform weights
            are used.
        n_subset: Number of transforms to apply per forward call. If ``None``,
            all transforms are applied. Must be in ``[1, len(transforms)]``.
        random_order: If ``True``, the selected transforms are applied in a
            random order. If ``False`` (default), they are applied in the
            order they appear in *transforms*.

    Example:
        >>> from torchvision.transforms import v2
        >>> tfs = [
        ...     v2.ColorJitter(brightness=(0.8, 1.2)),
        ...     v2.ColorJitter(contrast=(0.8, 1.2)),
        ...     v2.RandomAffine(degrees=5),
        ... ]
        >>> augment = RandomChoiceApply(tfs, n_subset=2)
    """

    def __init__(
        self,
        transforms: Sequence[Callable[..., Any]],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialise with a pool of transforms and sampling parameters.

        Raises:
            TypeError: If *transforms* is not a sequence or *n_subset* is not an int.
            ValueError: If *p* length mismatches *transforms* or *n_subset* is out of range.
        """
        super().__init__()
        if not isinstance(transforms, Sequence):
            msg = "Argument transforms should be a sequence of callables"
            raise TypeError(msg)

        if p is None:
            p = [1.0] * len(transforms)
        elif len(p) != len(transforms):
            msg = f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            raise ValueError(msg)

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            msg = "n_subset should be an int or None"
            raise TypeError(msg)
        elif not (1 <= n_subset <= len(transforms)):
            msg = f"n_subset should be in the interval [1, {len(transforms)}]"
            raise ValueError(msg)

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

    def forward(self, *inputs: Any) -> Any:  # noqa: ANN401
        """Sample a random subset of transforms and apply them sequentially.

        Returns:
            Transformed output(s) matching the input signature.
        """
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        for idx in selected_indices.tolist():
            outputs = self.transforms[idx](*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs  # type: ignore[possibly-undefined]

    def extra_repr(self) -> str:
        """Return a string representation of the transform configuration."""
        return f"transforms={self.transforms}, p={self.p}, n_subset={self.n_subset}, random_order={self.random_order}"


class RandomSharpness(Transform):
    """Randomly adjust the sharpness of an image or video.

    Unlike :class:`torchvision.transforms.v2.RandomAdjustSharpness` which
    applies a *fixed* sharpness factor with some probability,
    ``RandomSharpness`` samples a *continuous* random factor from a uniform
    distribution on every call, producing more diverse augmentations.

    A sharpness factor of 0 gives a blurred image, 1 gives the original,
    and 2 doubles the sharpness.

    Input tensors are expected to have ``[..., 1 or 3, H, W]`` shape.

    Args:
        sharpness: Either a single non-negative float ``s`` (interpreted as
            the range ``[max(0, 1-s), 1+s]``), or a ``[min, max]`` sequence.

    Example:
        >>> jitter = RandomSharpness(sharpness=[0.5, 1.5])
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        """Initialise with a sharpness range."""
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    @staticmethod
    def _check_input(sharpness: float | Sequence[float]) -> tuple[float, float]:
        if isinstance(sharpness, (int, float)):
            if sharpness < 0:
                msg = "If sharpness is a single number, it must be non negative."
                raise ValueError(msg)
            sharpness_range = [1.0 - sharpness, 1.0 + sharpness]
            sharpness_range[0] = max(sharpness_range[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == _EXPECTED_SHARPNESS_LEN:
            sharpness_range = [float(v) for v in sharpness]
        else:
            msg = f"{sharpness=} should be a single number or a sequence with length 2."
            raise TypeError(msg)

        if not 0.0 <= sharpness_range[0] <= sharpness_range[1]:
            msg = f"sharpness values should be between (0., inf), but got {sharpness_range}."
            raise ValueError(msg)

        return float(sharpness_range[0]), float(sharpness_range[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:  # noqa: ARG002
        """Sample a random sharpness factor from the configured range.

        Returns:
            Dictionary containing the sampled ``sharpness_factor``.
        """
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:  # noqa: ANN401
        """Apply the sampled sharpness factor to the input.

        Returns:
            The input with adjusted sharpness.
        """
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=params["sharpness_factor"])
