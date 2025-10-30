# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixin classes for handling PyTorch model checkpoints."""

from copy import copy
from os import PathLike
from typing import Self

import torch
import yaml

from getiaction.config.instantiate import instantiate_obj_from_dict

from .mixin_export import CONFIG_KEY


class FromCheckpoint:
    """Mixin class for loading torch models from checkpoints."""

    @classmethod
    def load_checkpoint(
        cls,
        snapshot: dict | PathLike | str,
    ) -> Self:
        """Load model state from a snapshot dictionary or file.

        This class method reconstructs a model instance from a snapshot containing
        the model's configuration and state dictionary. The snapshot can be provided
        either as a dictionary or as a path to a saved snapshot file.

        Args:
            snapshot (dict | PathLike | str): Either a dictionary containing the model's
                state_dict and configuration, or a path (string or PathLike object) to
                a saved snapshot file. When provided as a path, the snapshot is loaded
                using torch.load with CPU mapping and weights_only=True for security.

        Returns:
            Self: A new instance of the class initialized with the configuration from
                the snapshot.

        Raises:
            NotImplementedError: If the class does not implement the `from_dataclass`
                method, which is required for instantiation from the loaded configuration.

        Note:
            The snapshot must contain a configuration stored under the key defined by
            GETIACTION_CONFIG_KEY. This configuration is parsed as YAML and used to
            instantiate the appropriate dataclass configuration object, which is then
            passed to the `from_dataclass` method to create the model instance.
        """
        state_dict = {}
        if isinstance(snapshot, (str, PathLike)):
            state_dict = torch.load(snapshot, map_location="cpu", weights_only=True)  # nosemgrep
        else:
            state_dict = copy(snapshot)

        config = instantiate_obj_from_dict(yaml.safe_load(state_dict[CONFIG_KEY]))
        state_dict.pop(CONFIG_KEY)

        if hasattr(cls, "from_dataclass") and callable(cls.from_dataclass):  # type: ignore [attr-defined]
            return cls.from_dataclass(config)  # type: ignore [attr-defined]

        msg = "`FromCheckpoint` mixin requires the target class to implement `from_dataclass()` method."
        raise NotImplementedError(msg)
