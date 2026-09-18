# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Local subclasses of the physicalai base Policy and Model.

These wrap the physicalai base classes so new, template-specific behavior can
be layered on without modifying physicalai source. NewPolicy and NewPolicyModel
inherit from these instead of the physicalai base classes directly.
"""

from __future__ import annotations

import dataclasses
import inspect
from abc import abstractmethod
from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import IO, Any, Self, cast

from jsonargparse import FromConfigMixin
import torch

from physicalai.policies.base import Model as BaseModel
from physicalai.policies.base import Policy as BasePolicy

from .config import NewPolicyModelConfig


class TemplateModel(BaseModel, FromConfigMixin):
    """Model base with non-strict jsonargparse config construction."""

    @classmethod
    def from_config(cls, config: object) -> Self:
        if dataclasses.is_dataclass(config) and not isinstance(config, type):
            values = {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
        elif isinstance(config, Mapping):
            values = dict(config)
        else:
            return super().from_config(cast("str | PathLike[str]", config))

        parameters = inspect.signature(cls.__init__).parameters
        return super().from_config({name: value for name, value in values.items() if name in parameters})

class TemplatePolicy(BasePolicy):
    """Extension point for template-specific Policy behavior."""

    _config: NewPolicyModelConfig | None

    @abstractmethod
    def configure_model(self) -> None:
        """Materialize the model and processors once from the resolved config."""

    @abstractmethod
    def setup(self, stage: str) -> None:
        """Resolve and validate the policy feature contract against its data source."""

    @property
    def config(self) -> NewPolicyModelConfig:
        """Return the resolved policy-owned model config."""
        if self._config is None:
            raise RuntimeError("Policy config is not initialized")
        return self._config

    @property
    def _config_available(self) -> bool:
        """Whether the policy has resolved a model config."""
        return self._config is not None

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | IO[bytes],
        map_location: torch.device | str | int | Callable | dict | None = None,
        hparams_file: str | Path | None = None,
        strict: bool | None = None,  # noqa: FBT001
        weights_only: bool | None = None,  # noqa: FBT001
        **kwargs: Any,  # noqa: ANN401
    ) -> Self:
        """Load a Lightning checkpoint without resolving pretrained artifacts."""
        kwargs["pretrained_name_or_path"] = None
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self._config is None:
            raise RuntimeError("Policy config is not initialized")
        checkpoint["model_config"] = self._config.to_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        config_data = checkpoint.get("model_config")
        if not isinstance(config_data, Mapping):
            return

        resolved_config = NewPolicyModelConfig.from_dict(config_data)
        if self._config is not None:
            if self._config != resolved_config:
                raise ValueError("Checkpoint feature contract does not match the initialized policy")
            return

        self._config = resolved_config
        self.configure_model()
