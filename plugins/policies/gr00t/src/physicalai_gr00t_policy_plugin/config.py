# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Typed GR00T fine-tuning settings that map onto the native CLI."""

from __future__ import annotations

import dataclasses
import types
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

DEFAULT_BASE_MODEL = "nvidia/GR00T-N1.7-3B"

# Fields that configure the plugin rather than upstream ``launch_finetune.py``.
_PLUGIN_FIELDS = frozenset({"extra_args", "backbone_source"})
_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}


@dataclass(frozen=True)
class Gr00tTrainConfig:
    """Native GR00T N1.7 fine-tuning settings.

    Field names mirror upstream ``FinetuneConfig``; :meth:`to_argv` renders the
    exact flags that ``gr00t/experiment/launch_finetune.py`` accepts.

    Attributes:
        dataset_path: LeRobot v2.1 dataset with ``meta/modality.json``.
        output_dir: Native output directory; the final checkpoint is written here.
        embodiment_tag: Upstream embodiment tag name or value, such as ``LIBERO_PANDA``.
        base_model_path: Hugging Face repository or local base checkpoint.
        backbone_source: Development-only replacement for the checkpoint's
            backbone repository. Use it only when the upstream backbone
            repository is unavailable; packages record the substitution.
        extra_args: Additional native flags appended unchanged.
    """

    dataset_path: str
    output_dir: str
    embodiment_tag: str
    base_model_path: str = DEFAULT_BASE_MODEL
    max_steps: int = 10000
    global_batch_size: int = 64
    learning_rate: float = 1e-4
    save_steps: int = 1000
    save_total_limit: int = 5
    save_only_model: bool = False
    dataloader_num_workers: int = 2
    num_gpus: int = 1
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    use_wandb: bool = False
    modality_config_path: str | None = None
    backbone_source: str | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def to_argv(self) -> list[str]:
        """Render native ``launch_finetune.py`` flags.

        Returns:
            Ordered native argument tokens.
        """
        argv: list[str] = []
        for item in dataclasses.fields(self):
            if item.name in _PLUGIN_FIELDS:
                continue
            value = getattr(self, item.name)
            flag = "--" + item.name.replace("_", "-")
            if isinstance(value, bool):
                argv.append(flag if value else flag.replace("--", "--no-", 1))
            elif value is not None:
                argv.extend([flag, str(value)])
        argv.extend(self.extra_args)
        return argv

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any], *, dataset: Path, output_dir: Path) -> Gr00tTrainConfig:
        """Build a config from Studio settings plus Studio-owned paths.

        Returns:
            A config whose dataset and output paths come from Studio.

        Raises:
            ValueError: If *settings* contains unknown fields.
        """
        known = {item.name for item in dataclasses.fields(cls)}
        unknown = sorted(set(settings) - known)
        if unknown:
            msg = f"Unknown GR00T settings: {', '.join(unknown)}."
            raise ValueError(msg)
        values = dict(settings)
        if "extra_args" in values:
            values["extra_args"] = tuple(values["extra_args"])
        values["dataset_path"] = str(dataset)
        values["output_dir"] = str(output_dir)
        return cls(**values)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return a JSON schema describing the settings.

        Returns:
            JSON schema with defaults and required fields.
        """
        hints = get_type_hints(cls)
        properties: dict[str, Any] = {}
        required = []
        for item in dataclasses.fields(cls):
            properties[item.name] = _json_property(hints[item.name], item)
            if item.default is dataclasses.MISSING and item.default_factory is dataclasses.MISSING:
                required.append(item.name)
        return {
            "title": "GR00T N1.7 fine-tuning",
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }


def _json_property(annotation: Any, item: dataclasses.Field[Any]) -> dict[str, Any]:  # ruff: ignore[any-type]
    origin = get_origin(annotation)
    if origin is tuple:
        schema: dict[str, Any] = {"type": "array", "items": {"type": "string"}}
    else:
        members = get_args(annotation) if origin in {Union, types.UnionType} else (annotation,)
        names = ["null" if member is type(None) else _JSON_TYPES.get(member.__name__, "string") for member in members]
        schema = {"type": names[0] if len(names) == 1 else names}
    if item.default is not dataclasses.MISSING:
        schema["default"] = item.default
    return schema
