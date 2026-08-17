# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Loading released XR-1 checkpoints into this implementation.

The published checkpoints use the same parameter names as this port, because the
architecture was translated without renaming modules. Verified against
``XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa`` (1120 tensors): every DiT, projector,
timestep-embedder, sink and backbone key matches one-for-one.

Two things still stand between a checkpoint and a working model, and both are
handled here:

* **Checkpoint layout.** Benchmark checkpoints ship as sharded safetensors, while
  the 5B base ships a single DeepSpeed-style ``model_states.pt`` whose weights live
  under a ``module`` key, sometimes with a ``module.`` prefix on every name.
* **Config agreement.** The released weights are sized for a 60-dimensional
  dual-arm action space with a 36-layer, 1024-wide action expert. Loading them into
  a differently sized :class:`~physicalai.policies.xr1.config.XR1Config` fails deep
  inside ``load_state_dict`` with an unreadable shape error, so
  :func:`infer_config_overrides` reads the sizes back out of the checkpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import nn

logger = logging.getLogger(__name__)

SAFETENSORS_INDEX = "model.safetensors.index.json"
DEEPSPEED_WEIGHTS_KEY = "module"
MODULATION_TERMS = 6

# Keys the released checkpoints legitimately omit. Upstream never computes token
# logits (it passes skip_logits=True), so the language-model head was dropped on
# export; this port keeps it because stock Qwen3-VL always builds one, and it is
# unused at inference.
EXPECTED_MISSING = ("vlm.lm_head.weight",)


@dataclass
class LoadReport:
    """Outcome of loading a checkpoint into a model.

    Attributes:
        loaded: Number of tensors copied into the model.
        missing: Model parameters the checkpoint did not provide.
        unexpected: Checkpoint tensors the model has no home for.
    """

    loaded: int = 0
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Describe the load in one line.

        Returns:
            A human-readable summary.
        """
        return f"loaded {self.loaded} tensors, {len(self.missing)} missing, {len(self.unexpected)} unexpected"


def remap_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize a released checkpoint's keys onto this implementation's names.

    The published names already match, so this only strips wrapper prefixes:
    ``module.`` from DeepSpeed checkpoints and ``model.`` from checkpoints saved at
    the policy level rather than the model level.

    Args:
        state_dict: Raw tensors from a checkpoint.

    Returns:
        A new mapping with wrapper prefixes removed.
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        name = key.removeprefix("module.")
        # A policy-level checkpoint nests the model under `model.`; the backbone's own
        # `vlm.model.` prefix must survive, so only strip a leading occurrence.
        if name.startswith("model.") and not name.startswith("model.visual"):
            candidate = name.removeprefix("model.")
            if candidate.startswith(("dit.", "vlm.", "action_", "state_", "t_", "sink")):
                name = candidate
        remapped[name] = value
    return remapped


def resolve_checkpoint(pretrained_name_or_path: str | Path, **download_kwargs: Any) -> Path:  # noqa: ANN401
    """Resolve a local path or a Hugging Face repo id to a local directory or file.

    Args:
        pretrained_name_or_path: Local file, local directory, or Hub repo id.
        **download_kwargs: Forwarded to ``huggingface_hub.snapshot_download``.

    Returns:
        A local path.
    """
    path = Path(pretrained_name_or_path)
    if path.exists():
        return path

    from huggingface_hub import snapshot_download  # noqa: PLC0415  # optional at import time

    logger.info("Downloading XR-1 checkpoint %s", pretrained_name_or_path)
    # snapshot_download is typed as also returning a dry-run listing; with dry_run
    # unset it always returns the snapshot path.
    downloaded = snapshot_download(repo_id=str(pretrained_name_or_path), **download_kwargs)
    return Path(cast("str", downloaded))


def load_state_dict(pretrained_name_or_path: str | Path, **download_kwargs: Any) -> dict[str, torch.Tensor]:  # noqa: ANN401
    """Load a released checkpoint into a flat, remapped state dict.

    Handles the three shapes the released checkpoints come in: a sharded
    safetensors directory, a single safetensors file, and a DeepSpeed-style
    ``.pt`` whose tensors sit under a ``module`` key.

    Args:
        pretrained_name_or_path: Local path or Hub repo id.
        **download_kwargs: Forwarded to the Hub downloader.

    Returns:
        Remapped tensors ready for :func:`load_pretrained_weights`.

    Raises:
        FileNotFoundError: If no recognizable weight file is present.
    """
    path = resolve_checkpoint(pretrained_name_or_path, **download_kwargs)

    if path.is_file():
        return remap_state_dict(_read_weight_file(path))

    index = path / SAFETENSORS_INDEX
    if index.exists():
        return remap_state_dict(_read_sharded_safetensors(index))

    for pattern in ("*.safetensors", "model_states.pt", "*.pt", "*.bin"):
        candidates = sorted(path.glob(pattern))
        if candidates:
            merged: dict[str, torch.Tensor] = {}
            for candidate in candidates:
                merged.update(_read_weight_file(candidate))
            return remap_state_dict(merged)

    msg = f"No safetensors or .pt weights found in {path}"
    raise FileNotFoundError(msg)


def _read_weight_file(path: Path) -> dict[str, torch.Tensor]:
    """Read a single safetensors or torch checkpoint file.

    Args:
        path: File to read.

    Returns:
        The tensors it contains.
    """
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file  # noqa: PLC0415  # optional dependency

        return load_file(str(path))

    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(checkpoint, dict) and DEEPSPEED_WEIGHTS_KEY in checkpoint:
        return checkpoint[DEEPSPEED_WEIGHTS_KEY]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _read_sharded_safetensors(index_file: Path) -> dict[str, torch.Tensor]:
    """Read every shard referenced by a safetensors index.

    Args:
        index_file: Path to ``model.safetensors.index.json``.

    Returns:
        The merged tensors.
    """
    import json  # noqa: PLC0415  # only needed on this path

    from safetensors.torch import load_file  # noqa: PLC0415  # optional dependency

    with index_file.open(encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    merged: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        merged.update(load_file(str(index_file.parent / shard)))
    return merged


def infer_config_overrides(state_dict: Mapping[str, torch.Tensor]) -> dict[str, int]:
    """Read the architecture sizes back out of a checkpoint.

    The released weights are sized for a 60-dimensional dual-arm action space and a
    36-layer, 1024-wide action expert. Passing these to
    :class:`~physicalai.policies.xr1.config.XR1Config` turns an unreadable shape
    error into a configuration that simply matches.

    Args:
        state_dict: Remapped checkpoint tensors.

    Returns:
        Config field names mapped to the values the checkpoint implies. Only fields
        the checkpoint actually determines are included.
    """
    overrides: dict[str, int] = {}

    layers = {int(key.split(".")[2]) for key in state_dict if key.startswith("dit.layers.")}
    if layers:
        overrides["dit_num_layers"] = max(layers) + 1

    sink = state_dict.get("sink.weight")
    if sink is not None:
        overrides["dit_hidden_size"] = int(sink.shape[-1])

    action_in = state_dict.get("action_projector.layers.0.weight")
    if action_in is not None:
        overrides["max_action_dim"] = int(action_in.shape[-1])

    state_in = state_dict.get("state_projector.layers.0.weight")
    if state_in is not None:
        overrides["max_state_dim"] = int(state_in.shape[-1])

    # Any layer will do: a partially merged shard need not contain layer 0.
    q_norm_keys = sorted(key for key in state_dict if key.endswith("attn.q_norm.weight"))
    if q_norm_keys:
        overrides["dit_head_dim"] = int(state_dict[q_norm_keys[0]].shape[0])

    return overrides


def load_pretrained_weights(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
) -> LoadReport:
    """Copy checkpoint tensors into a model and report what happened.

    Args:
        model: The model to populate, normally an
            :class:`~physicalai.policies.xr1.vla.XR1Model`.
        state_dict: Remapped checkpoint tensors.
        strict: Raise when the checkpoint leaves parameters uninitialized beyond the
            known-omitted ones, or carries tensors the model cannot use.

    Returns:
        A report naming missing and unexpected keys.

    Raises:
        RuntimeError: If ``strict`` and the checkpoint does not fit the model.
    """
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in incompatible.missing_keys if key not in EXPECTED_MISSING]
    unexpected = list(incompatible.unexpected_keys)
    report = LoadReport(
        loaded=len(state_dict) - len(unexpected),
        missing=missing,
        unexpected=unexpected,
    )

    if strict and (missing or unexpected):
        msg = (
            f"Checkpoint does not match the configured model: {report.summary()}. "
            f"First missing: {missing[:3]}; first unexpected: {unexpected[:3]}. "
            "Pass the values from infer_config_overrides() to XR1Config so the sizes agree."
        )
        raise RuntimeError(msg)

    logger.info("XR-1 checkpoint: %s", report.summary())
    return report
