# Copyright (C) 2026 Xiaomi Corporation.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities for loading pretrained XR0 weights (e.g. the LIBERO checkpoint).

The source repository (``Xiaomi-Robotics-0``) publishes its checkpoints -- such
as ``XiaomiRobotics/Xiaomi-Robotics-0-LIBERO`` -- as the state dict of the
top-level ``XR0`` module, whose submodules are::

    vlm.*                 dit.*                 t_embedder.*
    state_projector.*     action_projector.*    t_projector.*
    action_output_layer.* sink.*                rotary_emb.*
"""

from __future__ import annotations

import json
import logging
import pickle  # noqa: S403  # only referenced for pickle.UnpicklingError in except; never used to deserialize
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

# Top-level submodules owned by ``XR0FlowModel`` (nested under ``flow.`` in the
# framework model but flat on the source ``XR0`` module).
_FLOW_SUBMODULES = frozenset(
    {
        "dit",
        "state_projector",
        "action_projector",
        "action_output_layer",
        "t_embedder",
        "t_projector",
        "sink",
    },
)

# Non-persistent / recomputed buffers that must not be force-loaded.
_DROP_PREFIXES = ("saved_causal_mask", "rotary_emb.")

# The VLM ties ``lm_head`` to the token embeddings; the checkpoint omits it.
_VLM_LM_HEAD_KEY = "vlm.lm_head.weight"
_VLM_EMBED_TOKENS_KEY = "vlm.model.language_model.embed_tokens.weight"

# Wrapper prefixes added by the training runner / DeepSpeed.
_WRAPPER_PREFIXES = ("module.", "model.")

# ``hf_hub`` download kwargs we forward from caller ``**kwargs``.
_HUB_KWARGS = frozenset(
    {
        "cache_dir",
        "force_download",
        "resume_download",
        "proxies",
        "token",
        "revision",
        "local_files_only",
    },
)


def remap_xr0_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap source ``XR0`` state-dict keys into the ``XR0Model`` namespace.

    Strips the training-runner / DeepSpeed wrapper prefixes (``module.`` /
    ``model.``), drops recomputed buffers, and nests the DiT action-expert
    submodules under ``flow.`` to match
    :class:`~physicalai.policies.xr0.model.XR0Model`.

    Args:
        state_dict: Raw source state dict (already unwrapped from any
            ``{"module": ...}`` / ``{"state_dict": ...}`` container).

    Returns:
        A new state dict keyed for ``XR0Model.load_state_dict``.
    """
    remapped: dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        key = raw_key
        # Strip a single leading wrapper prefix (never touches ``vlm.model.*``,
        # which starts with ``vlm.`` rather than ``model.``).
        for prefix in _WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break

        if any(key == p or key.startswith(p) for p in _DROP_PREFIXES):
            continue

        head = key.split(".", 1)[0]
        if head in _FLOW_SUBMODULES:
            key = f"flow.{key}"

        remapped[key] = value

    # The VLM ties its ``lm_head`` to the token embeddings, so the checkpoint
    # omits ``lm_head.weight``. Recreate it so the load is clean (the head is
    # unused by the DiT action path but is part of the module's state dict).
    if _VLM_LM_HEAD_KEY not in remapped and _VLM_EMBED_TOKENS_KEY in remapped:
        remapped[_VLM_LM_HEAD_KEY] = remapped[_VLM_EMBED_TOKENS_KEY].clone()

    return remapped


def _unwrap_container(obj: object) -> Mapping[str, torch.Tensor]:
    """Pull the tensor mapping out of a ``torch.load`` result.

    Handles the DeepSpeed ``{"module": {...}}`` and Lightning
    ``{"state_dict": {...}}`` wrappers.

    Returns:
        The underlying ``{name: tensor}`` mapping.

    Raises:
        TypeError: If a tensor mapping cannot be located.
    """
    if isinstance(obj, dict):
        for wrapper_key in ("module", "state_dict"):
            inner = obj.get(wrapper_key)
            if isinstance(inner, dict):
                return inner
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    msg = "Could not locate a tensor state dict in the checkpoint object"
    raise TypeError(msg)


def _torch_load_safe(path: Path) -> object:
    """Load a ``.pt`` / ``.ckpt`` / ``.bin`` checkpoint without executing pickle.

    Uses ``weights_only=True`` so only tensors and a safe allowlist of primitive
    containers are unpickled -- arbitrary code embedded in a malicious checkpoint
    is never executed. Legacy checkpoints that store non-tensor Python objects
    cannot be loaded this way and must be converted to ``.safetensors`` first.

    Args:
        path: Local checkpoint file to load.

    Returns:
        The loaded object (typically a state dict or a wrapper dict).

    Raises:
        ValueError: If the checkpoint cannot be loaded safely because it relies
            on arbitrary pickle contents; convert it to ``.safetensors`` instead.
    """
    try:
        # reason: weights_only=True restricts unpickling to tensors + safe primitives; no arbitrary code runs.
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, AttributeError, ImportError, EOFError) as exc:
        msg = (
            f"Refusing to load {path} with unsafe pickle deserialization. The checkpoint stores "
            "non-tensor Python objects and cannot be loaded with weights_only=True. Convert it "
            "to .safetensors before loading."
        )
        raise ValueError(msg) from exc


def _load_sharded_safetensors(index_file: Path) -> dict[str, torch.Tensor]:
    """Load a sharded ``*.safetensors`` checkpoint from its index json.

    Returns:
        The merged state dict across all shards.
    """
    with index_file.open(encoding="utf-8") as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]
    state_dict: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        state_dict.update(load_file(str(index_file.parent / shard)))
    return state_dict


def _load_state_dict_from_dir(path: Path) -> Mapping[str, torch.Tensor]:
    """Load a raw state dict from a checkpoint *directory*.

    Probes the known checkpoint layouts (DeepSpeed, sharded/single safetensors,
    sharded/single ``.bin``/``.pt``) in priority order.

    Returns:
        The raw source state dict.

    Raises:
        FileNotFoundError: If no recognized weights file is found.
    """
    deepspeed = path / "last.ckpt" / "checkpoint" / "mp_rank_00_model_states.pt"
    if deepspeed.is_file():
        return _unwrap_container(_torch_load_safe(deepspeed))

    safetensors_index = path / "model.safetensors.index.json"
    if safetensors_index.is_file():
        return _load_sharded_safetensors(safetensors_index)

    single_safetensors = path / "model.safetensors"
    if single_safetensors.is_file():
        return load_file(str(single_safetensors))

    bin_index = path / "pytorch_model.bin.index.json"
    if bin_index.is_file():
        with bin_index.open(encoding="utf-8") as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
        state_dict: dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            state_dict.update(_unwrap_container(_torch_load_safe(path / shard)))
        return state_dict

    for candidate in ("pytorch_model.bin", "pytorch_model.pt", "xr0_pretrained.pt"):
        file = path / candidate
        if file.is_file():
            return _unwrap_container(_torch_load_safe(file))

    msg = f"No recognized XR0 weights file found under {path}"
    raise FileNotFoundError(msg)


def _load_raw_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    """Load a raw (un-remapped) state dict from a file or checkpoint directory.

    Supports single ``.safetensors`` / ``.pt`` / ``.ckpt`` / ``.bin`` files, a
    sharded HF snapshot directory, and a DeepSpeed checkpoint directory.

    Returns:
        The raw source state dict.
    """
    if path.is_file():
        if path.suffix == ".safetensors":
            return load_file(str(path))
        return _unwrap_container(_torch_load_safe(path))
    return _load_state_dict_from_dir(path)


def resolve_pretrained_path(pretrained_name_or_path: str | Path, **kwargs: object) -> Path:
    """Resolve a local path for an XR0 checkpoint, downloading it if needed.

    A local file or directory is returned as-is; otherwise the argument is
    treated as a HuggingFace repo id and the weight files are downloaded via
    ``huggingface_hub.snapshot_download``.

    Args:
        pretrained_name_or_path: Local file/dir path or HuggingFace repo id.
        **kwargs: Optional ``huggingface_hub`` download options
            (``cache_dir``, ``revision``, ``token``, ...).

    Returns:
        A local :class:`~pathlib.Path` to the checkpoint file or directory.
    """
    path = Path(pretrained_name_or_path)
    if path.exists():
        return path

    from huggingface_hub import snapshot_download  # noqa: PLC0415

    hub_kwargs = {k: v for k, v in kwargs.items() if k in _HUB_KWARGS}
    if "revision" not in hub_kwargs:
        logger.warning(
            "Downloading '%s' without a pinned 'revision'; resolving to HEAD is not reproducible. "
            "Pass revision=<commit-sha> for a pinned, reproducible download.",
            pretrained_name_or_path,
        )
    # Only allow safe (non-pickle) formats to be downloaded. ``.bin`` / ``.pt``
    # checkpoints are supported solely for pre-existing local paths, where the
    # user already controls the file. See ``_load_raw_state_dict``.
    local = snapshot_download(  # pyrefly: ignore[no-matching-overload]
        repo_id=str(pretrained_name_or_path),
        allow_patterns=[
            "*.safetensors",
            "*.safetensors.index.json",
            "config.json",
            "preprocessor_config.json",
        ],
        **hub_kwargs,  # type: ignore[call-overload]
    )  # nosec B615
    return Path(local)


def load_xr0_pretrained_weights(pretrained_name_or_path: str | Path, **kwargs: object) -> dict[str, torch.Tensor]:
    """Load and remap pretrained XR0 weights ready for ``XR0Model``.

    Resolves (downloading if necessary) the checkpoint, loads the raw source
    state dict, and remaps it into the framework ``XR0Model`` key namespace.

    Args:
        pretrained_name_or_path: Local file/dir path or HuggingFace repo id
            (e.g. ``"XiaomiRobotics/Xiaomi-Robotics-0-LIBERO"``).
        **kwargs: Optional ``huggingface_hub`` download options.

    Returns:
        A remapped state dict suitable for
        ``XR0Model.load_state_dict(..., strict=False)``.
    """
    local_path = resolve_pretrained_path(pretrained_name_or_path, **kwargs)
    raw = _load_raw_state_dict(local_path)
    return remap_xr0_state_dict(raw)


# Dimensions with ``std`` below this threshold are inactive/padding -- matching
# the source ``get_action_mask`` rule (``std > 1e-5``).
_ACTION_ACTIVE_STD = 1e-5


def _load_json_artifact(pretrained_name_or_path: str | Path, filename: str, **kwargs: object) -> dict | None:
    """Read a json artifact from a local checkpoint or a HuggingFace repo.

    Returns:
        The parsed json object, or ``None`` if the artifact is unavailable.
    """
    path = Path(pretrained_name_or_path)
    candidate: Path | None = None
    if path.is_dir():
        candidate = path / filename
    elif path.is_file():
        candidate = path.parent / filename

    if candidate is not None:
        if not candidate.is_file():
            return None
        with candidate.open(encoding="utf-8") as f:
            return json.load(f)

    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from huggingface_hub.utils import EntryNotFoundError, HFValidationError  # noqa: PLC0415

    hub_kwargs = {k: v for k, v in kwargs.items() if k in _HUB_KWARGS}
    if "revision" not in hub_kwargs:
        logger.warning(
            "Downloading '%s' without a pinned 'revision'; resolving to HEAD is not reproducible. "
            "Pass revision=<commit-sha> for a pinned, reproducible download.",
            pretrained_name_or_path,
        )
    try:
        local = hf_hub_download(str(pretrained_name_or_path), filename, **hub_kwargs)  # type: ignore[call-overload] # nosec B615
    except (EntryNotFoundError, HFValidationError, OSError):
        return None
    with Path(local).open(encoding="utf-8") as f:
        return json.load(f)


def extract_xr0_dataset_stats(
    pretrained_name_or_path: str | Path,
    robot_type: str | None = None,
    **kwargs: object,
) -> dict[str, dict[str, object]] | None:
    """Extract action-normalization stats from a pretrained XR0 checkpoint.

    The source publishes per-robot action ``mean`` / ``std`` in the processor's
    ``preprocessor_config.json`` under ``action_config``. The stats are stored
    per action-timestep but are time-invariant, so they reduce to a single
    per-dimension vector. Only the leading dimensions with ``std > 1e-5`` are
    active (mirroring the source ``get_action_mask``); trailing padding
    dimensions are dropped so the postprocessor emits the true action size
    (e.g. 7 for LIBERO).

    Args:
        pretrained_name_or_path: Local checkpoint dir/file or HuggingFace repo id.
        robot_type: Which ``action_config`` entry to use. Defaults to the sole
            entry (or the first, when several are present).
        **kwargs: Optional ``huggingface_hub`` download options.

    Returns:
        A ``dataset_stats`` dict ``{"action": {...}}`` consumable by
        :func:`~physicalai.policies.xr0.preprocessor.make_xr0_preprocessors`,
        or ``None`` if no ``action_config`` is available.

    Raises:
        KeyError: If ``robot_type`` is not present in the checkpoint's
            ``action_config``.
    """
    import numpy as np  # noqa: PLC0415

    from physicalai.data.observation import ACTION  # noqa: PLC0415

    preproc = _load_json_artifact(pretrained_name_or_path, "preprocessor_config.json", **kwargs)
    action_config = preproc.get("action_config") if isinstance(preproc, dict) else None
    if not action_config:
        return None

    if robot_type is None:
        robot_type = next(iter(action_config))
    elif robot_type not in action_config:
        msg = f"robot_type '{robot_type}' not in action_config; available: {list(action_config)}"
        raise KeyError(msg)

    entry = action_config[robot_type]
    mean = np.asarray(entry["mean"], dtype=np.float64)
    std = np.asarray(entry["std"], dtype=np.float64)

    # Collapse the (time-invariant) per-timestep dimension to a per-dim vector.
    if mean.ndim > 1:
        mean = mean.reshape(-1, mean.shape[-1])[0]
        std = std.reshape(-1, std.shape[-1])[0]

    active = np.nonzero(std > _ACTION_ACTIVE_STD)[0]
    real_dim = int(active[-1]) + 1 if active.size else int(mean.shape[-1])
    mean = mean[:real_dim]
    std = std[:real_dim]

    return {
        ACTION: {
            "name": ACTION,
            "shape": (real_dim,),
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
    }
