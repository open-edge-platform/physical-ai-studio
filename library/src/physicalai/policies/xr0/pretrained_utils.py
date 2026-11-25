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

# Only referenced for pickle.UnpicklingError in an except clause; never used to deserialize.
import pickle  # noqa: S403  # nosec B403
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from safetensors.torch import load_file

from physicalai.data.constants import ACTION, STATE
from physicalai.devices.utils import get_device

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
    chunk_size: int | None = None,
    **kwargs: object,
) -> dict[str, dict[str, object]] | None:
    """Extract action-normalization stats from a pretrained XR0 checkpoint.

    Load per-timestep action mean/std from ``preprocessor_config.json``.
    Trim dimensions after the last with ``std > 1e-5``. If the requested
    ``chunk_size`` differs, repeat rows only when the stats are time-invariant.

    Args:
        pretrained_name_or_path: Local checkpoint dir/file or HuggingFace repo id.
        robot_type: Which ``action_config`` entry to use. Defaults to the sole
            entry (or the first, when several are present).
        chunk_size: Expected number of action timesteps. ``None`` keeps the
            published length.
        **kwargs: Optional ``huggingface_hub`` download options.

    Returns:
        A ``dataset_stats`` dict ``{"action": {...}}`` whose ``mean`` / ``std``
        are per-timestep ``(T, real_dim)`` arrays (``shape`` stays the action
        width ``(real_dim,)``), consumable by
        :func:`~physicalai.policies.xr0.preprocessor.make_xr0_preprocessors`,
        or ``None`` if no ``action_config`` is available.

    Raises:
        KeyError: If ``robot_type`` is not present in the checkpoint's
            ``action_config``.
        ValueError: If the published stats vary over time but their length does
            not match ``chunk_size``.
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
    mean = np.atleast_2d(np.asarray(entry["mean"], dtype=np.float64))
    std = np.atleast_2d(np.asarray(entry["std"], dtype=np.float64))

    if chunk_size is not None and mean.shape[0] != chunk_size:
        time_invariant = bool((mean == mean[0]).all() and (std == std[0]).all())
        if not time_invariant:
            msg = (
                f"checkpoint publishes {mean.shape[0]} timesteps of action stats that vary over time, "
                f"but the policy predicts {chunk_size} timesteps"
            )
            raise ValueError(msg)
        mean = np.repeat(mean[:1], chunk_size, axis=0)
        std = np.repeat(std[:1], chunk_size, axis=0)

    active = np.nonzero(std.max(axis=0) > _ACTION_ACTIVE_STD)[0]
    real_dim = int(active[-1]) + 1 if active.size else int(mean.shape[-1])
    mean = mean[:, :real_dim]
    std = std[:, :real_dim]

    return {
        ACTION: {
            "name": ACTION,
            # ``shape`` describes the action feature itself (its width); the
            # mean/std arrays carry the extra per-timestep axis.
            "shape": (real_dim,),
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
    }


def _feature_tensor(value: Any) -> torch.Tensor:  # noqa: ANN401
    """Coerce an observation field (tensor or single-entry dict) to a tensor.

    Returns:
        The tensor held by ``value``.

    Raises:
        TypeError: If no tensor can be recovered from ``value``.
    """
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for sub in value.values():
            if isinstance(sub, torch.Tensor):
                return sub
    msg = f"expected a tensor (or dict containing one), got {type(value)!r}"
    raise TypeError(msg)


_TEMPORAL_STATE_NDIM = 3
_BATCHED_ACTION_NDIM = 2


def compute_action_chunk_stats(  # noqa: PLR0914
    datamodule: Any,  # noqa: ANN401
    *,
    chunk_size: int,
    action_dim: int,
    max_action_dim: int = 32,
    action_mode: Literal["absolute", "delta"] = "absolute",
    max_batches: int | None = None,
    subset_size: int | None = None,
    setup_stage: str | None = "fit",
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-timestep action mean/std over a training dataset.

    Reads the train dataloader once, including padded episode endings. Computes
    mean and std for each action timestep: raw actions in ``"absolute"`` mode,
    or actions minus the current state in ``"delta"`` mode. Uses float64 while
    accumulating on the selected device, then returns CPU float32 stats padded
    with mean 0 and std 1.

    Chunk the datamodule first with
    :func:`physicalai.train.utils.reformat_dataset_to_match_policy`.

    Args:
        datamodule: A Lightning datamodule exposing ``train_dataloader()`` (and an
            optional ``setup`` method) yielding batched observations with ``state``
            and ``action`` fields.
        chunk_size: Number of action steps per chunk (the temporal dimension).
        action_dim: True (unpadded) action dimension of the dataset.
        max_action_dim: Padded action dimension of the returned buffers.
        action_mode: ``"absolute"`` (default) accumulates the raw action;
            ``"delta"`` subtracts the current state from the action target.
        max_batches: Optional cap on the number of batches to consume (for a quick
            estimate). ``None`` consumes the full train set.
        subset_size: Optional number of samples to use from the start of the
            train dataloader. ``None`` uses all samples (subject to ``max_batches``).
        setup_stage: Stage passed to ``datamodule.setup(...)`` before iterating.
            Pass ``None`` to skip setup (e.g. when already set up).
        device: Compute device. Defaults to CUDA or XPU when available, otherwise CPU.

    Returns:
        A ``(mean, std)`` tuple of ``(chunk_size, max_action_dim)`` float32 tensors.

    Raises:
        ValueError: If ``subset_size`` is not positive, the dataset yields no
            samples, or an action chunk has the wrong temporal dimension.
    """
    if subset_size is not None and subset_size <= 0:
        msg = "subset_size must be positive"
        raise ValueError(msg)

    if setup_stage is not None and hasattr(datamodule, "setup"):
        datamodule.setup(setup_stage)

    compute_device = get_device() if device is None else torch.device(device)
    sum_1 = torch.zeros(chunk_size, action_dim, dtype=torch.float64, device=compute_device)
    sum_2 = torch.zeros(chunk_size, action_dim, dtype=torch.float64, device=compute_device)
    count = 0

    for index, batch in enumerate(datamodule.train_dataloader()):
        if max_batches is not None and index >= max_batches:
            break

        action_field = batch.action if hasattr(batch, "action") else batch[ACTION]
        action = _feature_tensor(action_field)
        if subset_size is not None:
            action = action[: subset_size - count]
        action = action.to(device=compute_device, dtype=torch.float64)
        if action.ndim == _BATCHED_ACTION_NDIM:  # (B, D) -> (B, 1, D)
            action = action.unsqueeze(1)
        if action.shape[1] != chunk_size:
            msg = (
                f"action chunk has temporal dim {action.shape[1]}, expected chunk_size={chunk_size}; "
                "call reformat_dataset_to_match_policy(policy, datamodule) first"
            )
            raise ValueError(msg)
        target = action[..., :action_dim]

        if action_mode == "delta":
            state_field = batch.state if hasattr(batch, "state") else batch[STATE]
            state = _feature_tensor(state_field)[: action.shape[0]].to(device=compute_device, dtype=torch.float64)
            if state.ndim == _TEMPORAL_STATE_NDIM:  # (B, T, D) -> current (last) frame
                state = state[:, -1, :]
            # Do not modify the dataset's action tensor through ``target``.
            target = target.clone()
            overlap = min(action_dim, state.shape[-1])
            target[..., :overlap] -= state[..., :overlap].unsqueeze(1)

        sum_1 += target.sum(dim=0)
        sum_2 += (target * target).sum(dim=0)
        count += target.shape[0]
        if subset_size is not None and count >= subset_size:
            break

    if count == 0:
        msg = "no samples found while computing action chunk stats"
        raise ValueError(msg)

    mean = sum_1 / count
    var = (sum_2 / count - mean * mean).clamp_min(0.0)
    std = var.sqrt()

    mean_full = torch.zeros(chunk_size, max_action_dim, dtype=torch.float32)
    std_full = torch.ones(chunk_size, max_action_dim, dtype=torch.float32)
    width = min(action_dim, max_action_dim)
    mean_full[:, :width] = mean[:, :width].to(device="cpu", dtype=torch.float32)
    std_full[:, :width] = std[:, :width].to(device="cpu", dtype=torch.float32)
    return mean_full, std_full
