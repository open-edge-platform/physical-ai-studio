# Copyright 2025 2toINF (https://github.com/2toINF) and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Loading XVLA checkpoints published in the LeRobot layout.

A published XVLA checkpoint holds:

- ``config.json`` -- the architecture, the action space and the training hyperparameters;
- ``model.safetensors`` -- the weights, keyed ``model.*`` because LeRobot nests the network
  one level deeper than Studio does;
- optionally ``policy_preprocessor.json`` / ``policy_postprocessor.json`` plus their
  safetensors state files, carrying the state and action normalization statistics.

Two layout differences are reconciled here. The ``model.`` prefix is stripped, and
checkpoints saved against the old vendored (Microsoft remote-code) Florence-2 module tree
are remapped onto the native ``transformers.models.florence2`` layout -- the same kind of
remapping :mod:`physicalai.policies.pi05.pretrained_utils` does for openpi weights.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from physicalai.data.observation import ACTION

from .config import XVLAConfig

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_HUB_KWARGS = frozenset({
    "cache_dir",
    "force_download",
    "resume_download",
    "proxies",
    "token",
    "revision",
    "local_files_only",
})

_PROCESSOR_FILES = ("policy_preprocessor.json", "policy_postprocessor.json")

_LEROBOT_ONLY_KEYS = frozenset({
    "type",
    "input_features",
    "output_features",
    "normalization_mapping",
    "device",
    "use_amp",
    "use_peft",
    "push_to_hub",
    "repo_id",
    "private",
    "tags",
    "license",
    "pretrained_path",
    # Padding side/strategy are always "right"/"max_length" in Studio -- padding must stay
    # fixed-length (see XVLAPreprocessor._tokenize), and right-padding is the only side that
    # keeps a prompt's own tokens at the same positions regardless of padded length.
    # `tokenizer_max_length` is NOT dropped here: unlike these two, its value is load-bearing
    # (see extract_tokenizer_max_length) and must come from the checkpoint's own manifest.
    "tokenizer_padding_side",
    "pad_language_to",
    # Upstream soft-prompt LR warmup, which Studio folds into its scheduler.
    "optimizer_soft_prompt_warmup_lr_scale",
})

_STATE_DICT_PREFIX = "model."
_VLM_PREFIX = "vlm."


@dataclass(frozen=True)
class CheckpointFiles:
    """Resolved paths of an XVLA checkpoint.

    Attributes:
        config_file: Path to ``config.json``.
        weights_file: Path to ``model.safetensors``.
        processor_files: Published processor manifests, if any.
        processor_dir: Directory holding the processors' state files.
    """

    config_file: Path
    weights_file: Path
    processor_files: tuple[Path, ...] = ()
    processor_dir: Path | None = None


def resolve_checkpoint(pretrained_name_or_path: str | Path, **kwargs: Any) -> CheckpointFiles:  # noqa: ANN401
    """Resolve a checkpoint's files from a local directory or the HuggingFace Hub.

    Args:
        pretrained_name_or_path: HuggingFace repo id or local directory.
        **kwargs: Forwarded to ``huggingface_hub.hf_hub_download``; only the recognized
            download options are passed through.

    Returns:
        The resolved :class:`CheckpointFiles`.
    """
    path = Path(pretrained_name_or_path)
    if path.is_dir():
        processor_files = tuple(path / name for name in _PROCESSOR_FILES if (path / name).exists())
        return CheckpointFiles(
            config_file=path / "config.json",
            weights_file=path / "model.safetensors",
            processor_files=processor_files,
            processor_dir=path if processor_files else None,
        )

    hub_kwargs = {k: v for k, v in kwargs.items() if k in _HUB_KWARGS}
    repo_id = str(pretrained_name_or_path)
    config_file = Path(hf_hub_download(repo_id, "config.json", **hub_kwargs))  # nosec B615  # pyrefly: ignore[bad-argument-type]
    weights_file = Path(hf_hub_download(repo_id, "model.safetensors", **hub_kwargs))  # nosec B615  # pyrefly: ignore[bad-argument-type]

    downloaded: list[Path] = []
    processor_dir: Path | None = None
    for name in _PROCESSOR_FILES:
        try:
            processor_file = Path(hf_hub_download(repo_id, name, **hub_kwargs))  # nosec B615  # pyrefly: ignore[bad-argument-type]
        except Exception:  # noqa: BLE001
            logger.debug("No %s published for %s.", name, repo_id)
            continue
        downloaded.append(processor_file)
        processor_dir = processor_file.parent
        with processor_file.open(encoding="utf-8") as f:
            manifest = json.load(f)
        for step in manifest.get("steps", []):
            state_file = step.get("state_file")
            if state_file:
                hf_hub_download(repo_id, state_file, **hub_kwargs)  # nosec B615

    return CheckpointFiles(
        config_file=config_file,
        weights_file=weights_file,
        processor_files=tuple(downloaded),
        processor_dir=processor_dir,
    )


def load_config(config_file: Path, overrides: dict[str, Any] | None = None) -> XVLAConfig:
    """Parse a checkpoint's ``config.json`` into an :class:`XVLAConfig`.

    LeRobot-only keys (feature specs, device, hub metadata) are dropped; everything else is
    coerced by the dataclass. Caller overrides are applied last, so training-time settings
    always win over the published values.

    ``num_image_views`` is always recomputed as ``max(published value, declared visual
    features + empty_cameras)`` -- the same formula upstream's own ``validate_features()``
    applies, unconditionally, every time a config is loaded. This matters because a
    checkpoint's ``input_features`` can itself already include an ``empty_camera_N`` entry
    baked in from a *previous* validation pass (training scripts commonly re-save a config
    after running ``validate_features()`` once), so reapplying the formula on load adds
    ``empty_cameras`` again on top of that already-padded count -- e.g. 2 real cameras + 1
    baked-in empty one declared, plus ``empty_cameras=1`` published, resolves to 4 views,
    not 3. Only reproducing this exactly (rather than treating a published
    ``num_image_views`` as final) keeps the camera count -- and therefore every downstream
    sequence position -- aligned with what a checkpoint actually trained with.

    Args:
        config_file: Path to the checkpoint's ``config.json``.
        overrides: Values that take precedence over the file's contents. ``None`` entries
            are ignored, so callers can pass "unset" through.

    Returns:
        The resolved configuration.
    """
    raw = _read_json(config_file)

    config_kwargs = {k: v for k, v in raw.items() if k not in _LEROBOT_ONLY_KEYS}
    num_cameras = _count_visual_features(raw.get("input_features") or {})
    if num_cameras:
        from_features = num_cameras + int(raw.get("empty_cameras", 0) or 0)
        config_kwargs["num_image_views"] = max(int(config_kwargs.get("num_image_views") or 0), from_features)

    config_kwargs.update({key: value for key, value in (overrides or {}).items() if value is not None})

    # strict=False: tolerate keys published by newer or older LeRobot configs.
    return XVLAConfig.from_dict(config_kwargs, strict=False)


def read_action_dim(config_file: Path) -> int | None:
    """Read the action width a checkpoint was trained against.

    Args:
        config_file: Path to the checkpoint's ``config.json``.

    Returns:
        The action feature's width, or ``None`` when the checkpoint declares none.
    """
    raw = _read_json(config_file)
    shape = ((raw.get("output_features") or {}).get(ACTION) or {}).get("shape")
    return int(shape[0]) if shape else None


def extract_dataset_stats(
    processor_files: tuple[Path, ...] | list[Path],
    processor_dir: Path | None,
) -> dict[str, dict[str, Any]]:
    """Read the state and action normalization statistics out of a checkpoint's processors.

    Args:
        processor_files: Published processor manifests.
        processor_dir: Directory holding the referenced state files.

    Returns:
        Statistics keyed by feature name, or an empty dict when the checkpoint publishes
        none (which is the norm for XVLA, whose processors are identity).
    """
    if processor_dir is None:
        return {}

    stats: dict[str, dict[str, Any]] = {}
    for processor_file in processor_files:
        if not Path(processor_file).exists():
            continue
        for step in _read_json(Path(processor_file)).get("steps", []):
            state_file = step.get("state_file")
            if "normalizer" not in step.get("registry_name", "").lower() or not state_file:
                continue

            state_path = Path(processor_dir) / state_file
            if not state_path.exists():
                logger.warning("Normalizer state file not found: %s", state_path)
                continue

            features = step.get("config", {}).get("features", {})
            for flat_key, tensor in load_file(str(state_path)).items():
                feature_name, stat_name = flat_key.rsplit(".", 1)
                entry = stats.setdefault(
                    feature_name,
                    {"name": feature_name, "shape": tuple(features.get(feature_name, {}).get("shape", ()))},
                )
                entry[stat_name] = tensor.cpu().tolist()

    return stats


def extract_domain_id(processor_files: tuple[Path, ...] | list[Path]) -> int | None:
    """Read the domain index a checkpoint was published with.

    Upstream adds ``domain_id`` to the batch via a dedicated preprocessor step
    (``xvla_add_domain_id``) rather than storing it in ``config.json``, so a checkpoint
    trained on a domain other than 0 -- for example a single-embodiment finetune sharing a
    multi-domain base checkpoint's ``num_domains`` -- would otherwise silently run with the
    wrong domain's action decoder and soft prompts.

    Args:
        processor_files: Published processor manifests.

    Returns:
        The domain index, or ``None`` when no manifest declares one.
    """
    for processor_file in processor_files:
        if not Path(processor_file).exists():
            continue
        for step in _read_json(Path(processor_file)).get("steps", []):
            if "domain_id" in step.get("registry_name", "").lower():
                domain_id = step.get("config", {}).get("domain_id")
                if domain_id is not None:
                    return int(domain_id)
    return None


def extract_tokenizer_max_length(processor_files: tuple[Path, ...] | list[Path]) -> int | None:
    """Read the fixed prompt length a checkpoint's tokenizer step actually pads to.

    ``config.json``'s own ``tokenizer_max_length`` field is not reliably the value the
    published processor pipeline used -- it can be a stale default left over from an
    earlier training configuration -- while the ``tokenizer_processor`` step in
    ``policy_preprocessor.json`` records the exact length every training and evaluation
    prompt was actually padded to. Because the transformer's positional embedding is a
    learned, absolute (not relative or rotary) table, every token after the prompt sits at
    whatever index that padded length pushes it to; a mismatch here silently feeds the
    model camera and soft-prompt tokens at positions it was never trained to interpret
    there, even though the prompt's own tokens look identical either way.

    Args:
        processor_files: Published processor manifests.

    Returns:
        The tokenizer's padded length, or ``None`` when no manifest declares one.
    """
    for processor_file in processor_files:
        if not Path(processor_file).exists():
            continue
        for step in _read_json(Path(processor_file)).get("steps", []):
            if "tokenizer" in step.get("registry_name", "").lower():
                max_length = step.get("config", {}).get("max_length")
                if max_length is not None:
                    return int(max_length)
    return None


def detect_normalization_mode(processor_files: tuple[Path, ...] | list[Path]) -> str | None:
    """Detect the normalization mode a checkpoint was published with.

    Args:
        processor_files: Published processor manifests.

    Returns:
        ``"QUANTILES"``, ``"MEAN_STD"``, ``"IDENTITY"``, or ``None`` when the manifests
        disagree or declare nothing.
    """
    modes = {
        value
        for processor_file in processor_files
        if Path(processor_file).exists()
        for step in _read_json(Path(processor_file)).get("steps", [])
        for value in (step.get("config", {}).get("norm_map") or {}).values()
    }
    non_identity = modes - {"IDENTITY"}
    if non_identity == {"QUANTILES"}:
        return "QUANTILES"
    if non_identity == {"MEAN_STD"}:
        return "MEAN_STD"
    if modes == {"IDENTITY"}:
        return "IDENTITY"
    return None


def is_vendored_florence_state_dict(state_dict: dict[str, Any], prefix: str = _VLM_PREFIX) -> bool:
    """Detect a checkpoint saved with the old vendored Florence-2 module layout.

    Args:
        state_dict: The checkpoint's state dict, already stripped of its ``model.`` prefix.
        prefix: Module path of the VLM inside the model.

    Returns:
        ``True`` if the state dict carries the vendored layout's signature keys.
    """
    return f"{prefix}image_projection" in state_dict or any(
        key.startswith(f"{prefix}language_model.model.") for key in state_dict
    )


def remap_vendored_florence_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefix: str = _VLM_PREFIX,
) -> dict[str, torch.Tensor]:
    """Remap a state dict from the vendored Florence-2 layout to the native one.

    Only keys under ``prefix`` are rewritten; everything else passes through unchanged.

    Args:
        state_dict: The checkpoint's state dict, already stripped of its ``model.`` prefix.
        prefix: Module path of the VLM inside the model.

    Returns:
        The remapped state dict.
    """
    vision = re.escape(prefix) + r"vision_tower\."
    block = vision + r"blocks\.(\d+)\.(\d+)\.(spatial_block|channel_block)\."
    new_block = prefix + r"vision_tower.blocks.\1.\2.\3."
    rules: list[tuple[str, str]] = [
        # DaViT stem: ConvEmbed.proj -> Florence2VisionConvEmbed.conv
        (vision + r"convs\.(\d+)\.proj\.", prefix + r"vision_tower.convs.\1.conv."),
        # DaViT blocks: the native implementation flattens the PreNorm/Mlp wrappers.
        (block + r"conv1\.fn\.dw\.", new_block + r"conv1."),
        (block + r"conv2\.fn\.dw\.", new_block + r"conv2."),
        (block + r"(window_attn|channel_attn)\.norm\.", new_block + r"norm1."),
        (block + r"(window_attn|channel_attn)\.fn\.", new_block + r"\4."),
        (block + r"ffn\.norm\.", new_block + r"norm2."),
        (block + r"ffn\.fn\.net\.", new_block + r"ffn."),
        # The multimodal projection layers moved into a dedicated projector module.
        (re.escape(prefix) + r"image_proj_norm\.", prefix + r"multi_modal_projector.image_proj_norm."),
        (re.escape(prefix) + r"image_pos_embed\.", prefix + r"multi_modal_projector.image_position_embed."),
        (
            re.escape(prefix) + r"visual_temporal_embed\.",
            prefix + r"multi_modal_projector.visual_temporal_embed.",
        ),
        # Florence2LanguageForConditionalGeneration.model -> BartModel
        (re.escape(prefix) + r"language_model\.model\.", prefix + r"language_model."),
    ]

    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key == f"{prefix}language_model.final_logits_bias":
            # Generation-only buffer of the vendored language model; BartModel has none.
            continue
        if key == f"{prefix}image_projection":
            # Vendored: an nn.Parameter used as `x @ p`. Native: an nn.Linear without bias,
            # whose weight is the transpose.
            remapped[f"{prefix}multi_modal_projector.image_projection.weight"] = value.transpose(0, 1).contiguous()
            continue
        new_key = key
        for pattern, replacement in rules:
            new_key, count = re.subn(pattern, replacement, new_key, count=1)
            if count:
                break
        remapped[new_key] = value

    return remapped


def load_xvla_weights(model: torch.nn.Module, weights_file: Path) -> None:
    """Load a published checkpoint into an :class:`~physicalai.policies.xvla.model.XVLAModel`.

    Args:
        model: The model to load into.
        weights_file: Path to ``model.safetensors``.
    """
    state_dict = {key.removeprefix(_STATE_DICT_PREFIX): value for key, value in load_file(str(weights_file)).items()}

    if is_vendored_florence_state_dict(state_dict):
        logger.info("Detected the legacy vendored Florence-2 layout; remapping to the native transformers layout.")
        state_dict = remap_vendored_florence_state_dict(state_dict)

    # safetensors deduplicates tied tensors on save: restore whichever alias of the shared
    # token embedding is missing.
    shared_key = f"{_VLM_PREFIX}language_model.shared.weight"
    embed_key = f"{_VLM_PREFIX}language_model.encoder.embed_tokens.weight"
    if shared_key in state_dict and embed_key not in state_dict:
        state_dict[embed_key] = state_dict[shared_key]
    elif embed_key in state_dict and shared_key not in state_dict:
        state_dict[shared_key] = state_dict[embed_key]

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing:
        logger.warning("Missing keys when loading XVLA weights: %d (e.g. %s)", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys when loading XVLA weights: %d (e.g. %s)", len(unexpected), unexpected[:5])


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file.

    Args:
        path: File to read.

    Returns:
        The parsed object.
    """
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _count_visual_features(features: dict[str, Any]) -> int:
    """Count the visual entries of a LeRobot feature mapping.

    Args:
        features: The checkpoint's ``input_features``.

    Returns:
        The number of visual features.
    """
    return sum(1 for spec in features.values() if str(spec.get("type", "")).upper() == "VISUAL")


__all__ = [
    "CheckpointFiles",
    "detect_normalization_mode",
    "extract_dataset_stats",
    "extract_domain_id",
    "extract_tokenizer_max_length",
    "is_vendored_florence_state_dict",
    "load_config",
    "load_xvla_weights",
    "read_action_dim",
    "remap_vendored_florence_state_dict",
    "resolve_checkpoint",
]
