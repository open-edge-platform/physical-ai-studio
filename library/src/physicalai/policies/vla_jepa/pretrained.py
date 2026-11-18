# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Resolve pinned, safe checkpoint artifacts without constructing a model."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoProcessor, AutoVideoProcessor, Qwen3VLConfig, VJEPA2Config

from physicalai.data import Feature, FeatureType, NormalizationParameters

from .config import VLAJEPAConfig

DEFAULT_CHECKPOINT = "lerobot/VLA-JEPA-Pretrain"
CHECKPOINT_REVISIONS = {
    DEFAULT_CHECKPOINT: "e946c3e5b538d760f4b4ff239d1b1c12090c041d",
    "lerobot/VLA-JEPA-LIBERO": "735d9f692981e286ade093b5046627eda876e5d0",
    "lerobot/VLA-JEPA-SimplerEnv": "8b9ec33f57d8cdc0e064873a637a1f92e3164e53",
}
QWEN_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
JEPA_ENCODER = "facebook/vjepa2-vitl-fpc64-256"
JEPA_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        msg = f"Expected a JSON object in {path.name}."
        raise TypeError(msg)
    return value


def _setting[T](data: dict[str, Any], name: str, default: T) -> T:
    value = data.get(name, default)
    if type(value) is not type(default):
        msg = f"Invalid checkpoint setting {name!r}: expected {type(default).__name__}."
        raise TypeError(msg)
    return cast("T", value)


def _stats(path: Path) -> dict[str, dict[str, list]]:
    pipeline = _read_json(path / "policy_preprocessor.json")
    steps = pipeline.get("steps")
    if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
        msg = "Invalid checkpoint preprocessor steps."
        raise TypeError(msg)
    statistics: dict[str, dict[str, list]] = {}
    for step in steps:
        if step.get("registry_name") != "normalizer_processor":
            continue
        filename = step.get("state_file")
        if (
            not isinstance(filename, str)
            or any(part in filename for part in ("/", "\\", ".."))
            or Path(filename).suffix != ".safetensors"
        ):
            msg = "Invalid normalization state filename."
            raise ValueError(msg)
        # Hub snapshots use symlinks into a sibling blob cache; validate the asset name and parent.
        state_path = path / filename
        if not state_path.parent.resolve().is_relative_to(path.resolve()):
            msg = "Normalization state escapes the checkpoint directory."
            raise ValueError(msg)
        for key, tensor in load_file(state_path).items():
            feature, stat = key.rsplit(".", 1)
            if stat not in {"mean", "std", "min", "max", "q01", "q99", "mask"}:
                msg = f"Unexpected normalization statistic: {stat}"
                raise ValueError(msg)
            if tensor.ndim != 1 or not tensor.isfinite().all():
                msg = f"Invalid normalization tensor: {key}"
                raise ValueError(msg)
            statistics.setdefault(feature, {})[stat] = tensor.bool().tolist() if stat == "mask" else tensor.tolist()
    return statistics


def _features(data: object, stats: dict[str, dict[str, list]]) -> list[Feature]:
    if not isinstance(data, dict):
        msg = "Checkpoint features must be a JSON object."
        raise TypeError(msg)
    features = []
    for name, spec in data.items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            msg = "Invalid checkpoint feature definition."
            raise TypeError(msg)
        shape = spec.get("shape")
        if not isinstance(shape, list) or not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
            msg = f"Invalid feature shape: {name}"
            raise ValueError(msg)
        kind = FeatureType(spec.get("type"))
        feature_stats = stats.get(name)
        features.append(
            Feature(
                name=name.removeprefix("observation.images.").removeprefix("observation."),
                shape=tuple(shape),
                ftype=kind,
                normalization_data=NormalizationParameters(**feature_stats) if feature_stats else None,
            ),
        )
    return features


def resolve_pretrained(  # noqa: PLR0912, PLR0914, PLR0915
    name_or_path: str | Path,
    revision: str | None = None,
    *,
    enable_world_model: bool = True,
) -> tuple[VLAJEPAConfig, Path]:
    """Return a converted VLA-JEPA config, embedded processor assets, and weight path.

    Custom Hub repositories require an explicit commit SHA. Original author .pt
    files are intentionally unsupported; use the safetensors conversion.

    Raises:
        ValueError: If the checkpoint format, revision, or backbone is unsupported.
        TypeError: If checkpoint metadata has unexpected types.
        FileNotFoundError: If the safetensors weights are missing.
    """
    path = Path(name_or_path).expanduser()
    if not path.is_dir():
        revision = revision or CHECKPOINT_REVISIONS.get(str(name_or_path))
        if not isinstance(revision, str) or re.fullmatch(r"[a-fA-F0-9]{40}", revision) is None:
            msg = "Custom VLA-JEPA Hub checkpoints require a concrete 40-character pretrained_revision."
            raise ValueError(msg)
        path = Path(
            snapshot_download(
                repo_id=str(name_or_path),
                revision=revision,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "policy_preprocessor.json",
                    "policy_preprocessor*.safetensors",
                ],
            ),
        )
    path = path.resolve()
    data = _read_json(path / "config.json")
    if data.get("type") != "vla_jepa":
        msg = "Expected a converted VLA-JEPA checkpoint."
        raise ValueError(msg)
    if data.get("qwen_model_name") != QWEN_MODEL:
        msg = "Only the released Qwen3-VL-2B VLA-JEPA backbone is supported."
        raise ValueError(msg)
    weights_path = path / "model.safetensors"
    if not weights_path.is_file():
        msg = "VLA-JEPA requires model.safetensors."
        raise FileNotFoundError(msg)
    stats = _stats(path)
    inputs = _features(data.get("input_features"), stats)
    outputs = _features(data.get("output_features"), stats)
    processor = AutoProcessor.from_pretrained(QWEN_MODEL, revision=QWEN_REVISION)
    tokenizer = processor.tokenizer
    special_count = _setting(data, "chunk_size", 7) * 4
    tokenizer.add_tokens(
        [f"<|action_{index}|>" for index in range(special_count)] + ["<|embodied_action|>"],
        special_tokens=True,
    )
    qwen = Qwen3VLConfig.from_pretrained(QWEN_MODEL, revision=QWEN_REVISION)
    qwen_config = qwen.to_dict()
    qwen_config["text_config"]["vocab_size"] = max(qwen_config["text_config"]["vocab_size"], len(tokenizer))
    normalization = data.get("normalization_mapping")
    if not isinstance(normalization, dict):
        msg = "Missing checkpoint normalization mapping."
        raise TypeError(msg)
    defaults = VLAJEPAConfig()
    source_fields = {
        "dtype": "torch_dtype",
        "num_embodied_action_tokens": "num_embodied_action_tokens_per_instruction",
        "num_inference_steps": "num_inference_timesteps",
        "num_timestep_buckets": "action_num_timestep_buckets",
        "noise_beta_alpha": "action_noise_beta_alpha",
        "noise_beta_beta": "action_noise_beta_beta",
        "noise_s": "action_noise_s",
        "pre_snap_gripper": "pre_snap_gripper_action",
        "binarize_gripper": "binarize_gripper_action",
    }
    settings = {
        field: _setting(data, source_fields.get(field, field), getattr(defaults, field))
        for field in (
            "dtype",
            "chunk_size",
            "n_action_steps",
            "action_dim",
            "state_dim",
            "action_hidden_size",
            "action_num_layers",
            "action_num_heads",
            "action_attention_head_dim",
            "action_dropout",
            "num_target_vision_tokens",
            "action_max_seq_len",
            "num_inference_steps",
            "num_timestep_buckets",
            "noise_beta_alpha",
            "noise_beta_beta",
            "noise_s",
            "repeated_diffusion_steps",
            "num_action_tokens_per_timestep",
            "num_embodied_action_tokens",
            "prompt_template",
            "clip_normalized_actions",
            "pre_snap_gripper",
            "binarize_gripper",
            "gripper_dim",
            "gripper_threshold",
            "world_model_loss_weight",
            "predictor_depth",
            "predictor_num_heads",
            "predictor_mlp_ratio",
        )
    }
    frames = _setting(data, "num_video_frames", 8)
    tubelet = _setting(data, "jepa_tubelet_size", 2)
    if tubelet <= 0 or frames < 2 * tubelet or frames % tubelet:
        msg = "Invalid checkpoint temporal prompt layout."
        raise ValueError(msg)
    resolution = data.get("resize_images_to") or [224, 224]
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2  # noqa: PLR2004
        or any(type(size) is not int or size <= 0 for size in resolution)
    ):
        msg = "Invalid checkpoint image resolution."
        raise ValueError(msg)
    if not isinstance(processor.chat_template, str):
        msg = "Expected a single Qwen chat template."
        raise TypeError(msg)
    world_config = {}
    video_config = {}
    if enable_world_model:
        if data.get("jepa_encoder_name") != JEPA_ENCODER:
            msg = "Only the released V-JEPA2 target encoder is supported."
            raise ValueError(msg)
        world_config = VJEPA2Config.from_pretrained(JEPA_ENCODER, revision=JEPA_REVISION).to_dict()
        video_processor = AutoVideoProcessor.from_pretrained(JEPA_ENCODER, revision=JEPA_REVISION)
        video_config = json.loads(video_processor.to_json_string())
    views = data.get("world_model_num_views")
    if views is None:
        views = 2
    if type(views) is not int:
        msg = "Invalid world-model view count."
        raise TypeError(msg)
    config = replace(
        defaults,
        **settings,
        input_features=inputs,
        output_features=outputs,
        qwen_config=qwen_config,
        enable_world_model=enable_world_model,
        world_model_config=world_config,
        video_processor_config=video_config,
        num_video_frames=frames,
        world_model_num_views=views,
        tokenizer_json=tokenizer.backend_tokenizer.to_str(),
        image_processor_config=json.loads(processor.image_processor.to_json_string()),
        chat_template=processor.chat_template,
        embodied_action_token_id=tokenizer.convert_tokens_to_ids("<|embodied_action|>"),
        num_latent_tokens=frames // tubelet - 1,
        num_special_action_tokens=special_count,
        image_resolution=(resolution[0], resolution[1]),
        state_normalization=normalization.get("STATE", "IDENTITY"),
        action_normalization=normalization.get("ACTION", "IDENTITY"),
    )
    return config, weights_path
