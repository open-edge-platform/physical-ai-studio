# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot policy types and saved processors the plugin accepts.

Only policy types whose training, packaging and Runtime inference have been checked are
allowed by default; add a type here after checking it. Several LeRobot 0.7 policies
(``fineart_vla``, ``gaussian_actor``, ``groot``, ``pi0_fast``, ``wall_x``) load remote code
without a pinned revision and require explicit ``trust_remote_code=True`` opt-in for export.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

SUPPORTED_POLICY_TYPES = frozenset({"act", "diffusion", "pi05", "smolvla"})
VERIFIED_EXPORT_POLICIES = frozenset({"act", "diffusion", "pi05", "smolvla"})
PROCESSOR_CONFIGS = ("policy_preprocessor.json", "policy_postprocessor.json")
ALLOWED_PROCESSOR_STEPS = frozenset(
    {
        "absolute_actions_processor",
        "device_processor",
        "normalizer_processor",
        "pi05_prepare_state_tokenizer_processor_step",
        "relative_actions_processor",
        "rename_observations_processor",
        "smolvla_new_line_processor",
        "to_batch_processor",
        "tokenizer_processor",
        "unnormalizer_processor",
    },
)
ALLOWED_PROCESSOR_CLASSES = frozenset(
    {
        "lerobot.policies.pi05.processor_pi05.Pi05PrepareStateTokenizerProcessorStep",
        "lerobot.processor.batch_processor.AddBatchDimensionProcessorStep",
        "lerobot.processor.device_processor.DeviceProcessorStep",
        "lerobot.processor.newline_task_processor.NewLineTaskProcessorStep",
        "lerobot.processor.normalize_processor.NormalizerProcessorStep",
        "lerobot.processor.normalize_processor.UnnormalizerProcessorStep",
        "lerobot.processor.relative_action_processor.AbsoluteActionsProcessorStep",
        "lerobot.processor.relative_action_processor.RelativeActionsProcessorStep",
        "lerobot.processor.rename_processor.RenameObservationsProcessorStep",
        "lerobot.processor.tokenizer_processor.TokenizerProcessorStep",
    },
)

_POLICY_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_STATE_FILE_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.safetensors")


def require_supported_policy(policy_type: object) -> str:
    """Return *policy_type* if the plugin supports it for training and native packaging.

    Returns:
        The policy type.

    Raises:
        ValueError: If the policy type is unknown or not supported.
    """
    if not isinstance(policy_type, str) or not policy_type.strip():
        msg = "Cannot determine the LeRobot policy type; set `policy.type`."
        raise ValueError(msg)
    if policy_type not in SUPPORTED_POLICY_TYPES:
        supported = ", ".join(sorted(SUPPORTED_POLICY_TYPES))
        msg = f"LeRobot policy {policy_type!r} is not supported; supported policies: {supported}."
        raise ValueError(msg)
    return policy_type


def require_exportable_policy(policy_type: object, *, trust_remote_code: bool = False) -> str:
    """Return *policy_type* if it is allowed for bundle export.

    Policies in :data:`VERIFIED_EXPORT_POLICIES` are allowed by default. Other valid LeRobot
    policy identifiers require ``trust_remote_code=True`` because some upstream policies load
    unpinned remote Hugging Face code.

    Returns:
        The validated policy type.

    Raises:
        ValueError: If *policy_type* is invalid or outside the verified allowlist without opt-in.
    """
    if not isinstance(policy_type, str) or not policy_type.strip():
        msg = "Cannot determine the LeRobot policy type from the checkpoint or repository."
        raise ValueError(msg)
    normalized = policy_type.strip()
    if _POLICY_NAME_RE.fullmatch(normalized) is None:
        msg = f"LeRobot policy type must match [a-zA-Z0-9_-]{{1,64}}, got {policy_type!r}."
        raise ValueError(msg)
    if normalized in VERIFIED_EXPORT_POLICIES or trust_remote_code:
        return normalized
    verified = ", ".join(sorted(VERIFIED_EXPORT_POLICIES))
    msg = (
        f"LeRobot policy {normalized!r} is outside the audited export allowlist ({verified}) and may load "
        "unpinned remote code. Pass `options=LeRobotExportOptions(trust_remote_code=True)` to opt in."
    )
    raise ValueError(msg)


def validate_processor_configs(model_dir: Path) -> None:
    """Check a checkpoint's saved processor pipelines before LeRobot loads them.

    LeRobot instantiates any registered step or ``class`` path named in these files, and
    loads each step's ``state_file`` from the model directory. Allow only the processor
    steps used by supported policies, and plain ``.safetensors`` state files inside
    *model_dir*.

    Raises:
        FileNotFoundError: If a processor config or one of its state files is missing.
        ValueError: If a processor config is malformed or names a disallowed step or file.
    """
    for name in PROCESSOR_CONFIGS:
        path = model_dir / name
        if path.is_symlink():
            msg = f"LeRobot {name} cannot be a symbolic link: {path}"
            raise ValueError(msg)
        if not path.is_file():
            msg = f"LeRobot checkpoint is missing {name}: {model_dir}"
            raise FileNotFoundError(msg)
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            msg = f"LeRobot {name} is not valid JSON."
            raise ValueError(msg) from error
        steps = config.get("steps") if isinstance(config, dict) else None
        if not isinstance(steps, list):
            msg = f"LeRobot {name} must contain a list of steps."
            raise ValueError(msg)
        for index, step in enumerate(steps):
            _validate_step(model_dir, name, index, step)


def _validate_step(model_dir: Path, name: str, index: int, step: object) -> None:
    if not isinstance(step, dict):
        msg = f"LeRobot {name} step {index} must be an object."
        raise ValueError(msg)
    registry_name = step.get("registry_name")
    class_path = step.get("class")
    if registry_name is not None:
        if not isinstance(registry_name, str) or class_path is not None:
            msg = f"LeRobot {name} step {index} must name a registered step or a class, not both."
            raise ValueError(msg)
        if registry_name not in ALLOWED_PROCESSOR_STEPS:
            msg = f"LeRobot {name} step {index} uses a disallowed processor step: {registry_name!r}."
            raise ValueError(msg)
    elif not isinstance(class_path, str) or class_path not in ALLOWED_PROCESSOR_CLASSES:
        msg = f"LeRobot {name} step {index} names a disallowed processor class: {class_path!r}."
        raise ValueError(msg)
    state_file = step.get("state_file")
    if state_file is None:
        return
    if not isinstance(state_file, str) or ".." in state_file or _STATE_FILE_RE.fullmatch(state_file) is None:
        msg = f"LeRobot {name} step {index} has an invalid state file: {state_file!r}."
        raise ValueError(msg)
    state_path = model_dir / state_file
    if state_path.is_symlink():
        msg = f"LeRobot {name} step {index} state file cannot be a symbolic link: {state_file}"
        raise ValueError(msg)
    if not state_path.is_file():
        msg = f"LeRobot {name} step {index} state file is missing: {state_file}"
        raise FileNotFoundError(msg)
