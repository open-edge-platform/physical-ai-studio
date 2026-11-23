# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot policy types and saved processors the plugin accepts.

Only policy types whose training, packaging and Runtime inference have been checked are
allowed; add a type here after checking it. Several LeRobot 0.7 policies (``fineart_vla``,
``gaussian_actor``, ``groot``, ``pi0_fast``, ``wall_x``) load remote code without a pinned
revision and must stay out until that is reviewed.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

SUPPORTED_POLICY_TYPES = frozenset({"act", "diffusion", "pi05", "smolvla"})
PROCESSOR_CONFIGS = ("policy_preprocessor.json", "policy_postprocessor.json")

_CLASS_PATH_RE = re.compile(r"lerobot(\.[A-Za-z_][A-Za-z0-9_]*)+")
_STATE_FILE_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.safetensors")


def require_supported_policy(policy_type: object) -> str:
    """Return *policy_type* if the plugin supports it.

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


def validate_processor_configs(model_dir: Path) -> None:
    """Check a checkpoint's saved processor pipelines before LeRobot loads them.

    LeRobot imports any ``class`` path named in these files, and loads each step's
    ``state_file`` from the model directory. Allow only registered steps or classes under
    ``lerobot.``, and state files that are plain ``.safetensors`` names inside *model_dir*.

    Raises:
        FileNotFoundError: If a processor config or one of its state files is missing.
        ValueError: If a processor config is malformed or names a disallowed class or file.
    """
    for name in PROCESSOR_CONFIGS:
        path = model_dir / name
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
    elif not isinstance(class_path, str) or _CLASS_PATH_RE.fullmatch(class_path) is None:
        msg = f"LeRobot {name} step {index} names a class outside `lerobot.`: {class_path!r}."
        raise ValueError(msg)
    state_file = step.get("state_file")
    if state_file is None:
        return
    if not isinstance(state_file, str) or _STATE_FILE_RE.fullmatch(state_file) is None:
        msg = f"LeRobot {name} step {index} has an invalid state file: {state_file!r}."
        raise ValueError(msg)
    state_path = model_dir / state_file
    if not state_path.is_file():
        msg = f"LeRobot {name} step {index} state file is missing: {state_file}"
        raise FileNotFoundError(msg)
