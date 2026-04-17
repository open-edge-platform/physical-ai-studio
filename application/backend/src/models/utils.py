# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from physicalai.inference import InferenceModel
from physicalai.policies import ACT, Pi0, Pi05, SmolVLA
from physicalai.policies.base import Policy

from schemas import Model
from utils.device import get_torch_device


def load_policy(model: Model, *, compile_model: bool = False) -> Policy:
    """Load existing model."""
    model_path = str(Path(model.path) / "model.ckpt")
    if model.policy == "act":
        policy = ACT.load_from_checkpoint(model_path)
    elif model.policy == "pi0":
        policy = Pi0.load_from_checkpoint(model_path, weights_only=True)
    elif model.policy == "pi05":
        policy = Pi05.load_from_checkpoint(model_path)
    elif model.policy == "smolvla":
        policy = SmolVLA.load_from_checkpoint(model_path)
    else:
        raise ValueError(f"Policy {model.policy} not implemented.")

    if compile_model:
        compile_mode = "default"
        policy.forward = torch.compile(policy.forward, mode=compile_mode)  # type: ignore[method-assign]
    return policy


def load_inference_model(model: Model, backend: str) -> InferenceModel:
    """Loads inference model."""
    inference_device = "auto"
    if backend == "torch":
        inference_device = get_torch_device()

    export_dir = Path(model.path) / "exports" / backend
    return InferenceModel(export_dir=export_dir, policy_name=model.policy, backend=backend, device=inference_device)


def setup_policy(model: Model, *, compile_model: bool = False) -> Policy:
    """Setup policy for Model training."""
    if model.policy == "act":
        return ACT(compile_model=compile_model)
    if model.policy == "pi0":
        return Pi0(compile_model=compile_model)
    if model.policy == "pi05":
        return Pi05(pretrained_name_or_path="lerobot/pi05_base", compile_model=compile_model)
    if model.policy == "smolvla":
        return SmolVLA(compile_model=compile_model)

    raise ValueError(f"Policy not implemented yet: {model.policy}")
