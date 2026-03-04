# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from physicalai.export import Export
from physicalai.inference import InferenceModel
from physicalai.policies import ACT, Pi0, SmolVLA
from physicalai.policies.base import Policy

from schemas import Model
from utils.device import get_torch_device


def load_policy(model: Model) -> Export | Policy:
    """Load existing model."""
    model_path = str(Path(model.path) / "model.ckpt")
    if model.policy == "act":
        return ACT.load_from_checkpoint(model_path)
    if model.policy == "pi0":
        return Pi0.load_from_checkpoint(model_path, weights_only=True)
    if model.policy == "smolvla":
        return SmolVLA.load_from_checkpoint(model_path)
    raise ValueError(f"Policy {model.policy} not implemented.")


def load_inference_model(model: Model, backend: str) -> InferenceModel:
    """Loads inference model."""
    inference_device = "auto"
    if backend == "torch":
        inference_device = get_torch_device()

    export_dir = Path(model.path) / "exports" / backend
    return InferenceModel(export_dir=export_dir, policy_name=model.policy, backend=backend, device=inference_device)


def setup_policy(model: Model) -> Policy:
    """Setup policy for Model training."""
    if model.policy == "act":
        return ACT()
    if model.policy == "pi0":
        return Pi0()
    if model.policy == "smolvla":
        return SmolVLA()

    raise ValueError(f"Policy not implemented yet: {model.policy}")
