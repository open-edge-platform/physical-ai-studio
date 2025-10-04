# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pydantic-based configuration for Dummy policy.

This demonstrates how Pydantic models work seamlessly with jsonargparse.
"""

import torch
from pydantic import BaseModel, Field, field_validator


class DummyModelConfigPydantic(BaseModel):
    """Pydantic configuration for DummyModel.

    Pydantic provides:
    - Runtime validation
    - JSON schema generation
    - Better error messages
    - Default values with Field()
    """

    action_shape: list[int] = Field(
        description="Shape of action space",
        examples=[[7], [4, 2]],
    )
    n_action_steps: int = Field(
        default=1,
        ge=1,
        description="Number of action steps per chunk",
    )
    temporal_ensemble_coeff: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Temporal ensemble coefficient. If None, uses action queue.",
    )
    n_obs_steps: int = Field(
        default=1,
        ge=1,
        description="Number of observation steps",
    )
    horizon: int | None = Field(
        default=None,
        ge=1,
        description="Prediction horizon. If None, defaults to n_action_steps",
    )

    @field_validator("action_shape")
    @classmethod
    def validate_action_shape(cls, v: list[int]) -> list[int]:
        """Validate action shape is not empty."""
        if not v:
            msg = "action_shape cannot be empty"
            raise ValueError(msg)
        if any(x <= 0 for x in v):
            msg = "action_shape dimensions must be positive"
            raise ValueError(msg)
        return v

    def to_torch_size(self) -> torch.Size:
        """Convert action_shape to torch.Size."""
        return torch.Size(self.action_shape)


class OptimizerConfigPydantic(BaseModel):
    """Pydantic configuration for optimizer."""

    optimizer_type: str = Field(
        default="adam",
        description="Optimizer type (adam, sgd, adamw)",
    )
    learning_rate: float = Field(
        default=1e-4,
        gt=0.0,
        description="Learning rate",
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight decay (L2 regularization)",
    )
    betas: tuple[float, float] = Field(
        default=(0.9, 0.999),
        description="Betas for Adam optimizer",
    )

    @field_validator("optimizer_type")
    @classmethod
    def validate_optimizer_type(cls, v: str) -> str:
        """Validate optimizer type is supported."""
        allowed = {"adam", "sgd", "adamw"}
        if v.lower() not in allowed:
            msg = f"optimizer_type must be one of {allowed}, got {v}"
            raise ValueError(msg)
        return v.lower()


class DummyConfigPydantic(BaseModel):
    """Pydantic configuration for Dummy policy.

    This shows nested Pydantic models working with jsonargparse.
    """

    model: DummyModelConfigPydantic = Field(
        description="Model configuration",
    )
    optimizer: OptimizerConfigPydantic | None = Field(
        default=None,
        description="Optimizer configuration. If None, uses default Adam.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": {
                    "action_shape": [7],
                    "n_action_steps": 4,
                    "temporal_ensemble_coeff": 0.1,
                    "n_obs_steps": 2,
                    "horizon": 8,
                },
                "optimizer": {
                    "optimizer_type": "adam",
                    "learning_rate": 0.001,
                    "weight_decay": 0.00001,
                    "betas": [0.9, 0.999],
                },
            },
        },
    }
