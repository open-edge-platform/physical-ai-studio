# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bimanual WidowXAI adapter.

Composes two :class:`WidowXAIAdapter` instances (left + right) behind the
backend's :class:`RobotClient` interface, prefixing all keys with
``left_`` / ``right_`` respectively.
"""

from typing import Literal

from robots.robot_client import RobotClient
from robots.widowxai.adapter import WidowXAIAdapter
from schemas.robot import RobotType


class BimanualWidowXAIAdapter(RobotClient):
    """Two-arm WidowXAI adapter that merges left/right state with prefixed keys."""

    name = "BimanualWidowXAI"

    def __init__(
        self,
        left: WidowXAIAdapter,
        right: WidowXAIAdapter,
        mode: Literal["follower", "leader"],
    ) -> None:
        self._left = left
        self._right = right
        self._mode = mode

    # ------------------------------------------------------------------
    # RobotClient protocol
    # ------------------------------------------------------------------

    @property
    def robot_type(self) -> RobotType:
        if self._mode == "follower":
            return RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER
        return RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER

    @property
    def is_connected(self) -> bool:
        return self._left.is_connected and self._right.is_connected

    async def connect(self) -> None:
        await self._left.connect()
        await self._right.connect()

    async def disconnect(self) -> None:
        await self._left.disconnect()
        await self._right.disconnect()

    async def ping(self) -> dict:
        return self._create_event("pong")

    # ------------------------------------------------------------------
    # State read/write
    # ------------------------------------------------------------------

    async def read_state(self, *, normalize: bool = True) -> dict:
        left_result = await self._left.read_state(normalize=normalize)
        right_result = await self._right.read_state(normalize=normalize)

        merged: dict[str, float] = {}
        for k, v in left_result["state"].items():
            merged[f"left_{k}"] = v
        for k, v in right_result["state"].items():
            merged[f"right_{k}"] = v

        return self._create_event(
            "state_was_updated",
            state=merged,
            is_controlled=left_result.get("is_controlled", False),
        )

    async def read_forces(self) -> dict | None:
        left_result = await self._left.read_forces()
        right_result = await self._right.read_forces()

        if left_result is None and right_result is None:
            return None

        merged: dict[str, float] = {}
        if left_result is not None:
            for k, v in left_result["state"].items():
                merged[f"left_{k}"] = v
        if right_result is not None:
            for k, v in right_result["state"].items():
                merged[f"right_{k}"] = v

        return self._create_event(
            "force_was_updated",
            state=merged,
            is_controlled=self._left.is_controlled if hasattr(self._left, "is_controlled") else False,
        )

    def _split_prefixed(self, data: dict, prefix: str) -> dict:
        """Extract keys that start with *prefix* and strip the prefix."""
        return {k[len(prefix) :]: v for k, v in data.items() if k.startswith(prefix)}

    async def set_joints_state(self, joints: dict, goal_time: float) -> dict:
        left_joints = self._split_prefixed(joints, "left_")
        right_joints = self._split_prefixed(joints, "right_")
        await self._left.set_joints_state(left_joints, goal_time)
        await self._right.set_joints_state(right_joints, goal_time)
        return self._create_event("joints_state_was_set", joints=joints)

    async def set_forces(self, forces: dict) -> dict:
        left_forces = self._split_prefixed(forces, "left_")
        right_forces = self._split_prefixed(forces, "right_")
        await self._left.set_forces(left_forces)
        await self._right.set_forces(right_forces)
        return forces

    async def enable_torque(self) -> dict:
        await self._left.enable_torque()
        await self._right.enable_torque()
        return self._create_event("torque_was_enabled")

    async def disable_torque(self) -> dict:
        await self._left.disable_torque()
        await self._right.disable_torque()
        return self._create_event("torque_was_disabled")

    def features(self) -> list[str]:
        left_features = [f"left_{f}" for f in self._left.features()]
        right_features = [f"right_{f}" for f in self._right.features()]
        return left_features + right_features
