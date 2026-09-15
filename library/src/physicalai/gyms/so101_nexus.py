# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SO101-Nexus Gymnasium adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from gymnasium import spaces

from physicalai.data.observation import Observation

from .gymnasium_gym import GymnasiumGym

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

    from numpy.typing import NDArray

_SO101_NEXUS_AVAILABLE = False
_SO101_NEXUS_IMPORT_ERROR: str | None = None
# A 4096-position encoder spans inclusive tick values 0 through 4095.
_SO101_ENCODER_MAX_TICK = 4095.0
_SO101_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
_SO101_RUNTIME_CALIBRATION = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 2048, "range_min": 746, "range_max": 3412},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": 2048, "range_min": 885, "range_max": 3198},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 2048, "range_min": 907, "range_max": 3103},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 2048, "range_min": 771, "range_max": 3073},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 2048, "range_min": 143, "range_max": 3972},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 2048, "range_min": 2045, "range_max": 3492},
}

try:
    import so101_nexus
    import so101_nexus.mujoco

    _SO101_NEXUS_AVAILABLE = True
except ImportError as error:
    _SO101_NEXUS_IMPORT_ERROR = str(error)
    so101_nexus = None  # type: ignore[assignment]


def _check_so101_nexus_available() -> None:
    """Check whether SO101-Nexus is installed.

    Raises:
        ImportError: If the optional SO101-Nexus dependency is unavailable.
    """
    if not _SO101_NEXUS_AVAILABLE:
        message = (
            "SO101-Nexus is not installed. Install it with:\n"
            "  uv sync --extra so101-nexus\n"
            "or:\n"
            "  pip install so101-nexus\n"
            f"\nOriginal error: {_SO101_NEXUS_IMPORT_ERROR}"
        )
        raise ImportError(message)


def _get_so101_nexus() -> ModuleType:
    """Return the optional SO101-Nexus module after checking availability.

    Returns:
        The imported SO101-Nexus module.

    Raises:
        RuntimeError: If the availability flag and imported module disagree.
    """
    _check_so101_nexus_available()
    if so101_nexus is None:
        message = "SO101-Nexus availability state is inconsistent."
        raise RuntimeError(message)
    return so101_nexus


class SO101NexusGym(GymnasiumGym):
    """Adapt an SO101-Nexus visual environment to PhysicalAI's gym contract.

    The default environment is ``MuJoCoPickLift-v1`` with wrist and overhead
    RGB cameras. Observations are batched: state and actions have shape
    ``[1, 6]``, while images have shape ``[1, C, H, W]`` with values in
    ``[0, 1]``.

    Public state and action coordinates match PhysicalAI Runtime. Body joints
    use calibrated ``[-100, 100]`` units and the gripper uses ``[0, 100]``.
    The adapter converts through Runtime servo ticks, including quantization
    and clipping, while SO101-Nexus continues to use radians internally.

    Pass an explicit visual SO101-Nexus ``config`` for another environment or
    camera set. RGB keys are inferred by removing the ``_camera`` suffix;
    ``camera_key_map`` can select and rename configured cameras. Explicit
    configs must set their initial pose on ``config.robot``. Only absolute
    joint-position control (``pd_joint_pos``) is supported.

    Examples:
        >>> gym = SO101NexusGym()
        >>> observation, info = gym.reset(seed=0)
        >>> observation, reward, terminated, truncated, info = gym.step(
        ...     gym.sample_action(),
        ... )
        >>> gym.close()
    """

    def __init__(
        self,
        gym_id: str = "MuJoCoPickLift-v1",
        observation_width: int = 378,
        observation_height: int = 378,
        device: str | torch.device = "cpu",
        render_mode: str | None = "rgb_array",
        task_description: str | None = None,
        init_pose: str | None = None,
        config: Any | None = None,  # noqa: ANN401
        camera_key_map: Mapping[str, str] | None = None,
        **gym_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Create a visual SO101-Nexus environment.

        Args:
            gym_id: Registered SO101-Nexus environment ID. The default config
                supports only ``MuJoCoPickLift-v1``.
            observation_width: Default camera width in pixels.
            observation_height: Default camera height in pixels.
            device: Torch device used for returned tensors.
            render_mode: Rendering mode passed to the SO101-Nexus environment.
            task_description: Optional fixed task text replacing the
                environment instruction.
            init_pose: Named initial pose for the default config. Set the pose
                on ``config.robot`` when passing an explicit config.
            config: Optional visual SO101-Nexus environment configuration.
            camera_key_map: Mapping from configured RGB observation names to
                policy-facing image keys.
            **gym_kwargs: Additional arguments forwarded to ``gym.make``.

        Raises:
            ValueError: If the requested configuration is incompatible with
                visual absolute-joint-position control.
        """
        nexus = _get_so101_nexus()

        if config is None:
            if gym_id != "MuJoCoPickLift-v1":
                msg = f"config is required for SO101-Nexus environment {gym_id!r}"
                raise ValueError(msg)
            config = nexus.PickConfig(
                obs_mode="visual",
                robot=nexus.RobotConfig(init_pose=init_pose),
                observations=[
                    nexus.JointPositions(),
                    nexus.WristCamera(width=observation_width, height=observation_height),
                    nexus.OverheadCamera(width=observation_width, height=observation_height),
                ],
            )
        elif init_pose is not None:
            message = "set init_pose on config.robot when passing an explicit config"
            raise ValueError(message)

        if getattr(config, "obs_mode", None) != "visual":
            message = "SO101NexusGym requires a config with obs_mode='visual'"
            raise ValueError(message)
        control_mode = gym_kwargs.get("control_mode", "pd_joint_pos")
        if control_mode != "pd_joint_pos":
            message = (
                "SO101NexusGym exposes absolute joint positions and requires "
                f"control_mode='pd_joint_pos', got {control_mode!r}"
            )
            raise ValueError(message)

        inferred_camera_keys = {
            component.name: component.name.removesuffix("_camera")
            for component in config.observations or []
            if "rgb" in getattr(component, "modalities", ())
        }
        self._camera_key_map = dict(
            inferred_camera_keys if camera_key_map is None else camera_key_map,
        )
        if not self._camera_key_map:
            message = "SO101NexusGym requires at least one RGB camera observation"
            raise ValueError(message)
        unknown_camera_keys = self._camera_key_map.keys() - inferred_camera_keys.keys()
        if unknown_camera_keys:
            message = f"camera_key_map contains unconfigured RGB cameras: {unknown_camera_keys}"
            raise ValueError(message)
        if len(set(self._camera_key_map.values())) != len(self._camera_key_map):
            message = "camera_key_map values must be unique"
            raise ValueError(message)

        self._task_description_override = task_description
        super().__init__(
            gym_id=gym_id,
            device=device,
            render_mode=render_mode,
            config=config,
            **gym_kwargs,
        )

        simulator_action_space = cast("spaces.Box", self._env.action_space)
        low = np.asarray(simulator_action_space.low)
        high = np.asarray(simulator_action_space.high)
        self._gripper_limits_rad = (float(low[-1]), float(high[-1]))

    @classmethod
    def vectorize(cls, *args: Any, **kwargs: Any) -> SO101NexusGym:  # noqa: ANN401
        """Reject vectorization, which this adapter does not support.

        Raises:
            NotImplementedError: Always; SO101-Nexus conversion is currently
                implemented only for single environments.
        """
        message = "SO101NexusGym does not support vectorized environments"
        raise NotImplementedError(message)

    @property
    def task_description(self) -> str:
        """The fixed override or current environment instruction."""
        if self._task_description_override is not None:
            return self._task_description_override
        return str(getattr(self._env.unwrapped, "task_description", "Pick up the object."))

    @property
    def action_space(self) -> spaces.Space:
        """The simulator action space converted to Runtime units."""
        nexus = _get_so101_nexus()
        simulator_action_space = cast("spaces.Box", self._env.action_space)

        low_dataset = nexus.sim_qpos_to_dataset_row(
            np.asarray(simulator_action_space.low),
            gripper_limits_rad=self._gripper_limits_rad,
        )
        high_dataset = nexus.sim_qpos_to_dataset_row(
            np.asarray(simulator_action_space.high),
            gripper_limits_rad=self._gripper_limits_rad,
        )
        low = self._dataset_to_runtime(low_dataset)
        high = self._dataset_to_runtime(high_dataset)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Convert a Runtime-unit action to radians and step the simulator.

        Returns:
            The next observation, reward, termination flags, and environment info.
        """
        nexus = _get_so101_nexus()
        simulator_action_space = cast("spaces.Box", self._env.action_space)

        action_for_env = self._normalize_action_for_env(action)
        if action_for_env.dtype in {torch.bfloat16, torch.float16}:
            action_for_env = action_for_env.to(torch.float32)
        raw_action = self._runtime_to_dataset(action_for_env.detach().cpu().numpy())
        raw_action = nexus.dataset_row_to_sim_qpos(
            raw_action,
            gripper_limits_rad=self._gripper_limits_rad,
        )
        raw_action = np.clip(raw_action, simulator_action_space.low, simulator_action_space.high)

        raw_obs, reward, terminated, truncated, info = self._env.step(raw_action)
        normalized_obs = self._normalize_raw_obs(raw_obs)
        obs = self.to_observation(normalized_obs)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def sample_action(self) -> torch.Tensor:
        """Sample an action and expose it in Runtime units.

        Returns:
            A batched action tensor in PhysicalAI Runtime units.
        """
        nexus = _get_so101_nexus()

        dataset_action = nexus.sim_qpos_to_dataset_row(
            self._env.action_space.sample(),
            gripper_limits_rad=self._gripper_limits_rad,
        )
        action = self._dataset_to_runtime(dataset_action)
        return self._normalize_action_for_user(torch.as_tensor(action, device=self.device))

    def to_observation(self, raw_obs: NDArray | dict[str, Any]) -> Observation:
        """Convert cameras and joint radians to a Runtime-unit observation.

        Returns:
            A batched PhysicalAI observation in Runtime units.

        Raises:
            TypeError: If the wrapped environment does not return a dictionary.
        """
        if not isinstance(raw_obs, dict):
            message = f"expected a dictionary observation, got {type(raw_obs).__name__}"
            raise TypeError(message)
        nexus = _get_so101_nexus()

        dataset_state = nexus.sim_qpos_to_dataset_row(
            np.asarray(raw_obs["state"]),
            gripper_limits_rad=self._gripper_limits_rad,
        )
        state = self._dataset_to_runtime(dataset_state)
        images: dict[str, torch.Tensor | np.ndarray] = {
            policy_key: self._convert_image(raw_obs[raw_key]) for raw_key, policy_key in self._camera_key_map.items()
        }
        batch_size = int(state.shape[0])
        return Observation(
            images=cast("Any", images),
            state=torch.as_tensor(state, dtype=torch.float32, device=self.device),
            task=cast("Any", [self.task_description] * batch_size),
        )

    def _dataset_to_runtime(self, values: NDArray) -> NDArray:
        """Convert LeRobot degree/percent rows to Runtime-normalized rows.

        Returns:
            A converted copy using Runtime's calibrated joint ranges.
        """
        ticks = self._dataset_to_ticks(values)
        flat_ticks = ticks.reshape(-1, ticks.shape[-1])
        normalized = np.stack(
            [self._ticks_to_runtime(row) for row in flat_ticks],
        )
        return normalized.reshape(ticks.shape)

    def _runtime_to_dataset(self, values: NDArray) -> NDArray:
        """Convert Runtime-normalized rows to LeRobot degree/percent rows.

        Returns:
            A converted copy using LeRobot degrees for body joints.
        """
        normalized = np.asarray(values, dtype=np.float32)
        flat_normalized = normalized.reshape(-1, normalized.shape[-1])
        ticks = np.stack(
            [self._runtime_to_ticks(row) for row in flat_normalized],
        ).reshape(normalized.shape)
        return self._ticks_to_dataset(ticks)

    @staticmethod
    def _ticks_to_runtime(ticks: NDArray) -> NDArray:
        """Convert encoder ticks to PhysicalAI Runtime coordinates.

        Returns:
            Runtime-normalized joint values.
        """
        result = np.empty(len(_SO101_JOINT_ORDER), dtype=np.float32)
        for index, name in enumerate(_SO101_JOINT_ORDER):
            calibration = _SO101_RUNTIME_CALIBRATION[name]
            range_min = calibration["range_min"]
            range_max = calibration["range_max"]
            tick = int(np.clip(ticks[index], range_min, range_max))
            if name == "gripper":
                result[index] = (tick - range_min) / (range_max - range_min) * 100.0
            else:
                result[index] = (tick - range_min) / (range_max - range_min) * 200.0 - 100.0
        return result

    @staticmethod
    def _runtime_to_ticks(values: NDArray) -> NDArray:
        """Convert PhysicalAI Runtime coordinates to encoder ticks.

        Returns:
            Calibrated integer encoder positions.
        """
        result = np.empty(len(_SO101_JOINT_ORDER), dtype=np.int32)
        for index, name in enumerate(_SO101_JOINT_ORDER):
            calibration = _SO101_RUNTIME_CALIBRATION[name]
            range_min = calibration["range_min"]
            range_max = calibration["range_max"]
            if name == "gripper":
                value = float(np.clip(values[index], 0.0, 100.0))
                tick = round(range_min + value / 100.0 * (range_max - range_min))
            else:
                value = float(np.clip(values[index], -100.0, 100.0))
                tick = round(range_min + (value + 100.0) / 200.0 * (range_max - range_min))
            result[index] = int(np.clip(tick, range_min, range_max))
        return result

    @staticmethod
    def _dataset_to_ticks(values: NDArray) -> NDArray:
        """Convert LeRobot degree/percent rows to encoder ticks.

        Returns:
            Integer encoder positions shaped like the input.
        """
        dataset = np.asarray(values, dtype=np.float32)
        ticks = np.empty_like(dataset, dtype=np.int32)
        for index, name in enumerate(_SO101_JOINT_ORDER):
            calibration = _SO101_RUNTIME_CALIBRATION[name]
            if name == "gripper":
                ticks[..., index] = np.rint(
                    calibration["range_min"]
                    + dataset[..., index] / 100.0 * (calibration["range_max"] - calibration["range_min"]),
                )
            else:
                midpoint = (calibration["range_min"] + calibration["range_max"]) / 2.0
                ticks[..., index] = np.rint(
                    midpoint + dataset[..., index] * _SO101_ENCODER_MAX_TICK / 360.0,
                )
        return ticks

    @staticmethod
    def _ticks_to_dataset(values: NDArray) -> NDArray:
        """Convert encoder ticks to LeRobot degree/percent rows.

        Returns:
            Degree/percent rows shaped like the input.
        """
        ticks = np.asarray(values, dtype=np.float32)
        dataset = np.empty_like(ticks, dtype=np.float32)
        for index, name in enumerate(_SO101_JOINT_ORDER):
            calibration = _SO101_RUNTIME_CALIBRATION[name]
            if name == "gripper":
                dataset[..., index] = (
                    (ticks[..., index] - calibration["range_min"])
                    / (calibration["range_max"] - calibration["range_min"])
                    * 100.0
                )
            else:
                midpoint = (calibration["range_min"] + calibration["range_max"]) / 2.0
                dataset[..., index] = (ticks[..., index] - midpoint) * 360.0 / _SO101_ENCODER_MAX_TICK
        return dataset

    def _convert_image(self, image: Any) -> torch.Tensor:  # noqa: ANN401
        """Convert a batched HWC image to float BCHW.

        Returns:
            The image tensor scaled to ``[0, 1]`` when its input is uint8.

        Raises:
            ValueError: If the input is not a batched HWC image.
        """
        tensor = torch.as_tensor(image, device=self.device)
        expected_dimensions = 4
        if tensor.ndim != expected_dimensions or tensor.shape[-1] not in {1, 3, 4}:
            msg = f"expected a batched HWC image, got shape {tuple(tensor.shape)}"
            raise ValueError(msg)
        is_uint8 = tensor.dtype == torch.uint8
        tensor = tensor.permute(0, 3, 1, 2).to(torch.float32)
        return tensor / 255.0 if is_uint8 else tensor
