# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LIBERO benchmark environment wrapper with optional dependencies.

This module provides a Gym wrapper for the LIBERO benchmark suite, designed
to be used with the existing DataModule for validation during policy training.

This is PR1 of the benchmarking pipeline infrastructure:
- LiberoGym class extending Gym base class
- Support for all 5 LIBERO task suites
- LeRobot-compatible observation format
- Control mode support (relative/absolute)

Example:
    Single gym instance:

    from physicalai.gyms import LiberoGym

    # Create a single gym for one task
    gym = LiberoGym(
        task_suite="libero_spatial",
        task_id=0,
        observation_height=224,
        observation_width=224,
    )

    # Use with DataModule for validation
    from physicalai.data import LeRobotDataModule

    datamodule = LeRobotDataModule(
        repo_id="your_dataset",
        batch_size=32,
        val_gyms=gym,  # Pass gym for validation
    )

    Multiple gyms across suites:

    from physicalai.gyms import create_libero_gyms

    # Create gyms for multiple task suites
    gyms = create_libero_gyms(
        task_suites=["libero_spatial", "libero_object"],
        task_ids=[0, 1, 2],  # First 3 tasks from each suite
        observation_height=224,
        observation_width=224,
    )

    # Use with DataModule
    datamodule = LeRobotDataModule(
        repo_id="your_dataset",
        batch_size=32,
        val_gyms=gyms,  # Pass all gyms for multi-suite validation
    )

Note:
    - Requires optional dependencies: `uv pip install hf-libero`
    - Images are automatically rotated 180° to match LeRobot conventions
    - State space: 8-dim (3 pos + 3 orientation + 2 gripper)
    - Action space: 7-dim
    - See TASK_SUITE_MAX_STEPS for episode limits per suite
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces

from physicalai.data.observation import Observation
from physicalai.gyms.base import Gym

if TYPE_CHECKING:
    from libero.libero.benchmark import Benchmark

logger = logging.getLogger(__name__)

# Optional imports - check at runtime
_LIBERO_AVAILABLE = False
_LIBERO_IMPORT_ERROR: str | None = None

try:
    from libero.libero import benchmark as libero_benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle

    _LIBERO_AVAILABLE = True
except ImportError as e:
    _LIBERO_IMPORT_ERROR = str(e)
    # Create placeholder for type hints
    libero_benchmark = None  # type: ignore[assignment]
    get_libero_path = None  # type: ignore[assignment]
    OffScreenRenderEnv = None  # type: ignore[assignment, misc]
    quat2axisangle = None  # type: ignore[assignment]


def _check_libero_available() -> None:
    """Check if LIBERO is available and raise helpful error if not.

    Raises:
        ImportError: If LIBERO is not installed.
    """
    if not _LIBERO_AVAILABLE:
        msg = (
            "LIBERO is not installed. Please install it with:\n"
            "  uv pip install -e '.[libero]'\n"
            "or:\n"
            "  pip install 'libero @ git+https://github.com/Lifelong-Robot-Learning/LIBERO.git'\n"
            f"\nOriginal error: {_LIBERO_IMPORT_ERROR}"
        )
        raise ImportError(msg)


# Task suite maximum episode steps (from LeRobot's LiberoEnv)
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# Observation/action dimensions (from LeRobot)
OBS_STATE_DIM = 8
ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0


class LiberoGym(Gym):
    """LIBERO benchmark environment wrapper.

    Wraps a single LIBERO task for evaluation with exact LeRobot compatibility.
    Handles optional imports gracefully and provides clear error messages.

    Example:
        >>> gym = LiberoGym(
        ...     task_suite="libero_object",
        ...     task_id=0,
        ...     camera_names=["agentview_image", "robot0_eye_in_hand_image"],
        ... )
        >>> obs, info = gym.reset(seed=42)
        >>> observation = gym.to_observation(obs)
    """

    # Camera name mapping (LeRobot convention)
    # Maps LIBERO camera names to our standard naming
    CAMERA_NAME_MAPPING: ClassVar[dict[str, str]] = {
        "agentview_image": "image",
        "robot0_eye_in_hand_image": "image2",
    }

    def __init__(
        self,
        *,
        task_suite: str,
        task_id: int,
        camera_names: list[str] | None = None,
        obs_type: str = "pixels_agent_pos",
        observation_height: int = 256,
        observation_width: int = 256,
        init_states: bool = True,
        episode_index: int = 0,
        num_steps_wait: int = 10,
        control_mode: str = "relative",
    ) -> None:
        """Initialize LIBERO gym environment.

        Args:
            task_suite: Suite name ("libero_spatial", "libero_object", etc.)
            task_id: Task index within suite (0-based)
            camera_names: Cameras to include ["agentview_image", "robot0_eye_in_hand_image"]
            obs_type: "pixels_agent_pos" for images + state, "pixels" for images only
            observation_height: Image height in pixels (default: 256)
            observation_width: Image width in pixels (default: 256)
            init_states: Whether to use pre-defined init states for reproducibility
            episode_index: Which init state to use (when init_states=True)
            num_steps_wait: Steps to wait after reset for stabilization (default: 10)
            control_mode: "relative" for delta actions, "absolute" for absolute control
        """
        _check_libero_available()

        # Store initialization parameters
        if camera_names is None:
            camera_names = ["agentview_image", "robot0_eye_in_hand_image"]

        self.task_suite_name = task_suite
        self.task_id = task_id
        self.camera_names = camera_names
        self.obs_type = obs_type
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.init_states = init_states
        self.episode_index = episode_index
        self.num_steps_wait = num_steps_wait
        self.control_mode = control_mode

        # Load LIBERO suite
        self.task_suite = self._load_suite(task_suite)

        # Get task details
        task = self.task_suite.get_task(self.task_id)
        self.task_name = task.name
        self.task_description = task.language

        # Load init states if requested
        self._init_states: np.ndarray | None = None
        self._init_state_id = episode_index
        if self.init_states:
            self._init_states = self._load_init_states(task)

        # Create environment
        self.env = self._create_env(task)

        # Set observation and action spaces
        self._setup_spaces()

        logger.info(
            "Created LiberoGym: %s task %d (%s)",
            self.task_suite_name,
            self.task_id,
            self.task_name,
        )

    def _setup_spaces(self) -> None:
        """Set up observation and action spaces.

        Raises:
            ValueError: If obs_type is unsupported.
        """
        images: dict[str, spaces.Space] = {}
        for cam in self.camera_names:
            images[self.CAMERA_NAME_MAPPING[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        if self.obs_type == "pixels":
            self.observation_space = spaces.Dict({"pixels": spaces.Dict(images)})
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "agent_pos": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(OBS_STATE_DIM,),
                        dtype=np.float64,
                    ),
                },
            )
        else:
            msg = f"obs_type '{self.obs_type}' not supported. Use 'pixels' or 'pixels_agent_pos'"
            raise ValueError(msg)

        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

    @staticmethod
    def _load_suite(name: str) -> Benchmark:
        """Load LIBERO task suite by name.

        Args:
            name: Name of the LIBERO suite to load.

        Returns:
            The loaded LIBERO benchmark suite.

        Raises:
            ValueError: If suite name is unknown or suite has no tasks.
        """
        bench = libero_benchmark.get_benchmark_dict()
        if name not in bench:
            available = ", ".join(sorted(bench.keys()))
            msg = f"Unknown LIBERO suite '{name}'. Available: {available}"
            raise ValueError(msg)
        suite = bench[name]()
        if not getattr(suite, "tasks", None):
            msg = f"Suite '{name}' has no tasks"
            raise ValueError(msg)
        return suite

    @staticmethod
    def _load_init_states(task: Any) -> np.ndarray:  # noqa: ANN401
        """Load pre-defined initial states for reproducibility.

        Args:
            task: Task object with problem_folder and init_states_file attributes.

        Returns:
            Array of pre-defined initial states for the current task.
        """
        init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        # nosemgrep
        return torch.load(init_states_path, weights_only=False)  # nosec B614

    def _create_env(self, task: Any) -> OffScreenRenderEnv:  # type: ignore[name-defined]  # noqa: ANN401
        """Create the LIBERO environment.

        Args:
            task: Task object with name and problem_folder attributes.

        Returns:
            Configured LIBERO environment.
        """
        bddl_path = Path(get_libero_path("bddl_files"))
        task_bddl_file = str(bddl_path / task.problem_folder / task.bddl_file)

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.observation_height,
            "camera_widths": self.observation_width,
        }

        env = OffScreenRenderEnv(**env_args)
        env.reset()
        return env

    @staticmethod
    def _get_dummy_action() -> list[float]:
        """Get no-op action for settling physics after reset.

        Returns:
            7-dim action with zeros and gripper close command.
        """
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

    def _apply_control_mode(self) -> None:
        """Apply control mode settings to robot controllers."""
        use_delta = self.control_mode == "relative"
        for robot in self.env.robots:
            robot.controller.use_delta = use_delta

    def reset(
        self,
        *,
        seed: int | None = None,
        **reset_kwargs: Any,  # noqa: ANN401, ARG002
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset environment with optional init state.

        Follows LeRobot's reset logic exactly for compatibility.

        Args:
            seed: Random seed for reproducibility
            **reset_kwargs: Additional reset options (unused, for base class compatibility)

        Returns:
            Tuple of (Observation, info_dict)
        """
        if seed is not None:
            self.env.seed(seed)

        # Apply init state if available
        if self.init_states and self._init_states is not None:
            self.env.set_init_state(self._init_states[self._init_state_id])

        raw_obs = self.env.reset()

        # After reset, objects may be unstable (slightly floating, intersecting, etc.).
        # Step the simulator with a no-op action for a few frames so everything settles.
        dummy_action = self._get_dummy_action()
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self.env.step(dummy_action)

        # Apply control mode settings
        self._apply_control_mode()

        formatted_obs = self._format_raw_obs(raw_obs)
        observation = self.to_observation(formatted_obs)
        info = {"is_success": False, "task": self.task_name, "task_id": self.task_id}

        return observation, info

    def step(
        self,
        action: np.ndarray | torch.Tensor,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action to execute (7-dim continuous control)

        Returns:
            Tuple of (Observation, reward, terminated, truncated, info)

        Raises:
            ValueError: If action has wrong dimensions.
        """
        # Convert tensor to numpy if needed
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        # Validate action shape
        if action.ndim != 1:
            msg = f"Expected 1-D action (shape ({ACTION_DIM},)), got shape {action.shape}"
            raise ValueError(msg)

        raw_obs, reward, done, info = self.env.step(action)
        formatted_obs = self._format_raw_obs(raw_obs)
        observation = self.to_observation(formatted_obs)

        # Check for success (follows LeRobot pattern)
        # Cast numpy bools to Python bools for type consistency
        is_success = bool(self.env.check_success())
        terminated = bool(done) or is_success
        truncated = False

        info.update({
            "task": self.task_name,
            "task_id": self.task_id,
            "done": bool(done),
            "is_success": is_success,
        })

        return observation, float(reward), terminated, truncated, info

    def _format_raw_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Format raw LIBERO observation to dict format.

        This matches LeRobot's observation format exactly.

        Args:
            raw_obs: Raw observation from LIBERO environment

        Returns:
            Formatted observation dict with 'pixels' and optionally 'agent_pos'
        """
        # Process images (rotate 180° to match LeRobot convention)
        images = {}
        for camera_name in self.camera_names:
            image = raw_obs[camera_name]
            # Rotate 180 degrees (copy to avoid negative strides)
            image = image[::-1, ::-1].copy()
            images[self.CAMERA_NAME_MAPPING[camera_name]] = image

        # For pixels-only mode, skip state processing
        if self.obs_type == "pixels":
            return {"pixels": images.copy()}

        # Process state (eef position + orientation + gripper)
        # Matches LeRobot's state construction exactly
        state = np.concatenate(
            (
                raw_obs["robot0_eef_pos"],
                quat2axisangle(raw_obs["robot0_eef_quat"]),
                raw_obs["robot0_gripper_qpos"],
            ),
        )

        # pixels_agent_pos mode
        return {
            "pixels": images.copy(),
            "agent_pos": state,
        }

    def render(self) -> np.ndarray:
        """Render the environment for visualization.

        Returns:
            RGB image array

        Note:
            Uses private method _get_observations() as there's no public API.
            This is the standard pattern in LIBERO/LeRobot codebases.
        """
        # OffScreenRenderEnv wraps a robosuite environment (double nesting)
        # Using _get_observations() is the standard LIBERO pattern - no public alternative
        raw_obs = self.env.env._get_observations()  # noqa: SLF001
        return self._format_raw_obs(raw_obs)["pixels"]["image"]

    def close(self) -> None:
        """Close the environment and release resources.

        Handles LIBERO's close() bug where calling it twice raises AttributeError.
        """
        if hasattr(self, "env") and self.env is not None:
            # LIBERO has a bug where close() can't be called twice
            # (OffScreenRenderEnv loses self.env after first close)
            with contextlib.suppress(AttributeError):
                self.env.close()
            self.env = None

    def sample_action(self) -> torch.Tensor:
        """Sample a random valid action.

        Returns:
            Tensor of shape (ACTION_DIM,) with values in [-1, 1].
        """
        action = self.action_space.sample()
        return torch.from_numpy(action).float()

    def to_observation(
        self,
        raw_obs: dict[str, Any],
        camera_keys: list[str] | None = None,
    ) -> Observation:
        """Convert LIBERO observation to Observation dataclass.

        This method converts the LIBERO dict format to our standardized
        Observation dataclass with proper tensor formatting.

        Args:
            raw_obs: Raw observation dict (output from reset/step after _format_raw_obs)
            camera_keys: Camera names to include (uses CAMERA_NAME_MAPPING values if None)

        Returns:
            Observation dataclass with images and optionally state

        Examples:
            >>> obs_dict, info = gym.reset(seed=42)
            >>> observation = gym.to_observation(obs_dict)
            >>> observation.images["image"].shape  # (1, C, H, W)
            torch.Size([1, 3, 256, 256])
        """
        if camera_keys is None:
            camera_keys = list(self.CAMERA_NAME_MAPPING.values())

        # Extract images from pixels dict (LeRobot convention)
        images = {}
        if "pixels" in raw_obs:
            for our_name in camera_keys:
                if our_name in raw_obs["pixels"]:
                    img = raw_obs["pixels"][our_name]

                    # Convert to torch tensor if needed
                    if not isinstance(img, torch.Tensor):
                        img = torch.from_numpy(img)

                    # Convert HWC → CHW if needed
                    if img.ndim == 3 and img.shape[-1] == 3:  # noqa: PLR2004
                        img = img.permute(2, 0, 1)  # (H, W, C) → (C, H, W)

                    # Normalize to [0, 1] float32 and add batch dimension
                    if img.dtype == torch.uint8:
                        img = img.float() / 255.0
                    img = img.unsqueeze(0)  # Add batch dim: (C, H, W) → (1, C, H, W)

                    images[our_name] = img

        # Build observation dict
        obs_dict: dict[str, Any] = {"images": images or None}

        # Add state if present (with batch dimension)
        if "agent_pos" in raw_obs:
            state = raw_obs["agent_pos"]
            if not isinstance(state, torch.Tensor):
                state = torch.from_numpy(state)
            state = state.float().unsqueeze(0)  # Add batch dim: (D,) → (1, D)
            obs_dict["state"] = state

        return Observation(**obs_dict)

    def get_max_episode_steps(self) -> int:
        """Return max steps for this task suite.

        Uses LeRobot's values exactly for compatibility.

        Returns:
            Maximum episode steps for this suite
        """
        return TASK_SUITE_MAX_STEPS.get(self.task_suite_name, 500)

    def check_success(self) -> bool:
        """Check if current episode is successful.

        Returns:
            True if the task has been completed successfully.
        """
        return bool(self.env.check_success())

    @property
    def max_episode_steps(self) -> int:
        """Maximum steps for this task suite (for compatibility with wrappers)."""
        return self.get_max_episode_steps()


def create_libero_gyms(
    task_suites: str | list[str],
    task_ids: list[int] | None = None,
    *,
    camera_names: list[str] | None = None,
    observation_height: int = 256,
    observation_width: int = 256,
    init_states: bool = True,
    obs_type: str = "pixels_agent_pos",
    num_steps_wait: int = 10,
    control_mode: str = "relative",
) -> list[LiberoGym]:
    """Create LiberoGym instances for LIBERO benchmark evaluation.

    Convenience function to create multiple gym environments across
    task suites. The returned list can be passed directly to any
    DataModule's val_gyms parameter.

    Args:
        task_suites: Suite name(s) - string for single, list for multi
        task_ids: Specific task IDs to include (None = all tasks)
        camera_names: Camera views to include
        observation_height: Image height in pixels (default: 256)
        observation_width: Image width in pixels (default: 256)
        init_states: Use predefined init states for reproducibility
        obs_type: "pixels_agent_pos" or "pixels"
        num_steps_wait: Steps to wait after reset for stabilization
        control_mode: "relative" for delta actions, "absolute" for absolute control

    Returns:
        List of LiberoGym instances ready for validation

    Raises:
        ValueError: If an unknown task suite is specified.

    Examples:
        Single suite, all tasks:

        >>> gyms = create_libero_gyms("libero_object")
        >>> len(gyms)  # 10 tasks
        10

        Multi-suite:

        >>> gyms = create_libero_gyms(
        ...     ["libero_spatial", "libero_object"],
        ... )
        >>> len(gyms)  # 20 tasks
        20

        Specific tasks only:

        >>> gyms = create_libero_gyms(
        ...     "libero_object",
        ...     task_ids=[0, 1, 2],
        ... )
        >>> len(gyms)  # 3 tasks
        3
    """
    _check_libero_available()

    if camera_names is None:
        camera_names = ["agentview_image", "robot0_eye_in_hand_image"]

    # Normalize to list
    if isinstance(task_suites, str):
        task_suites = [task_suites]

    gyms: list[LiberoGym] = []
    bench = libero_benchmark.get_benchmark_dict()

    for suite_name in task_suites:
        # Load suite
        if suite_name not in bench:
            available = ", ".join(sorted(bench.keys()))
            msg = f"Unknown LIBERO suite '{suite_name}'. Available: {available}"
            raise ValueError(msg)

        suite = bench[suite_name]()
        num_tasks = len(suite.tasks)

        # Determine which tasks (validate task IDs)
        if task_ids is not None:
            selected_tasks = []
            for tid in task_ids:
                if 0 <= tid < num_tasks:
                    selected_tasks.append(tid)
                else:
                    logger.warning(
                        "Task ID %d out of range [0, %d) for suite '%s', skipping",
                        tid,
                        num_tasks,
                        suite_name,
                    )
        else:
            selected_tasks = list(range(num_tasks))

        # Create gym for each task
        for task_id in selected_tasks:
            gym = LiberoGym(
                task_suite=suite_name,
                task_id=task_id,
                camera_names=camera_names,
                observation_height=observation_height,
                observation_width=observation_width,
                init_states=init_states,
                obs_type=obs_type,
                num_steps_wait=num_steps_wait,
                control_mode=control_mode,
            )
            gyms.append(gym)

    logger.info(
        "Created %d LIBERO gym(s) across %d suite(s)",
        len(gyms),
        len(task_suites),
    )
    return gyms


__all__ = ["ACTION_DIM", "TASK_SUITE_MAX_STEPS", "LiberoGym", "create_libero_gyms"]
