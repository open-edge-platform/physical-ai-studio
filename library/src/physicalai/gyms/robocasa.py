# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa Kitchen benchmark environment wrapper with optional dependencies.

Ported from ``lerobot/src/lerobot/envs/robocasa.py`` with PAS-native types
and the lerobot ``AsyncVectorEnv`` plumbing removed. The Studio benchmark
loop drives episodes single-environment-at-a-time and resets explicitly
between rollouts (see ``LiberoBenchmark``), so the upstream
auto-reset-on-terminal step is also dropped.

Task list resolution mirrors lerobot's and accepts robocasa's own
dataset_registry group keywords (``atomic_seen``, ``composite_seen``,
``composite_unseen``, ``pretrain50``/``100``/``200``/``300``). Pinned to
robocasa SHA ``56e355c``. Eval-specific task slices (e.g. the RLDX-1
paper-parity subset) live alongside the benchmark that consumes them,
not here.

Example:
    Single gym for one task::

        from physicalai.gyms import RoboCasaGym

        gym = RoboCasaGym(task="CloseFridge")
        obs, info = gym.reset(seed=0)
        next_obs, reward, terminated, truncated, info = gym.step(gym.sample_action())

    Multiple gyms for a task group::

        from physicalai.gyms import create_robocasa_gyms

        gyms = create_robocasa_gyms(tasks="atomic_seen")  # 18 v1.0 tasks
        assert len(gyms) == 18

Note:
    Requires optional dependencies. Install into a dedicated venv via
    ``library/scripts/benchmark/install_robocasa.sh`` — there is intentionally no
    ``[robocasa]`` extra in ``pyproject.toml`` because the dep graph is
    unsolvable in a single uv resolve (lerobot, robosuite, tianshou
    pins all conflict). See §7.4 Step 1 for details.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces

from physicalai.data.observation import Observation
from physicalai.gyms.base import Gym

__all__ = [
    "ACTION_DIM",
    "DEFAULT_CAMERAS",
    "DEFAULT_OBJ_REGISTRIES",
    "OBS_STATE_DIM",
    "RoboCasaGym",
    "convert_action",
    "create_robocasa_gyms",
]

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)

# Optional imports - checked at runtime.
# Imports are deferred to call sites (guarded by _check_robocasa_available) to
# keep the top-level import clean when robocasa is not installed.
_ROBOCASA_AVAILABLE = False
_ROBOCASA_IMPORT_ERROR: str | None = None

try:
    import robocasa  # noqa: F401
    import robosuite  # noqa: F401

    _ROBOCASA_AVAILABLE = True
except ImportError as e:
    _ROBOCASA_IMPORT_ERROR = str(e)


def _check_robocasa_available() -> None:
    """Check if RoboCasa is available and raise a helpful error if not.

    Raises:
        ImportError: If RoboCasa or robosuite is not installed.
    """
    if not _ROBOCASA_AVAILABLE:
        msg = (
            "RoboCasa is not installed. The dep graph is unsolvable in a single uv "
            "resolve (lerobot, robosuite, tianshou pins all conflict), so RoboCasa "
            "is NOT a pyproject extra. Install into a dedicated venv:\n"
            "  uv venv .venv-robocasa\n"
            "  source .venv-robocasa/bin/activate\n"
            "  uv sync --extra cu128\n"
            "  bash library/scripts/benchmark/install_robocasa.sh\n"
            "Then download kitchen assets:\n"
            "  yes y | python -m robocasa.scripts.download_kitchen_assets "
            "--type tex tex_generative fixtures_lw objs_lw\n"
            f"\nOriginal error: {_ROBOCASA_IMPORT_ERROR}"
        )
        raise ImportError(msg)


# Dimensions for the flat action/state vectors. These correspond to the
# PandaOmron robot in RoboCasa v1.0.
OBS_STATE_DIM = 16  # base_pos(3) + base_quat(4) + ee_pos_rel(3) + ee_quat_rel(4) + gripper_qpos(2)
ACTION_DIM = 12  # base_motion(4) + control_mode(1) + ee_pos(3) + ee_rot(3) + gripper(1)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# Default PandaOmron cameras. Raw RoboCasa names are surfaced verbatim as
# `Observation.images["robot0_*"]` so the keys match the upstream RoboCasa
# dataset exactly. Per-policy renames go through the
# RLDX_CAMERA_REMAP_KITCHEN adapter at policy-input time, not here.
DEFAULT_CAMERAS: tuple[str, ...] = (
    "robot0_agentview_left",
    "robot0_eye_in_hand",
    "robot0_agentview_right",
)

# Object-mesh registries to sample from. The objaverse pack is huge
# (~30 GB) and not on disk in our setup; sampling from a registry whose
# category has zero candidates crashes with `Probabilities contain NaN`
# (0/0 in the normalization). Restrict to lightwheel only.
DEFAULT_OBJ_REGISTRIES: tuple[str, ...] = ("lightwheel",)

# Task-group shortcuts accepted as `task=`. Single task names (or a
# comma-separated list) take precedence; this only triggers on an exact
# group-name match. All groups resolve via robocasa's own
# dataset_registry; eval-specific subsets (e.g. paper-parity slices)
# live alongside the benchmark that consumes them, not here.
_TASK_GROUP_SPLITS: dict[str, str] = {
    "atomic_seen": "target",
    "composite_seen": "target",
    "composite_unseen": "target",
    "pretrain50": "pretrain",
    "pretrain100": "pretrain",
    "pretrain200": "pretrain",
    "pretrain300": "pretrain",
}


def _resolve_tasks(task: str) -> tuple[list[str], str | None]:
    """Resolve a ``task`` value to ``(task_names, split_override)``.

    If ``task`` is a known group keyword (e.g. ``atomic_seen``,
    ``pretrain100``), expand it via ``robocasa.utils.dataset_registry``
    and return the matching split. Otherwise treat ``task`` as an explicit
    task name (or comma-separated list) and leave the split untouched
    (``None``).

    Args:
        task: Group keyword or one-or-more comma-separated task names.

    Returns:
        Tuple of ``(list[task_name], split_or_None)``.

    Raises:
        ValueError: If ``task`` is empty, or if it is a recognized group
            keyword but missing from the installed robocasa registry.
    """
    key = task.strip()

    if key in _TASK_GROUP_SPLITS:
        from robocasa.utils.dataset_registry import PRETRAINING_TASKS, TARGET_TASKS  # noqa: PLC0415

        combined = {**TARGET_TASKS, **PRETRAINING_TASKS}
        if key not in combined:
            msg = f"Task group '{key}' is not available in this version of robocasa. Known groups: {sorted(combined)}."
            raise ValueError(msg)
        return list(combined[key]), _TASK_GROUP_SPLITS[key]

    names = [t.strip() for t in key.split(",") if t.strip()]
    if not names:
        msg = "`task` must contain at least one RoboCasa task name."
        raise ValueError(msg)
    return names, None


def convert_action(flat_action: np.ndarray) -> dict[str, np.ndarray]:
    """Split a flat ``(12,)`` action vector into a RoboCasa action dict.

    Layout: ``base_motion(4) + control_mode(1) + ee_pos(3) + ee_rot(3) + gripper(1)``.

    Args:
        flat_action: 1-D array of shape ``(ACTION_DIM,)``.

    Returns:
        Dict with the five RoboCasa action keys.
    """
    return {
        "action.base_motion": flat_action[0:4],
        "action.control_mode": flat_action[4:5],
        "action.end_effector_position": flat_action[5:8],
        "action.end_effector_rotation": flat_action[8:11],
        "action.gripper_close": flat_action[11:12],
    }


class RoboCasaGym(Gym):
    """RoboCasa Kitchen environment wrapper.

    Wraps a single RoboCasa kitchen task for evaluation. Observations come
    out as ``Observation`` with raw RoboCasa camera names (`robot0_*`) in
    ``images`` and a 16-D ``state`` vector. Actions are 12-D flat
    ``torch.Tensor`` that get split into RoboCasa's action dict inside
    ``step()``.

    Example:
        >>> gym = RoboCasaGym(task="CloseFridge")
        >>> obs, info = gym.reset(seed=42)
        >>> obs.state.shape
        torch.Size([1, 16])
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        *,
        task: str,
        camera_names: Sequence[str] | None = None,
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_height: int = 256,
        observation_width: int = 256,
        split: str | None = None,
        episode_length: int | None = None,
        obj_registries: Sequence[str] | None = None,
    ) -> None:
        """Initialize a RoboCasa gym for one task.

        Args:
            task: Single RoboCasa task name (e.g. ``"CloseFridge"``).
                Group keywords are resolved by ``create_robocasa_gyms``, not
                here.
            camera_names: Camera views to include. Defaults to the three
                ``robot0_*`` cameras in ``DEFAULT_CAMERAS``.
            obs_type: ``"pixels_agent_pos"`` for images + 16-D state, or
                ``"pixels"`` for images only.
            render_mode: Passed through to RoboCasa (``"rgb_array"`` only).
            observation_height: Image height in pixels.
            observation_width: Image width in pixels.
            split: RoboCasa dataset split (``None``/``"all"``/
                ``"pretrain"``/``"target"``). When ``None``, the underlying
                env defaults to ``"all"`` (RoboCasa's own ``"test"`` default
                is rejected by ``create_env``).
            episode_length: Max steps per episode. Defaults to 1000 when
                ``None``.
            obj_registries: Object-mesh registries to sample from.
                Defaults to ``("lightwheel",)`` to avoid the
                ``Probabilities contain NaN`` crash when objaverse meshes
                are not on disk.
        """
        _check_robocasa_available()

        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.split = split
        self.obj_registries = tuple(obj_registries) if obj_registries is not None else DEFAULT_OBJ_REGISTRIES
        self.camera_names = list(camera_names) if camera_names is not None else list(DEFAULT_CAMERAS)

        self._max_episode_steps = episode_length if episode_length is not None else 1000

        # Deferred — the underlying MuJoCo env is created lazily on first
        # reset() so that constructing many RoboCasaGym instances stays
        # cheap (used by `create_robocasa_gyms` when expanding a task group).
        self._env: Any = None
        self.task_description = ""

        self._setup_spaces()

    def _setup_spaces(self) -> None:
        """Build observation/action spaces.

        Raises:
            ValueError: If ``obs_type`` is unsupported.
        """
        images = {
            cam: spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )
            for cam in self.camera_names
        }

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
                        dtype=np.float32,
                    ),
                },
            )
        else:
            msg = f"Unsupported obs_type '{self.obs_type}'. Use 'pixels' or 'pixels_agent_pos'."
            raise ValueError(msg)

        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

    def _ensure_env(self) -> None:
        """Create the underlying ``RoboCasaGymEnv`` on first use."""
        if self._env is not None:
            return

        from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv  # noqa: PLC0415

        # `RoboCasaGymEnv` defaults split="test", which `create_env`
        # rejects. Always pass a valid value.
        self._env = RoboCasaGymEnv(
            env_name=self.task,
            camera_widths=self.observation_width,
            camera_heights=self.observation_height,
            split=self.split if self.split is not None else "all",
            obj_registries=self.obj_registries,
        )

        ep_meta = self._env.env.get_ep_meta()
        self.task_description = ep_meta.get("lang", self.task)

    def _format_raw_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Format raw ``RoboCasaGymEnv`` output into a ``pixels[/agent_pos]`` dict.

        Kept separate from ``to_observation`` so the conversion is
        directly unit-testable without instantiating ``Observation``.

        Args:
            raw_obs: Raw dict from ``RoboCasaGymEnv.reset/step``. Contains
                ``video.<cam>`` and ``state.*`` keys.

        Returns:
            Dict with ``pixels: {cam: HWC uint8}`` and, when
            ``obs_type="pixels_agent_pos"``, ``agent_pos: (16,) float32``.
        """
        images = {cam: raw_obs[f"video.{cam}"] for cam in self.camera_names if f"video.{cam}" in raw_obs}

        if self.obs_type == "pixels":
            return {"pixels": images}

        # `state.*` keys come from PandaOmronKeyConverter inside the wrapper.
        agent_pos = np.concatenate(
            [
                raw_obs.get("state.base_position", np.zeros(3)),
                raw_obs.get("state.base_rotation", np.zeros(4)),
                raw_obs.get("state.end_effector_position_relative", np.zeros(3)),
                raw_obs.get("state.end_effector_rotation_relative", np.zeros(4)),
                raw_obs.get("state.gripper_qpos", np.zeros(2)),
            ],
            axis=-1,
        ).astype(np.float32)

        return {"pixels": images, "agent_pos": agent_pos}

    def to_observation(
        self,
        raw_obs: dict[str, Any],
        camera_keys: list[str] | None = None,
    ) -> Observation:
        """Wrap a ``_format_raw_obs`` dict into an ``Observation``.

        HWC→CHW transpose, uint8→float/255 normalisation, batch unsqueeze,
        and (optionally) ``task_description`` attachment — mirrors
        ``LiberoGym.to_observation``.

        Args:
            raw_obs: Output of ``_format_raw_obs`` (a ``pixels[/agent_pos]``
                dict).
            camera_keys: Camera names to include. Defaults to
                ``self.camera_names``.

        Returns:
            ``Observation`` with ``images``, optional ``state``, and
            optional ``task`` (the per-episode language description).
        """
        if camera_keys is None:
            camera_keys = self.camera_names

        images: dict[str, torch.Tensor] = {}
        if "pixels" in raw_obs:
            for cam in camera_keys:
                if cam not in raw_obs["pixels"]:
                    continue
                img = raw_obs["pixels"][cam]
                if not isinstance(img, torch.Tensor):
                    img = torch.from_numpy(img)
                if img.ndim == 3 and img.shape[-1] == 3:  # noqa: PLR2004
                    img = img.permute(2, 0, 1)  # HWC → CHW
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                images[cam] = img.unsqueeze(0)  # (C, H, W) → (1, C, H, W)

        obs_dict: dict[str, Any] = {"images": images or None}

        if self.task_description:
            obs_dict["task"] = [self.task_description]

        if "agent_pos" in raw_obs:
            state = raw_obs["agent_pos"]
            if not isinstance(state, torch.Tensor):
                state = torch.from_numpy(state)
            obs_dict["state"] = state.float().unsqueeze(0)  # (D,) → (1, D)

        return Observation(**obs_dict)

    def reset(
        self,
        *,
        seed: int | None = None,
        **reset_kwargs: Any,  # noqa: ANN401, ARG002
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment and return ``(Observation, info)``.

        Args:
            seed: Optional random seed forwarded to the MuJoCo env.
            **reset_kwargs: Ignored, present for ``Gym`` ABC compatibility.

        Returns:
            Tuple of ``(Observation, info_dict)``.
        """
        self._ensure_env()
        raw_obs, _ = self._env.reset(seed=seed)

        ep_meta = self._env.env.get_ep_meta()
        self.task_description = ep_meta.get("lang", self.task)

        formatted = self._format_raw_obs(raw_obs)
        observation = self.to_observation(formatted)
        info: dict[str, Any] = {"is_success": False, "task": self.task}
        return observation, info

    def step(
        self,
        action: torch.Tensor | np.ndarray,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Step the environment by one action.

        Args:
            action: 1-D action of shape ``(ACTION_DIM,)``.

        Returns:
            ``(Observation, reward, terminated, truncated, info)``.

        Raises:
            ValueError: If ``action`` is not 1-D or has the wrong length.
        """
        self._ensure_env()

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        if action.ndim != 1 or action.shape[0] != ACTION_DIM:
            msg = f"Expected 1-D action shape ({ACTION_DIM},), got shape {action.shape}"
            raise ValueError(msg)

        action_dict = convert_action(action)
        raw_obs, reward, done, truncated, info = self._env.step(action_dict)

        is_success = bool(info.get("success", False))
        terminated = bool(done) or is_success
        info = {**info, "task": self.task, "done": bool(done), "is_success": is_success}

        formatted = self._format_raw_obs(raw_obs)
        observation = self.to_observation(formatted)

        return observation, float(reward), terminated, bool(truncated), info

    def render(self) -> np.ndarray | None:
        """Render the current frame as an RGB array.

        Returns:
            ``(H, W, 3)`` uint8 array, or ``None`` if the env has not been
            created yet.
        """
        if self._env is None:
            return None
        return self._env.render()

    def close(self) -> None:
        """Close the underlying MuJoCo environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def sample_action(self) -> torch.Tensor:
        """Sample a random valid action.

        Returns:
            ``torch.Tensor`` of shape ``(ACTION_DIM,)`` in ``[-1, 1]``.
        """
        return torch.from_numpy(self.action_space.sample()).float()

    def get_max_episode_steps(self) -> int:
        """Return the configured per-episode step limit."""
        return self._max_episode_steps


def create_robocasa_gyms(
    tasks: str | Sequence[str],
    *,
    camera_names: Sequence[str] | None = None,
    obs_type: str = "pixels_agent_pos",
    observation_height: int = 256,
    observation_width: int = 256,
    split: str | None = None,
    episode_length: int | None = None,
    obj_registries: Sequence[str] | None = None,
) -> list[RoboCasaGym]:
    """Create one ``RoboCasaGym`` per resolved task name.

    Accepts either a group keyword (resolved via ``_resolve_tasks``) or an
    explicit list/comma-separated string of task names. RoboCasa tasks are
    named, not indexed — there is no ``task_ids`` parameter on purpose.

    Args:
        tasks: A group keyword (``"atomic_seen"``, ``"composite_seen"``,
            ``"composite_unseen"``, ``"pretrain50/100/200/300"``), a
            single task name, a comma-separated string of task names, or
            a list of task names.
        camera_names: Forwarded to each ``RoboCasaGym``.
        obs_type: Forwarded to each ``RoboCasaGym``.
        observation_height: Image height in pixels.
        observation_width: Image width in pixels.
        split: Forwarded to each ``RoboCasaGym``. When ``None`` and
            ``tasks`` is a group keyword, the group's natural split is
            used (e.g. ``"target"`` for ``"atomic_seen"``).
        episode_length: Forwarded to each ``RoboCasaGym``.
        obj_registries: Forwarded to each ``RoboCasaGym``.

    Returns:
        ``list[RoboCasaGym]``, one per resolved task name.

    Raises:
        ValueError: If ``tasks`` is empty or names an unknown task group.

    Examples:
        All v1.0 atomic tasks::

            gyms = create_robocasa_gyms("atomic_seen")
            assert len(gyms) == 18

        Explicit task list::

            gyms = create_robocasa_gyms(["CloseFridge", "OpenDrawer"])
            assert len(gyms) == 2
    """
    _check_robocasa_available()

    if isinstance(tasks, str):
        task_names, resolved_split = _resolve_tasks(tasks)
    else:
        task_names = [str(t).strip() for t in tasks if str(t).strip()]
        resolved_split = None
        if not task_names:
            msg = "`tasks` must contain at least one RoboCasa task name."
            raise ValueError(msg)

    effective_split = split if split is not None else resolved_split

    gyms = [
        RoboCasaGym(
            task=name,
            camera_names=camera_names,
            obs_type=obs_type,
            observation_height=observation_height,
            observation_width=observation_width,
            split=effective_split,
            episode_length=episode_length,
            obj_registries=obj_registries,
        )
        for name in task_names
    ]

    logger.info("Created %d RoboCasa gym(s) | split=%s | tasks=%s", len(gyms), effective_split, task_names)
    return gyms
