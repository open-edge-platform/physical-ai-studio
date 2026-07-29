# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""vla-eval bridge for Physical AI Studio policies.

Config-driven: one class handles any Physical AI Studio policy or exported
``InferenceModel`` via a jsonargparse-style policy config (``class_path`` /
``init_args``).  No per-benchmark subclass is required — point
``policy_config`` at a YAML that instantiates the policy the same way
``physicalai fit --config`` does.

The policy config can target either:

* a ``physicalai.policies.Policy`` subclass (constructed via ``__init__``), or
* a ``physicalai.inference.InferenceModel`` (constructed via ``__init__`` with
  ``export_dir`` / ``device`` / …).

``jsonargparse.add_subclass_arguments`` accepts a tuple of base classes, so
both are admitted without any special-casing in this bridge.

Run directly in the active Python environment (no ``uv run`` / ``vla-eval serve``
wrapper required)::

    python model_servers/physicalai_harness.py \
        --config configs/pi05_libero_policy.yaml

CLI overrides work as usual::

    python model_servers/physicalai_harness.py \
        --config configs/pi05_libero_policy.yaml --port 8001 --args.device=cpu
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import RAW, DimSpec

if TYPE_CHECKING:
    from vla_eval.model_servers.base import SessionContext
    from vla_eval.types import Action, Observation

    from physicalai.data import Observation as PhysicalAIObservation
    from physicalai.inference import InferenceModel
    from physicalai.policies import Policy

logger = logging.getLogger(__name__)


def load_policy_from_config(policy_config: str) -> Policy | InferenceModel:
    """Instantiate a policy from a jsonargparse-style YAML config.

    The config must use the ``class_path`` / ``init_args`` pattern, e.g.::

        class_path: physicalai.inference.InferenceModel
        init_args:
          export_dir: /path/to/exported/model
          device: cuda

    Both flat (``class_path`` at the top level) and nested
    (``policy: {class_path, init_args}``) layouts are accepted.

    Works for any :class:`physicalai.policies.Policy` subclass or
    :class:`physicalai.inference.InferenceModel` because
    :meth:`jsonargparse.ArgumentParser.add_subclass_arguments` accepts a tuple
    of base classes.

    Args:
        policy_config: Path to a policy-only YAML config.

    Returns:
        Instantiated policy (a ``Policy`` or ``InferenceModel``).
    """
    from jsonargparse import ArgumentParser  # noqa: PLC0415

    from physicalai.inference import InferenceModel  # noqa: PLC0415
    from physicalai.policies.base import Policy  # noqa: PLC0415

    parser = ArgumentParser()
    parser.add_subclass_arguments((Policy, InferenceModel), "policy", required=True)

    # Accept flat (class_path/init_args at top level) as well as nested
    # (policy: {class_path, init_args}) layouts by normalising before parse.
    with open(policy_config, encoding="utf-8") as f:  # noqa: PTH123
        raw = yaml.safe_load(f)
    if isinstance(raw, dict) and "class_path" in raw:
        raw = {"policy": raw}
    cfg = parser.parse_object(raw)
    init = parser.instantiate_classes(cfg)
    return init.policy


class PhysicalAIHarness(PredictModelServer):
    """Bridge from a Physical AI Studio policy to a vla-eval model server.

    The policy is loaded from a jsonargparse-style YAML config (``class_path``
    / ``init_args``) that targets either a ``physicalai.policies.Policy``
    subclass or a ``physicalai.inference.InferenceModel`` export directory.

    CLI path: ``policy_config`` is a path to a policy-only YAML — see
    ``configs/policies/*.yaml`` for examples.
    Python API path: pass an already-built ``_policy`` and skip
    ``policy_config`` entirely (the parameter is underscore-prefixed so
    ``run_server``'s argparse auto-discovery skips it).
    """

    def __init__(
        self,
        policy_config: str | None = None,
        image_keys: dict[str, str] | None = None,
        state_key: str | None = "state",
        action_key: str = "action",
        device: str | None = None,
        *,
        chunk_size: int | None = None,
        action_ensemble: str = "newest",
        _policy: Policy | InferenceModel | None = None,
        **vla_eval_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the model server.

        Args:
            policy_config: Path to a policy-only YAML using ``class_path`` /
                ``init_args``.  Mutually exclusive with ``_policy``.
            image_keys: Map from benchmark camera name (key in
                ``obs["images"]``) to the policy's image feature slot.
                When ``None``, benchmark cameras are mapped positionally
                (sorted) onto the policy's declared image keys.
            state_key: Proprioceptive state feature flag.  Pass ``None`` (or
                the strings ``"None"`` / ``"none"``) to send no state.
            action_key: Output key to read from the policy's return value
                when the policy returns a dict (unused for ``InferenceModel``,
                which returns a plain array).
            device: Torch device for ``Policy`` subclasses (moved after load
                via ``.to(device)``).  ``InferenceModel`` manages its own
                device through ``init_args`` and is left untouched.
            chunk_size: Actions buffered per inference.  ``None`` (default)
                uses the policy's ``chunk_size`` attribute when available.
            action_ensemble: How to combine multiple action predictions
                (``"newest"``, ``"mean"``, …).  Forwarded to
                ``PredictModelServer``.
            _policy: Already-built policy instance (Python API / subclass
                handoff).  Mutually exclusive with ``policy_config``.
            **vla_eval_kwargs: Extra keyword arguments forwarded to
                ``PredictModelServer``.

        Raises:
            ValueError: If neither or both of ``policy_config`` / ``_policy``
                are provided.
        """
        if (policy_config is None) == (_policy is None):
            msg = "Pass exactly one of `policy_config` or `_policy`."
            raise ValueError(msg)

        self.image_keys = image_keys
        self.state_key = None if state_key in {None, "None", "none"} else state_key
        self.action_key = action_key
        self.device = device
        self._logged_image_map = False

        if _policy is not None:
            self._policy = _policy
        else:
            logger.info("Loading policy from %s", policy_config)
            self._policy = load_policy_from_config(policy_config)  # type: ignore[arg-type]

        # Policy subclasses (LightningModules) are moved to the requested
        # device and set to eval mode.  InferenceModel manages its own device
        # via init_args and has no .to() / .eval().
        if device and hasattr(self._policy, "to"):
            self._policy = self._policy.to(device)
        if hasattr(self._policy, "eval"):
            self._policy.eval()

        if chunk_size is None:
            chunk_size = getattr(self._policy, "chunk_size", None)

        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **vla_eval_kwargs)

        self._expected_image_keys = list(getattr(self._policy, "image_keys", None) or [])

    # Observation mapping

    def _resolve_image_map(self, images: dict[str, np.ndarray]) -> dict[str, str]:
        """Map benchmark camera names to policy image feature slots.

        Returns:
            Dict mapping benchmark camera name to policy image key.
        """
        if self.image_keys:
            resolved = {b: k for b, k in self.image_keys.items() if b in images}
            missing = [b for b in self.image_keys if b not in images]
        else:
            # Positional fallback: sorted benchmark cameras onto sorted policy features.
            resolved = dict(zip(sorted(images), self._expected_image_keys, strict=False))
            missing = []
        if not self._logged_image_map:
            if missing:
                logger.warning(
                    "image_keys cameras absent from observation: %s (available: %s)",
                    missing,
                    list(images),
                )
            logger.info("Image mapping (benchmark -> policy slot): %s", resolved)
            self._logged_image_map = True
        return resolved

    def _build_policy_observation(
        self,
        obs: Observation,
        *,
        channels_first: bool,
    ) -> PhysicalAIObservation:
        """Build a ``physicalai.data.Observation`` from a vla-eval observation.

        Args:
            obs: vla-eval observation dict (images as HWC uint8).
            channels_first: When ``True`` (Policy path), convert images to
                ``(B, C, H, W)`` float32 in ``[0, 1]`` — the layout the
                training-time preprocessors expect.  When ``False``
                (InferenceModel path), keep images as ``(B, H, W, C)`` uint8
                — the exported preprocessors handle layout detection and
                normalisation internally.

        Returns:
            Observation populated with batched numpy images / state / task.
        """
        from physicalai.data import Observation  # noqa: PLC0415

        images = obs.get("images", {}) or {}
        images_nested: dict[str, np.ndarray] = {}
        for bench_key, policy_slot in self._resolve_image_map(images).items():
            img = np.asarray(images[bench_key])
            if img.ndim == 3:  # noqa: PLR2004
                img = img[None, ...]  # (H, W, C) → (1, H, W, C)

            # (B, H, W, C) uint8 → (B, C, H, W) float32 [0, 1]
            img = np.transpose(img, (0, 3, 1, 2)).astype(np.float32) / 255.0 if channels_first else img.astype(np.uint8)

            images_nested[policy_slot] = img

        state = None
        if self.state_key:
            raw_state = obs.get("states") if obs.get("states") is not None else obs.get("state")
            if raw_state is not None:
                state_arr = np.asarray(raw_state, dtype=np.float32)
                state = state_arr[None, :] if state_arr.ndim == 1 else state_arr

        instruction = obs.get("task_description")
        task = instruction if isinstance(instruction, str) else None

        return Observation(images=images_nested or None, state=state, task=task)

    # Inference

    def _policy_device(self) -> str:
        """Infer the torch device the policy weights live on.

        Returns:
            Device string (e.g. ``"cuda"``, ``"cpu"``).
        """
        if self.device:
            return self.device
        dev = getattr(self._policy, "device", None)
        if dev is not None:
            return str(dev)
        try:
            return str(next(self._policy.parameters()).device)  # type: ignore[attr-defined]
        except (StopIteration, AttributeError):
            return "cpu"

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        """Run inference on a single observation.

        Returns:
            Dict with key ``"actions"`` mapping to a ``(chunk_size, action_dim)``
            float32 numpy array.
        """
        del ctx

        from physicalai.inference import InferenceModel  # noqa: PLC0415

        is_inference = isinstance(self._policy, InferenceModel)
        # Policy subclasses expect (B, C, H, W) float32 [0, 1] images; the
        # exported InferenceModel preprocessors accept (B, H, W, C) uint8 and
        # handle layout/normalisation internally.
        policy_obs = self._build_policy_observation(obs, channels_first=not is_inference)

        if is_inference:
            # InferenceModel expects a plain numpy dict; build it from the
            # Observation fields, skipping None values so preprocessors don't
            # see unset features.
            inputs: dict[str, Any] = {}
            if policy_obs.images is not None:
                inputs["images"] = policy_obs.images
            if policy_obs.state is not None:
                inputs["state"] = policy_obs.state
            if policy_obs.task is not None:
                inputs["task"] = policy_obs.task
            raw = self._policy.predict_action_chunk(inputs)
            actions = np.asarray(raw, dtype=np.float32)
        else:
            # Policy subclasses expect a torch Observation on the policy's device.
            policy_obs = policy_obs.to_torch(device=self._policy_device())
            raw = self._policy.predict_action_chunk(policy_obs)
            if isinstance(raw, dict):
                raw = raw[self.action_key]
            actions = raw.detach().to("cpu").numpy().astype(np.float32)

        # Strip a leading batch dimension of 1 if present; vla-eval buffers
        # per session via PredictModelServer.
        if actions.ndim == 3:  # noqa: PLR2004
            actions = actions.squeeze(0)
        return {"actions": actions}

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        """Reset policy state at episode start (clears action queues, etc.)."""
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()
        await super().on_episode_start(config, ctx)

    # Interface declarations
    def get_observation_params(self) -> dict[str, Any]:
        """Declare what observations this server needs.

        Returns:
            Dict of observation flags sent to the benchmark orchestrator.
        """
        params: dict[str, Any] = {}
        if self.state_key:
            params["send_state"] = True
        if len(self._expected_image_keys) > 1 or len(self.image_keys or {}) > 1:
            params["send_wrist_image"] = True
        return params

    def get_action_spec(self) -> dict[str, DimSpec]:  # noqa: PLR6301
        """Declare the action output format of this model server.

        Returns:
            Mapping from action component name to dimension spec.
        """
        # Action convention is checkpoint-specific; declared RAW so the
        # orchestrator doesn't warn spuriously, same as the LeRobot bridge.
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Declare the observation input format this model server expects.

        Returns:
            Mapping from observation component name to dimension spec.
        """
        spec: dict[str, DimSpec] = {"image": RAW, "language": RAW}
        if self.state_key:
            spec["state"] = RAW
        return spec


def _run_current_env(server_cls: type[PredictModelServer]) -> None:
    """Run ``run_server`` after rewriting the YAML for current-env usage.

    ``vla-eval serve`` pops ``port`` / ``host`` out of the YAML's ``args:``
    block and strips the ``script:`` metadata key before invoking the server.
    ``run_server``'s own ``--config`` loader keeps them nested under ``args:``,
    which would pass ``port`` / ``host`` to the model-server ``__init__`` and
    fail.  This helper mirrors the ``vla-eval serve`` normalisation so the same
    config files work when the script is run directly with ``python``.

    The original config path is preserved for relative-path resolution by
    writing the sanitised copy next to it.
    """
    from vla_eval.model_servers.serve import run_server  # noqa: PLC0415

    pre_parser = argparse.ArgumentParser()
    pre_parser.add_argument("--config", required=True)
    pre_parser.add_argument("--port", type=int, default=None)
    pre_parser.add_argument("--host", default=None)
    known, remainder = pre_parser.parse_known_args()

    config_path = Path(known.config)
    with config_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # ``script`` was metadata for ``vla-eval serve``; drop it.
    raw.pop("script", None)

    args_block: dict[str, Any] = raw.setdefault("args", {})
    server_level: dict[str, Any] = {}
    for key in ("port", "host"):
        if key in args_block:
            server_level[key] = args_block.pop(key)

    # CLI flags take precedence over YAML values.
    if known.port is not None:
        server_level["port"] = known.port
    if known.host is not None:
        server_level["host"] = known.host

    # Keep the sanitised file next to the original so relative paths inside
    # the YAML (e.g. ``policy_config: configs/policies/...``) resolve the
    # same way ``vla-eval serve`` would have resolved them.
    config_dir = config_path.parent
    fd, temp_path_str = tempfile.mkstemp(
        suffix=".yaml",
        dir=config_dir,
        prefix=f"._sanitized_{server_cls.__name__.lower()}_",
    )
    temp_path = Path(temp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f)
    except Exception:
        os.close(fd)
        raise

    def _cleanup() -> None:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()

    atexit.register(_cleanup)

    new_argv = [sys.argv[0], "--config", str(temp_path)]
    new_argv.extend(f"--{key}={server_level[key]}" for key in ("port", "host") if key in server_level)
    new_argv.extend(remainder)
    sys.argv = new_argv

    run_server(server_cls)


if __name__ == "__main__":
    _run_current_env(PhysicalAIHarness)
