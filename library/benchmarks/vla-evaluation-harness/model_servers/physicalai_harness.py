# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "vla-eval",
#     "physicalai-train[cu128,pi05]",
#     "numpy>=1.24",
#     "lerobot[dataset]",
# ]
#
# [tool.uv.sources]
# vla-eval = { git = "https://github.com/allenai/vla-evaluation-harness.git", tag = "v0.4.0" }
# physicalai-train = { path = "../../..", editable = true }
# torch = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv]
# exclude-newer = "2026-07-20T00:00:00Z"
# ///
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
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
    import yaml  # noqa: PLC0415

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

    def _build_policy_observation(self, obs: Observation) -> PhysicalAIObservation:
        """Build a ``physicalai.data.Observation`` from a vla-eval observation.

        Returns:
            Observation populated with batched numpy images / state / task.
        """
        from physicalai.data import Observation  # noqa: PLC0415

        images = obs.get("images", {}) or {}
        images_nested: dict[str, np.ndarray] = {}
        for bench_key, policy_slot in self._resolve_image_map(images).items():
            img = np.asarray(images[bench_key], dtype=np.uint8)
            images_nested[policy_slot] = img[None, ...] if img.ndim == 3 else img  # noqa: PLR2004

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
        policy_obs = self._build_policy_observation(obs)

        from physicalai.inference import InferenceModel  # noqa: PLC0415

        if isinstance(self._policy, InferenceModel):
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

    def get_action_spec(self) -> dict[str, DimSpec]:
        # Action convention is checkpoint-specific; declared RAW so the
        # orchestrator doesn't warn spuriously, same as the LeRobot bridge.
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {"image": RAW, "language": RAW}
        if self.state_key:
            spec["state"] = RAW
        return spec


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(PhysicalAIHarness)
