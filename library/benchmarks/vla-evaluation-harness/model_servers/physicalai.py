# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vla-eval bridge for Physical AI Studio policies."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if __name__ == "__main__" and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

import numpy as np
from jsonargparse import ArgumentParser
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import run_server
from vla_eval.specs import RAW, DimSpec

from physicalai.data import Observation as PhysicalAIObservation
from physicalai.inference import InferenceModel
from physicalai.policies.base import Policy

if TYPE_CHECKING:
    from vla_eval.model_servers.base import SessionContext
    from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

_BATCHED_IMAGE_NDIM = 3
_BATCHED_ACTION_NDIM = 3


def _reshape_camera_image(image: np.ndarray, *, for_inference: bool) -> np.ndarray:
    """Batch and normalize one camera image for policy consumption.

    Returns:
        A batched image in policy-expected dtype and layout.
    """
    if image.ndim == _BATCHED_IMAGE_NDIM:
        image = image[None, ...]
    if for_inference:
        return image.astype(np.uint8, copy=False)
    return np.transpose(image, (0, 3, 1, 2)).astype(np.float32) / 255.0


def _format_actions(result: object) -> np.ndarray:
    """Convert policy output to float32 actions and remove singleton batch dim.

    Returns:
        Action array in the format expected by vla-eval.
    """
    actions = np.asarray(result, dtype=np.float32)
    # Unbatch [1, T, A] outputs to [T, A].
    if actions.ndim == _BATCHED_ACTION_NDIM and actions.shape[0] == 1:
        actions = actions[0]
    return actions


def _instantiate_policy(declaration: dict[str, Any]) -> Policy | InferenceModel:
    """Validate and instantiate an inline jsonargparse policy declaration.

    Returns:
        The configured policy or exported inference model.
    """
    parser = ArgumentParser()
    parser.add_subclass_arguments((Policy, InferenceModel), "policy", required=True)
    config = parser.parse_object({"policy": declaration})
    instantiated = parser.instantiate_classes(config)
    return cast("Policy | InferenceModel", instantiated["policy"])


class PhysicalAIModelServer(PredictModelServer):
    """Adapt a Physical AI policy to the vla-eval model-server protocol."""

    def __init__(
        self,
        policy: dict[str, Any] | None = None,
        image_keys: dict[str, str] | None = None,
        state_key: str | None = "state",
        device: str | None = None,
        *,
        chunk_size: int | None = None,
        action_ensemble: str = "newest",
        _policy: Policy | InferenceModel | None = None,
        **vla_eval_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the bridge from an inline declaration or policy object.

        Raises:
            ValueError: If exactly one policy construction path is not given.
        """
        if (policy is None) == (_policy is None):
            msg = "Pass exactly one of `policy` or `_policy`."
            raise ValueError(msg)

        self.image_keys = image_keys
        self.state_key = None if state_key in {None, "None", "none"} else state_key
        self.device = device
        self._logged_image_map = False
        self._policy = _policy if _policy is not None else _instantiate_policy(policy)  # type: ignore[arg-type]

        self._is_inference_model = isinstance(self._policy, InferenceModel)
        if not self._is_inference_model:
            if device:
                self._policy = self._policy.to(device)  # type: ignore[union-attr]
            self._policy.eval()  # type: ignore[union-attr]

        if chunk_size is None:
            chunk_size = getattr(self._policy, "chunk_size", None)
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **vla_eval_kwargs)

        self._expected_image_keys = list(getattr(self._policy, "image_keys", None) or [])
        logger.info("PhysicalAI model server is ready")

    def _resolve_image_map(self, images: dict[str, np.ndarray]) -> dict[str, str]:
        """Resolve benchmark camera names to policy image feature keys.

        Returns:
            A benchmark camera to policy feature mapping.
        """
        if self.image_keys is not None:
            resolved = {camera: key for camera, key in self.image_keys.items() if camera in images}
            missing = [camera for camera in self.image_keys if camera not in images]
        else:
            resolved = dict(zip(sorted(images), sorted(self._expected_image_keys), strict=False))
            missing = []

        if not self._logged_image_map:
            if missing:
                logger.warning(
                    "Configured cameras are absent from the observation: %s (available: %s)",
                    missing,
                    list(images),
                )
            logger.info("Image mapping (benchmark -> policy): %s", resolved)
            self._logged_image_map = True
        return resolved

    def _build_policy_observation(self, obs: Observation) -> PhysicalAIObservation:
        """Convert a vla-eval observation to a Physical AI observation.

        Returns:
            The batched observation expected by the policy.
        """
        source_images = obs.get("images", {}) or {}
        images: dict[str, np.ndarray] = {}
        for camera, feature_key in self._resolve_image_map(source_images).items():
            image = np.asarray(source_images[camera])
            images[feature_key] = _reshape_camera_image(image, for_inference=self._is_inference_model)

        state = None
        if self.state_key:
            raw_state = obs.get("states")
            if raw_state is None:
                raw_state = obs.get("state")
            if raw_state is not None:
                state = np.asarray(raw_state, dtype=np.float32)
                if state.ndim == 1:
                    state = state[None, :]

        task_description = obs.get("task_description")
        task = task_description if isinstance(task_description, str) else None
        # physicalai Observation typing for task/images is currently broader/narrower than runtime usage here.
        return PhysicalAIObservation(images=images or None, state=state, task=task)  # pyright: ignore[reportArgumentType]

    def _policy_device(self) -> str:
        """Return the device used by a live policy."""
        if self.device:
            return self.device
        policy_device = getattr(self._policy, "device", None)
        if policy_device is not None:
            return str(policy_device)
        try:
            return str(next(self._policy.parameters()).device)  # type: ignore[union-attr]
        except (AttributeError, StopIteration):
            return "cpu"

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        """Predict an action chunk for one vla-eval observation.

        Returns:
            A float32 NumPy action array under the ``actions`` key.

        Raises:
            ValueError: If a dictionary policy output has no action value.
        """
        del ctx
        policy_observation = self._build_policy_observation(obs)

        if self._is_inference_model:
            inputs = {
                key: value
                for key, value in {
                    "images": policy_observation.images,
                    "state": policy_observation.state,
                    "task": policy_observation.task,
                }.items()
                if value is not None
            }
            result = self._policy.predict_action_chunk(inputs)  # type: ignore[union-attr]
        else:
            policy_observation = policy_observation.to_torch(device=self._policy_device())
            result = self._policy.predict_action_chunk(policy_observation)  # type: ignore[union-attr]

        if isinstance(result, dict):
            result = result.get("actions", result.get("action"))
            if result is None:
                msg = "Policy output dictionary must contain `actions` or `action`."
                raise ValueError(msg)
        if hasattr(result, "detach"):
            result = cast("Any", result).detach().to("cpu").numpy()

        actions = _format_actions(result)
        return {"actions": actions}

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        """Reset stateful policies and vla-eval action buffers."""
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()
        await super().on_episode_start(config, ctx)

    def get_observation_params(self) -> dict[str, Any]:
        """Declare optional observations required from the benchmark.

        Returns:
            Observation flags understood by vla-eval.
        """
        params: dict[str, Any] = {}
        if self.state_key:
            params["send_state"] = True
        if len(self._expected_image_keys) > 1 or len(self.image_keys or {}) > 1:
            params["send_wrist_image"] = True
        return params

    def get_action_spec(self) -> dict[str, DimSpec]:  # noqa: PLR6301
        """Declare the policy's raw action format.

        Returns:
            The raw action specification.
        """
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Declare the observation fields consumed by the policy.

        Returns:
            The raw image, language, and optional state specifications.
        """
        spec: dict[str, DimSpec] = {"image": RAW, "language": RAW}
        if self.state_key:
            spec["state"] = RAW
        return spec


if __name__ == "__main__":
    run_server(PhysicalAIModelServer)
