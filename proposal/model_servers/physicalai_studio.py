"""Physical AI Studio model server.

Bridges Studio policies for vla-eval: accepts either a direct `Policy` instance
(from physicalai.policies) or an `InferenceModel` (exported policy loaded via
physicalai.inference). Flexible observation key mapping for different policy
architectures.

Example usage:
    # Direct library policy
    from physicalai.policies import ACT
    policy = ACT()
    server = PhysicalAIStudioModelServer(
        policy=policy,
        observation_key_map={"images": "observation.images", "state": "observation.state"}
    )

    # Exported policy (InferenceModel)
    from physicalai.inference import InferenceModel
    inference_model = InferenceModel.load("path/to/exported")
    server = PhysicalAIStudioModelServer(
        policy=inference_model,
        observation_key_map={"images": "observation.images", "state": "observation.state"}
    )
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer


from physicalai.policies import Policy
from physicalai.inference import InferenceModel

logger = logging.getLogger(__name__)

# Type hints for the two supported policy types
# Lazy imports avoid hard dependencies
PolicyLike = Any  # Union[physicalai.policies.Policy, physicalai.inference.InferenceModel]


class PhysicalAIStudioModelServer(PredictModelServer):
    """Model server for Physical AI Studio policies.

    Supports both:
    1. Direct library policies: `physicalai.policies.Policy` instances
    2. Exported policies: `physicalai.inference.InferenceModel` instances

    Flexible observation key mapping bridges vla-eval's observation format
    to the policy's expected input format.
    """

    def __init__(
        self,
        policy: Policy | InferenceModel | str,
        observation_key_map: dict[str, str] | None = None,
        action_key: str = "action",
        state_key: str | None = None,
        language_key: str | None = None,
        image_resolution: tuple[int, int] | None = None,
        *,
        chunk_size: int | None = None,
        action_ensemble: str = "newest",
        use_select_action: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the model server.

        Args:
            policy: Either a physicalai.policies.Policy or physicalai.inference.InferenceModel instance.
                Must have ``predict_action_chunk()`` (or ``select_action()`` if use_select_action=True).
            observation_key_map: Maps vla-eval observation keys to policy input keys.
                Example: {"images": "observation.images", "state": "observation.state"}
                When None, uses sensible defaults (see _init_observation_key_map).
            action_key: Output key for actions in the returned Action dict (default: "action").
            state_key: Policy input key for proprioceptive state, or None for no state.
            language_key: Policy input key for language instructions, or None for no language.
            image_resolution: Optional (H, W) to resize images before passing to policy.
            chunk_size: Actions buffered per inference. None uses policy's n_action_steps.
            action_ensemble: How to combine multiple action predictions ("newest", "mean", etc.).
            use_select_action: Route through policy's stateful select_action() instead of
                predict_action_chunk() (for policies like LingBot-VA that feed observations back).
        """
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        
        self.policy = policy
        self.action_key = action_key
        self.state_key = None if state_key in (None, "None", "none") else state_key
        self.language_key = None if language_key in (None, "None", "none") else language_key
        self.image_resolution = image_resolution
        self.use_select_action = use_select_action

        # Detect policy type and initialize key mapping
        self._is_inference_model = self._detect_inference_model()
        if observation_key_map is None:
            self.observation_key_map = self._init_observation_key_map()
        else:
            self.observation_key_map = observation_key_map

        logger.info(
            "Initialized PhysicalAIStudioModelServer with %s",
            "InferenceModel" if self._is_inference_model else "Policy"
        )

    def _detect_inference_model(self) -> bool:
        """Check if policy is an InferenceModel (exported) vs Policy (library)."""
        policy_type = type(self.policy).__name__
        # InferenceModel classes have "InferenceModel" in their name or module
        return "InferenceModel" in policy_type or "inference" in str(type(self.policy)).lower()

    def _init_observation_key_map(self) -> dict[str, str]:
        """Default observation key mapping for common policy architectures."""
        obs_map = {}
        
        # Primary image input
        if self._is_inference_model:
            # Exported models often use "images" directly
            obs_map["images"] = "images"
        else:
            # Library policies typically nest under "observation.images"
            obs_map["images"] = "observation.images"
        
        # State and language
        if self.state_key:
            obs_map["state"] = self.state_key
        if self.language_key:
            obs_map["language"] = self.language_key
        
        return obs_map

    def _maybe_resize(self, img: np.ndarray) -> np.ndarray:
        """Resize image to target resolution if set and size differs."""
        if self.image_resolution is None:
            return img
        target_h, target_w = self.image_resolution
        if img.shape[:2] == (target_h, target_w):
            return img
        
        from PIL import Image
        pil = Image.fromarray(img)
        pil = pil.resize((target_w, target_h), Image.Resampling.BILINEAR)
        return np.asarray(pil)

    def _build_policy_observation(self, obs: Observation) -> dict[str, Any]:
        """Build policy input dict from vla-eval observation."""
        policy_obs: dict[str, Any] = {}
        
        # Map images
        images_key = self.observation_key_map.get("images", "images")
        if "images" in obs:
            images_dict = obs["images"]
            if isinstance(images_dict, dict):
                # Multiple cameras: extract values (order-dependent)
                img_list = list(images_dict.values())
            else:
                # Already a list or array
                img_list = [images_dict] if not isinstance(images_dict, (list, tuple)) else images_dict
            
            # Process first image
            if img_list:
                img = np.asarray(img_list[0], dtype=np.uint8)
                img = self._maybe_resize(img)
                policy_obs[images_key] = img
        
        # Map state
        if self.state_key:
            state_key = self.observation_key_map.get("state", self.state_key)
            # Policies may expect "state" or "states".
            # Avoid boolean evaluation on numpy arrays.
            raw_state = obs.get("states", None)
            if raw_state is None:
                raw_state = obs.get("state", None)
            if raw_state is not None:
                policy_obs[state_key] = np.asarray(raw_state, dtype=np.float32)
        
        # Map language
        if self.language_key:
            lang_key = self.observation_key_map.get("language", self.language_key)
            instruction = obs.get("task_description", "")
            policy_obs[lang_key] = instruction
        
        return policy_obs

    def get_observation_params(self) -> dict[str, Any]:
        """Declare what observations this server needs."""
        params: dict[str, Any] = {}
        if "images" in self.observation_key_map:
            params["send_images"] = True
        if self.state_key:
            params["send_state"] = True
        if self.language_key:
            params["send_language"] = True
        return params

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Return action output spec."""
        return {self.action_key: RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Return expected input observation spec."""
        spec: dict[str, DimSpec] = {"images": IMAGE_RGB}
        if self.state_key:
            spec["state"] = RAW
        if self.language_key:
            spec["language"] = LANGUAGE
        return spec

    def _get_policy_device_and_dtype(self) -> tuple[torch.device | None, torch.dtype | None]:
        """Infer policy device and floating dtype for input alignment."""
        device = getattr(self.policy, "device", None)
        dtype = getattr(self.policy, "dtype", None)

        if device is None or dtype is None:
            try:
                param = next(self.policy.parameters())
                if device is None:
                    device = param.device
                if dtype is None:
                    dtype = param.dtype
            except Exception:
                pass

        if device is not None and not isinstance(device, torch.device):
            device = torch.device(device)

        return device, dtype

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        """Run inference on a single observation."""
        policy_obs = self._build_policy_observation(obs)

        # physicalai Policy implementations typically expect Observation objects,
        # while exported InferenceModel expects plain dict inputs.
        policy_input: Any = policy_obs
        if not self._is_inference_model:
            from physicalai.data import Observation as PhysicalAIObservation

            if isinstance(policy_obs, PhysicalAIObservation):
                policy_input = policy_obs.to_torch()
            else:
                try:
                    policy_input = PhysicalAIObservation.from_dict(policy_obs).to_torch()
                except Exception as exc:
                    msg = (
                        "Failed to convert model-server observation into physicalai.data.Observation. "
                        f"Keys: {sorted(policy_obs.keys())}."
                    )
                    raise RuntimeError(msg) from exc

            # Align Observation tensors with policy device/dtype (e.g. bfloat16 on cuda).
            policy_device, policy_dtype = self._get_policy_device_and_dtype()
            if policy_device is not None:
                policy_input = policy_input.to(policy_device)

        policy_device, policy_dtype = self._get_policy_device_and_dtype()
        use_autocast = (
            policy_dtype in (torch.bfloat16, torch.float16)
            and isinstance(policy_device, torch.device)
            and policy_device.type in ("cuda", "cpu")
        )

        if use_autocast:
            with torch.autocast(device_type=policy_device.type, dtype=policy_dtype):
                if self.use_select_action:
                    # Stateful inference: one action at a time
                    action = self.policy.select_action(policy_input)
                else:
                    # Stateless inference: predict full action chunk
                    action = self.policy.predict_action_chunk(policy_input)
        else:
            if self.use_select_action:
                # Stateful inference: one action at a time
                action = self.policy.select_action(policy_input)
            else:
                # Stateless inference: predict full action chunk
                action = self.policy.predict_action_chunk(policy_input)
        
        # Move actions off device for transport.
        if isinstance(action, torch.Tensor):
            action = action.detach().to("cpu")
            if torch.is_floating_point(action):
                action = action.to(dtype=torch.float32)
            action = action.numpy()
        else:
            action = np.asarray(action, dtype=np.float32)

        # Most policies return a batch dimension (e.g. [1, 7]); vla-eval step
        # actions for this adapter should be a flat vector.
        while action.ndim > 1 and action.shape[0] == 1:
            action = action[0]
        if self.action_key != "actions" and action.ndim == 2:
            action = action[0]
        
        return {self.action_key: action}

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        """Reset policy state at episode start (e.g., for select_action)."""
        if self.use_select_action:
            if hasattr(self.policy, "reset"):
                self.policy.reset()
            else:
                logger.warning("Policy does not have reset() method but use_select_action=True")


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(PhysicalAIStudioModelServer)
