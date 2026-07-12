"""Pi0.5 model server for vla-eval benchmarking.

Wraps physicalai.policies.Pi05 for use with vla-eval benchmarks.
Reproduces the LIBERO benchmark using the studio's policy directly.
"""

import logging
from typing import Any

import numpy as np
import torch
from physicalai.inference.preprocessors import Preprocessor
from physicalai_studio import PhysicalAIStudioModelServer
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_AA,
    STATE_EEF_POS_AA_GRIP,
    DimSpec,
)

logger = logging.getLogger(__name__)


class EnsureChannelsFirst(Preprocessor):
    """Optional image layout callback: convert batched NHWC to NCHW."""

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = dict(inputs)
        for key, value in out.items():
            if key.startswith("images") and value.ndim == 4 and value.shape[-1] == 3 and value.shape[1] != 3:
                out[key] = np.transpose(value, (0, 3, 1, 2))
        return out


class Pi05LiberoBenchmarkServer(PhysicalAIStudioModelServer):
    """Pi0.5 model server optimized for LIBERO benchmarks.
    
    Loads the LeRobot pretrained Pi0.5 checkpoint and configures
    observation mapping for LIBERO tasks.
    """

    def __init__(
        self,
        pretrained_name_or_path: str = "lerobot/pi05_libero_finetuned_v044",
        chunk_size: int = 10,
        device: str | None = None,
        policy_dtype: str | None = None,
        enable_channels_first_callback: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize Pi0.5 server for LIBERO.
        
        Args:
            pretrained_name_or_path: HuggingFace model identifier or local path.
            chunk_size: Action chunk size (default 10 for LIBERO).
            device: Torch device for policy weights (e.g. "cuda", "cuda:0", "cpu").
            policy_dtype: Optional torch dtype name for policy weights
                ("bfloat16", "float16", "float32").
            **kwargs: Additional arguments passed to parent class.
        """
        from physicalai.policies.pi05 import Pi05

        logger.info(f"Loading Pi0.5 from: {pretrained_name_or_path}")
        
        # Load policy
        policy = Pi05(pretrained_name_or_path=pretrained_name_or_path)
        if device:
            logger.info("Moving Pi0.5 policy to device: %s", device)
            policy = policy.to(device)
        if policy_dtype:
            dtype_map: dict[str, torch.dtype] = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype_key = policy_dtype.strip().lower()
            if dtype_key not in dtype_map:
                allowed = ", ".join(sorted(dtype_map.keys()))
                msg = f"Unsupported policy_dtype '{policy_dtype}'. Supported values: {allowed}"
                raise ValueError(msg)
            target_dtype = dtype_map[dtype_key]
            logger.info("Casting Pi0.5 policy to dtype: %s", dtype_key)
            policy = policy.to(dtype=target_dtype)
        policy.eval()
        
        # Initialize model server with LIBERO-specific config
        super().__init__(
            policy=policy,
            # Pi05 uses observation.images.image (base) and observation.images.image2 (wrist)
            observation_key_map={
                "images": "observation.images",  # Studio policy expects nested dict
                "state": "observation.state",
            },
            state_key="observation.state",
            action_key="action",
            chunk_size=chunk_size,
            **kwargs,
        )
        self.pretrained_name_or_path = pretrained_name_or_path
        self.device = device
        self.policy_dtype = policy_dtype
        self.enable_channels_first_callback = enable_channels_first_callback
        self._ensure_channels_first = EnsureChannelsFirst() if enable_channels_first_callback else None

    def get_observation_params(self) -> dict[str, Any]:
        # Ask LIBERO to send proprio + wrist image, which matches pi05 LIBERO checkpoints.
        return {
            "send_state": True,
            "send_wrist_image": True,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        # LIBERO benchmark expresses actions as pos+rot+gripper; server returns a 7-D action vector.
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_POS,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        # Match LIBERO benchmark schema to avoid handshake mismatch warnings.
        return {
            "agentview": IMAGE_RGB,
            "wrist": IMAGE_RGB,
            "state": STATE_EEF_POS_AA_GRIP,
            "language": LANGUAGE,
        }

    def _build_policy_observation(self, obs: Any) -> dict[str, Any]:
        """Override to handle LIBERO's image structure.
        
        LIBERO provides images dict with keys like "agentview" and "wrist".
        PhysicalAI Pi05 expects top-level Observation fields:
        - images: dict[str, np.ndarray]
        - state: np.ndarray
        - task: str
        """
        policy_obs: dict[str, Any] = {}
        
        # Extract images
        images_dict = obs.get("images", {})
        if isinstance(images_dict, dict):
            # Get sorted list of images (agentview first, then wrist)
            img_list = [v for k, v in sorted(images_dict.items())]
        else:
            img_list = [images_dict] if not isinstance(images_dict, (list, tuple)) else images_dict
        
        # Build images dict expected by physicalai.data.Observation
        images_nested: dict[str, Any] = {}
        if len(img_list) > 0:
            img = np.asarray(img_list[0], dtype=np.uint8)
            if img.ndim == 3:
                img = img[None, ...]
            images_nested["image"] = img
        if len(img_list) > 1:
            img = np.asarray(img_list[1], dtype=np.uint8)
            if img.ndim == 3:
                img = img[None, ...]
            images_nested["image2"] = img

        if self._ensure_channels_first is not None and images_nested:
            flat_images = {f"images.{k}": v for k, v in images_nested.items()}
            flat_images = self._ensure_channels_first(flat_images)
            images_nested = {k.split(".", 1)[1]: v for k, v in flat_images.items()}
        
        # Store under top-level "images" field.
        if images_nested:
            policy_obs["images"] = images_nested
        
        # Add state
        if self.state_key:
            raw_state = obs.get("states", None)
            if raw_state is None:
                raw_state = obs.get("state", None)
            if raw_state is not None:
                state = np.asarray(raw_state, dtype=np.float32)
                # physicalai Pi05 preprocessor expects a batch dimension [B, D].
                if state.ndim == 1:
                    state = state[None, :]
                policy_obs["state"] = state

        # Pass language instruction under the canonical "task" field.
        instruction = obs.get("task_description")
        if isinstance(instruction, str):
            policy_obs["task"] = instruction
        
        return policy_obs


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(Pi05LiberoBenchmarkServer)
