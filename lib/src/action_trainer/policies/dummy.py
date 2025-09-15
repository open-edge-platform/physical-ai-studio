# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dummy lightning module and policy for testing usage"""

import time
from collections import deque
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from action_trainer.gyms import BaseGym
from action_trainer.policies.base import ActionTrainerModule


def infer_batch_size(batch: dict[str, Any]) -> int:
    """Infer batch size from the first array/tensor-like entry in the batch."""
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            return v.shape[0]
        if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
            return len(v)
    msg = "Could not infer batch size from batch."
    raise ValueError(msg)


class DummyModel(nn.Module):
    """
    Useful for testing training/evaluation loops.
    """

    name = "dummy"

    def __init__(
        self,
        action_shape: torch.Size,
        n_action_steps: int = 1,
        temporal_ensemble_coeff: float | None = None,
    ):
        super().__init__()
        self.action_shape = action_shape
        self.n_action_steps = n_action_steps
        self.temporal_ensemble_coeff = temporal_ensemble_coeff

        if self.temporal_ensemble_coeff is not None:
            # simple placeholder for temporal ensemble
            self.temporal_buffer: None = None
        else:
            self._action_queue: deque = deque(maxlen=self.n_action_steps)

        # dummy parameter for optimizer and backward
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def reset(self):
        """Reset buffers (like ACTPolicy.reset)."""
        if self.temporal_ensemble_coeff is not None:
            self.temporal_buffer = None
        else:
            self._action_queue.clear()

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a single random action, managing queue like ACTPolicy."""
        self.eval()

        if self.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            # simple stand-in for ensembler: just take mean
            return actions.mean(dim=1)

        # Handle action queue logic
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a chunk of random actions (batch_size, n_action_steps, action_dim)."""
        batch_size = infer_batch_size(batch)
        return torch.randn((batch_size, self.n_action_steps, *self.action_shape))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        if self.training:
            batch_size = infer_batch_size(batch)
            # pred now depends on a parameter so it has grad_fn
            pred = (
                torch.randn((batch_size, self.n_action_steps, *self.action_shape), device=self.dummy_param.device)
                + self.dummy_param
            )
            target = torch.zeros_like(pred)
            return F.mse_loss(pred, target)
        return self.predict_action_chunk(batch)


class DummyPolicy(ActionTrainerModule):
    def __init__(self, action_shape: torch.Size | Iterable) -> None:
        super().__init__()
        self.action_shape = self._validate_action_shape(action_shape)

        # model
        self.model = DummyModel(self.action_shape)

    def _validate_action_shape(self, shape: torch.Size | Iterable) -> torch.Size:
        """Validate the action shape."""
        if shape is None:
            msg = "Action is missing a 'shape' key in its features dictionary."
            raise ValueError(msg)

        if isinstance(shape, torch.Size):
            return shape

        if isinstance(shape, str):
            msg = f"Shape for action '{shape}' must be a sequence of numbers, but received a string."
            raise TypeError(msg)

        if isinstance(shape, Iterable):
            return torch.Size(shape)

        msg = f"The 'action_shape' argument must be a torch.Size or Iterable, but received type {type(shape).__name__}."
        raise TypeError(msg)

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model.select_action(batch)

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """Defines the training step for the module.

        Args:
            batch (dict): The training batch.
            batch_idx (int): The index of the current batch.

        Returns:
            dict: A dictionary containing the loss.
        """
        loss = self.forward(batch)
        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return {"loss": loss}

    def configure_optimizers(self) -> torch.optim.Adam:
        """Configures the optimizer for the policy.

        Returns:
            torch.optim.Adam: The Adam optimizer.
        """
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def process_observation(self, obs: dict) -> dict:
        """Processes the raw observation from the environment.

        Args:
            obs (dict): The raw observation dictionary from the environment.

        Returns:
            dict: A processed observation dictionary with tensors.
        """
        state = torch.from_numpy(obs["agent_pos"]).to(torch.float32).unsqueeze(0).to(self.device)
        image = (
            (torch.from_numpy(obs["pixels"]).to(torch.float32).permute(2, 0, 1) / 255.0).unsqueeze(0).to(self.device)
        )
        return {"observation.state": state, "observation.image": image}

    def run_rollout(self, env: BaseGym) -> dict:
        """Runs a complete episode rollout in the environment.

        Args:
            env (gym.Env): The gymnasium environment to run the rollout in.

        Returns:
            dict: A dictionary containing rollout metrics like success, steps,
                  frames, and rewards.
        """
        obs, _ = env.reset()
        rewards = []
        inference_times = []
        frames = []
        done = False
        step_count = 0

        while not done and step_count < env.max_episode_steps:
            observation = self.process_observation(obs)
            frames.append(obs["pixels"].transpose(2, 0, 1))

            start_time = time.perf_counter()
            with torch.inference_mode():
                action = self.model.select_action(observation)
            inference_times.append(time.perf_counter() - start_time)

            action_np = action.squeeze(0).detach().cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action_np)
            rewards.append(reward)
            step_count += 1
            done = terminated or truncated

        return {
            "success": terminated,
            "steps": step_count,
            "frames": frames,
            "inference_times": inference_times,
            "sum_reward": sum(rewards) if rewards else 0,
            "max_reward": max(rewards) if rewards else 0,
        }

    def evaluation_step(self, batch: dict, stage: str) -> dict:
        """
        No logging happens here.
        """
        env = batch["env"]
        return self.run_rollout(env)

    def validation_step(self, batch: dict, batch_idx: int):
        """
        Defines the validation step.
        """
        return self.evaluation_step(batch=batch, stage="val")

    def test_step(self, batch: dict, batch_idx: int):
        """
        Defines the test step.
        """
        return self.evaluation_step(batch=batch, stage="test")
