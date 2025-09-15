# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base Lightning Module for Policies"""

from __future__ import annotations

import time
from abc import ABC
from typing import TYPE_CHECKING

import lightning as L
import torch
from torch import nn

if TYPE_CHECKING:
    from action_trainer.gyms import BaseGym


class ActionTrainerModule(L.LightningModule, ABC):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model: nn.Module

    def forward(self, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        """Perform forward pass of the policy.
        The input batched is preprocessed before being passed to the model.

        Args:
            batch (dict[str, torch.Tensor]): Input batch
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            torch.Tensor: Model predictions
        """
        del args, kwargs
        return self.model(batch)

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

        while not done:
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
        Defines the evaluation step
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
