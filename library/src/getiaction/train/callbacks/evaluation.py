# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Gym evaluation callback for Lightning training.

This callback runs policy evaluation in gym environments during validation
steps, providing real-world performance metrics alongside training metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lightning.pytorch.callbacks import Callback

from getiaction.eval import rollout

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer

    from getiaction.gyms import Gym


class GymEvaluation(Callback):
    """Callback for evaluating policies in gym environments during training.

    This callback integrates with Lightning's validation loop to run policy
    rollouts in gym environments. It collects success rates, rewards, and
    episode lengths, logging them to the trainer's logger.

    The callback is automatically triggered during validation steps when
    the datamodule provides gym environments via `val_dataloader()`.

    Example:
        >>> from getiaction.train.callbacks import GymEvaluation
        >>> from getiaction.cli import Trainer
        >>> callback = GymEvaluation()
        >>> trainer = Trainer(callbacks=[callback])
        >>> trainer.fit(model, datamodule)
        # Evaluation metrics are logged during validation

    Note:
        The validation_step in the Policy should return the gym environment
        in a dict with key 'env': {'env': Gym}. This is handled
        automatically by the DataModule's val_dataloader collate function.
    """

    def __init__(self, max_steps: int | None = None) -> None:
        """Initialize the gym evaluation callback.

        Args:
            max_steps: Maximum steps per rollout. If None, uses the environment's
                default max_episode_steps.
        """
        super().__init__()
        self.max_steps = max_steps

    def on_validation_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,
        outputs: Any,  # noqa: ANN401
        batch: dict[str, Gym],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Run evaluation rollout after validation batch.

        This hook is called after each validation batch is processed. When
        the batch contains a gym environment, we run a complete rollout
        and log the metrics.

        Args:
            trainer: The Lightning trainer.
            pl_module: The Lightning module (policy).
            outputs: Outputs from validation_step (unused).
            batch: Batch data containing gym environment under 'env' key.
            batch_idx: Index of the batch.
            dataloader_idx: Index of the dataloader (for multiple val dataloaders).
        """
        del outputs, dataloader_idx  # Unused arguments

        # Check if batch contains a gym environment
        if not isinstance(batch, dict) or "env" not in batch:
            return

        env = batch["env"]

        # Run rollout with the policy
        result = rollout(
            env=env,
            policy=pl_module,  # type: ignore[arg-type]
            seed=batch_idx,  # Use batch_idx as seed for reproducibility
            max_steps=self.max_steps,
            return_observations=False,
        )

        # Log metrics
        metrics = {
            "val/gym/episode_length": result["episode_length"],
            "val/gym/sum_reward": result["sum_reward"],
            "val/gym/max_reward": result["max_reward"],
            "val/gym/success": float(result["is_success"]),
        }

        pl_module.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute aggregate metrics at the end of validation epoch.

        This hook aggregates all logged gym metrics across the validation
        epoch and computes success rate and average rewards.

        Args:
            trainer: The Lightning trainer.
            pl_module: The Lightning module (policy).
        """
        # Lightning automatically aggregates the metrics we logged above
        # We don't need to do anything additional here
