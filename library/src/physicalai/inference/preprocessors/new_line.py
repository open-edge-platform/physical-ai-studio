# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor that adds a new line character to text inputs."""

from __future__ import annotations

from typing import Any

from physicalai.inference.constants import TASK

from .base import Preprocessor


class NewLinePreprocessor(Preprocessor):
    """Preprocessor for adding a new line character to text inputs.

    This preprocessor appends a new line character to the task description.
    """

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Add a new line character to the task description.

        Args:
            inputs: Dictionary containing TASK key with a string value.

        Returns:
            Dictionary with updated TASK value.

        Raises:
            TypeError: If TASK is not a list of strings, or if any element in TASK is not a string.
        """
        batch_tasks = inputs[TASK]
        if not isinstance(batch_tasks, list):
            msg = f"Expected TASK to be a list of strings, got {type(batch_tasks)}"
            raise TypeError(msg)

        for i, task in enumerate(batch_tasks):
            if not isinstance(task, str):
                msg = f"Expected TASK to be a string, got {type(task)}"
                raise TypeError(msg)
            if not task.endswith("\n"):
                batch_tasks[i] = task + "\n"

        inputs[TASK] = batch_tasks
        return inputs
