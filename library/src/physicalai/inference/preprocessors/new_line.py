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

    Attributes:
        new_line_char (str): The new line character to append. Defaults to "\n".
    """

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Add a new line character to the task description.

        Args:
            inputs: Dictionary containing TASK key with a string value.

        Returns:
            Dictionary with updated TASK value.
        """
        batch_tasks = inputs[TASK]
        if not isinstance(batch_tasks, list):
            raise ValueError(f"Expected TASK to be a list of strings, got {type(batch_tasks)}")

        for i, task in enumerate(batch_tasks):
            if not isinstance(task, str):
                raise ValueError(f"Expected TASK to be a string, got {type(task)}")
            if not task.endswith("\n"):
                batch_tasks[i] = task + "\n"

        inputs[TASK] = batch_tasks
        return inputs

