# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for documentation code examples.

This module uses pytest-examples to extract and test Python code blocks
from markdown documentation files. Code blocks can be marked with special
comments to control test behavior:

    ```python
    # This code will be tested
    from getiaction.policies import ACT
    policy = ACT()
    ```

    ```python test="skip" reason="requires checkpoint"
    # This code will be skipped
    policy = ACT.load_from_checkpoint("path/to/model.ckpt")
    ```

Run with:
    pytest tests/test_docs.py -v

Or run only doc tests:
    pytest tests/test_docs.py -v -m "not slow"

Note:
    The explanation/ folder contains design documentation with pseudocode
    examples that are not meant to be executed. These are automatically
    skipped by this test module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_examples import CodeExample, EvalExample

# Find all markdown files in docs directory
DOCS_DIR = Path(__file__).parent.parent / "docs"
ROOT_README = Path(__file__).parent.parent / "README.md"

# Folders containing design docs with pseudocode (not meant to be executed)
DESIGN_DOC_FOLDERS = {"explanation"}


def find_markdown_files() -> list[Path]:
    """Find all markdown files to test."""
    files = list(DOCS_DIR.rglob("*.md"))
    if ROOT_README.exists():
        files.append(ROOT_README)
    return files


def get_example_id(example: "CodeExample") -> str:
    """Generate a readable test ID for an example."""
    # Get relative path from library root
    try:
        rel_path = example.path.relative_to(Path(__file__).parent.parent)
    except ValueError:
        rel_path = example.path.name
    return f"{rel_path}:{example.start_line}"


def is_design_doc(example: "CodeExample") -> bool:
    """Check if an example is from a design documentation folder.

    Design docs contain pseudocode and interface definitions that
    are not meant to be executed as tests.
    """
    try:
        rel_path = example.path.relative_to(DOCS_DIR)
        # Check if first part of path is a design doc folder
        return rel_path.parts[0] in DESIGN_DOC_FOLDERS
    except (ValueError, IndexError):
        return False


try:
    from pytest_examples import CodeExample, EvalExample, find_examples

    # Collect examples from all markdown files
    EXAMPLES = list(find_examples(str(DOCS_DIR)))
    if ROOT_README.exists():
        EXAMPLES.extend(find_examples(str(ROOT_README)))

    @pytest.mark.parametrize(
        "example",
        EXAMPLES,
        ids=get_example_id,
    )
    def test_docs_python_examples(example: CodeExample, eval_example: EvalExample) -> None:
        """Test Python code examples in documentation.

        This test extracts Python code blocks from markdown files and runs them
        to verify they work correctly. Examples can be skipped by adding
        test="skip" to the code fence.

        Args:
            example: The code example extracted from markdown.
            eval_example: Helper for evaluating the example.
        """
        # Get prefix settings (test="skip", reason="...", etc.)
        settings = example.prefix_settings()

        # Skip examples marked with test="skip"
        if settings.get("test") == "skip":
            reason = settings.get("reason", "marked as skip")
            pytest.skip(f"Example skipped: {reason}")

        # Skip design documentation (pseudocode, interface definitions)
        if is_design_doc(example):
            pytest.skip("Design documentation (pseudocode/interface examples)")

        # Get the language from the prefix (e.g., "python", "bash", "yaml")
        # The prefix contains the info string after the backticks
        lang = example.prefix.split()[0] if example.prefix else ""

        # Skip bash/shell examples
        if lang in ("bash", "shell", "sh", "console"):
            pytest.skip("Bash examples are not tested")

        # Skip YAML examples
        if lang in ("yaml", "yml"):
            pytest.skip("YAML examples are not tested")

        # Skip non-Python examples
        if lang not in ("python", "py", "python3"):
            pytest.skip(f"Non-Python example: {lang}")

        # Run the example
        eval_example.run(example)

except ImportError:
    # pytest-examples not installed, create a placeholder test
    @pytest.mark.skip(reason="pytest-examples not installed")
    def test_docs_python_examples() -> None:
        """Placeholder test when pytest-examples is not installed."""
