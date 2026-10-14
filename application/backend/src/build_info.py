# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Resolve the source provenance of the running Studio application."""

import asyncio
import os
import shutil
import subprocess  # nosec B404
from dataclasses import dataclass
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class BuildInfo:
    """Source revision metadata safe to expose in the Studio UI and support reports."""

    git_sha: str | None = None
    git_commit_date: str | None = None
    git_tag: str | None = None
    git_dirty: bool = False


def _run_git(*args: str) -> str | None:
    """Return git command output when this is a usable source checkout."""
    git_path = shutil.which("git")
    if git_path is None:
        return None

    try:
        # The executable is discovered locally and every argument is a fixed literal from this module.
        result = subprocess.run(  # noqa: S603  # nosec B603
            [git_path, *args],
            cwd=_SOURCE_ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    output = result.stdout.strip()
    return output or None


def _environment_value(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value if value and value.lower() != "unknown" else None


def _get_git_metadata() -> tuple[str | None, str | None, str | None]:
    """Return the current checkout's SHA, commit date, and an exact app tag."""
    output = _run_git("show", "-s", "--format=%H%n%cI%n%D", "HEAD")
    if output is None:
        return None, None, None

    git_sha, git_commit_date, *decorations = output.splitlines()
    git_tag = next(
        (decoration.removeprefix("tag: ") for decoration in decorations if decoration.startswith("tag: app/v")),
        None,
    )
    return git_sha, git_commit_date, git_tag


def get_build_info() -> BuildInfo:
    """Return injected build metadata, falling back to an editable git checkout.

    Container images do not retain ``.git`` and therefore receive this metadata
    through build arguments. An editable checkout, including one on ``main``,
    instead resolves its current revision directly so support reports remain
    useful without a separate build step.
    """
    git_sha = _environment_value("STUDIO_GIT_SHA")
    git_commit_date = _environment_value("STUDIO_GIT_COMMIT_DATE")
    git_tag = _environment_value("STUDIO_GIT_TAG")
    has_injected_provenance = any((git_sha, git_commit_date, git_tag))
    if not has_injected_provenance:
        checkout_sha, checkout_commit_date, checkout_tag = _get_git_metadata()
        git_sha = checkout_sha
        git_commit_date = checkout_commit_date
        git_tag = checkout_tag

    injected_dirty = _environment_value("STUDIO_GIT_DIRTY")
    if injected_dirty is not None:
        git_dirty = injected_dirty.lower() in {"1", "true", "yes"}
    elif _environment_value("STUDIO_GIT_SHA") is None:
        git_dirty = _run_git("status", "--porcelain") is not None
    else:
        git_dirty = False

    return BuildInfo(
        git_sha=git_sha,
        git_commit_date=git_commit_date,
        git_tag=git_tag,
        git_dirty=git_dirty,
    )


async def get_build_info_async() -> BuildInfo:
    """Resolve build provenance without blocking the API event loop on Git."""
    return await asyncio.to_thread(get_build_info)
