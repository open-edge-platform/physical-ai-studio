# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Keep the latest uploaded training outputs for this workflow, branch, and policy."""

from __future__ import annotations

import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _request(repository: str, method: str, path: str) -> dict:
    request = Request(
        f"https://api.github.com/repos/{repository}/{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method=method,
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - Request uses the fixed HTTPS GitHub API origin.
        body = response.read()
    return json.loads(body) if body else {}


def prune(repository: str, run_id: int, artifact_id: int, name: str) -> None:
    """Delete older matching artifacts only after verifying the replacement exists.

    Args:
        repository: Current owner/repository.
        run_id: Current workflow run ID.
        artifact_id: Successful upload's artifact ID.
        name: Exact training artifact name for the current policy.

    Raises:
        ValueError: The replacement is not a live training artifact from this run.
    """
    current = _request(repository, "GET", f"actions/runs/{run_id}")
    uploaded = _request(repository, "GET", f"actions/artifacts/{artifact_id}")
    if (
        not name.startswith("training-outputs-")
        or uploaded["name"] != name
        or uploaded["workflow_run"]["id"] != run_id
        or uploaded["expired"]
        or uploaded["size_in_bytes"] <= 0
    ):
        raise ValueError("A live training artifact from this run is required before pruning")

    # Collect all pages before deleting; deletion shifts pagination offsets.
    obsolete = []
    page = 1
    while True:
        query = urlencode({"name": name, "per_page": 100, "page": page})
        artifacts = _request(repository, "GET", f"actions/artifacts?{query}")["artifacts"]
        for artifact in artifacts:
            if artifact["name"] != name or artifact["id"] == artifact_id or artifact["expired"]:
                continue
            previous_id = (artifact.get("workflow_run") or {}).get("id")
            if previous_id is None or previous_id == run_id:
                continue
            previous = _request(repository, "GET", f"actions/runs/{previous_id}")
            if (
                previous["workflow_id"] == current["workflow_id"]
                and previous["head_branch"] == current["head_branch"]
                and previous["run_number"] < current["run_number"]
            ):
                obsolete.append(artifact["id"])
        if len(artifacts) < 100:
            break
        page += 1
    for old_id in obsolete:
        _request(repository, "DELETE", f"actions/artifacts/{old_id}")
        print(f"Deleted training artifact {old_id}; retained {artifact_id}")


if __name__ == "__main__":
    prune(
        os.environ["GITHUB_REPOSITORY"],
        int(os.environ["GITHUB_RUN_ID"]),
        int(os.environ["ARTIFACT_ID"]),
        os.environ["ARTIFACT_NAME"],
    )
