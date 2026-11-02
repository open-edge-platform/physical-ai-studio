# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline safety checks for cross-run training artifact deletion."""

from unittest.mock import MagicMock
from urllib.error import URLError

import pytest
from scripts import prune_simulation_artifacts as cleanup


def artifact(identifier, run_id=9, name="training-outputs-act", **changes):
    return {
        "id": identifier,
        "name": name,
        "workflow_run": {"id": run_id},
        "expired": False,
        "size_in_bytes": 1024,
        **changes,
    }


@pytest.fixture
def github(monkeypatch):
    current = {"workflow_id": 20, "head_branch": "main", "run_number": 10}
    responses = {
        ("GET", "actions/runs/10"): current,
        ("GET", "actions/runs/9"): {**current, "run_number": 9},
        ("GET", "actions/artifacts/200"): artifact(200, 10),
        ("GET", "actions/artifacts?name=training-outputs-act&per_page=100&page=1"): {
            "artifacts": [artifact(200, 10), artifact(100)]
        },
    }

    def request(repository, method, path):
        assert repository == "owner/repo"
        return {} if method == "DELETE" else responses[(method, path)]

    mock = MagicMock(side_effect=request)
    monkeypatch.setattr(cleanup, "_request", mock)
    return responses, mock


def deleted(request):
    return [call.args[2] for call in request.call_args_list if call.args[1] == "DELETE"]


def test_only_older_same_workflow_branch_and_policy_outputs_are_deleted(github):
    responses, request = github
    current = responses[("GET", "actions/runs/10")]
    for run_id, changes in {
        8: {"workflow_id": 99, "run_number": 8},
        7: {"head_branch": "feature", "run_number": 7},
        11: {"run_number": 11},
    }.items():
        responses[("GET", f"actions/runs/{run_id}")] = {**current, **changes}
    responses[("GET", "actions/artifacts?name=training-outputs-act&per_page=100&page=1")]["artifacts"] += [
        artifact(101, 8),
        artifact(102, 7),
        artifact(103, 11),
        artifact(104, name="training-outputs-xr0"),
        artifact(105, name="sim-full-act"),
        artifact(106, expired=True),
        artifact(107, workflow_run=None),
        artifact(108, 10),
    ]
    cleanup.prune("owner/repo", 10, 200, "training-outputs-act")
    assert deleted(request) == ["actions/artifacts/100"]


@pytest.mark.parametrize(
    "changes,name",
    [
        ({"name": "training-outputs-xr0"}, "training-outputs-act"),
        ({"workflow_run": {"id": 9}}, "training-outputs-act"),
        ({"expired": True}, "training-outputs-act"),
        ({"size_in_bytes": 0}, "training-outputs-act"),
        ({"name": "sim-full-act"}, "sim-full-act"),
    ],
)
def test_replacement_must_be_verified_before_deleting_anything(github, changes, name):
    responses, request = github
    responses[("GET", "actions/artifacts/200")].update(changes)
    with pytest.raises(ValueError, match="live training artifact"):
        cleanup.prune("owner/repo", 10, 200, name)
    assert not deleted(request)


def test_all_pages_are_collected_before_deletion(github):
    responses, request = github
    responses[("GET", "actions/artifacts?name=training-outputs-act&per_page=100&page=1")]["artifacts"] = [
        artifact(identifier) for identifier in range(100)
    ]
    second_page = "actions/artifacts?name=training-outputs-act&per_page=100&page=2"
    responses[("GET", second_page)] = {"artifacts": [artifact(100)]}
    cleanup.prune("owner/repo", 10, 200, "training-outputs-act")
    assert len(deleted(request)) == 101
    methods = [(call.args[1], call.args[2]) for call in request.call_args_list]
    assert methods.index(("GET", second_page)) < methods.index(("DELETE", "actions/artifacts/0"))


@pytest.mark.parametrize(
    "failed_path",
    [
        "actions/artifacts/200",
        "actions/artifacts?name=training-outputs-act&per_page=100&page=1",
        "actions/runs/9",
    ],
)
def test_api_failure_preserves_previous_outputs(github, failed_path):
    _, request = github
    respond = request.side_effect

    def fail(repository, method, path):
        if path == failed_path:
            raise URLError("GitHub API unavailable")
        return respond(repository, method, path)

    request.side_effect = fail
    with pytest.raises(URLError):
        cleanup.prune("owner/repo", 10, 200, "training-outputs-act")
    assert not deleted(request)


def test_api_transport_uses_runner_credentials_and_handles_empty_delete_response(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "test-token")
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b""
    open_url = MagicMock(return_value=response)
    monkeypatch.setattr(cleanup, "urlopen", open_url)
    assert cleanup._request("owner/repo", "DELETE", "actions/artifacts/100") == {}
    request = open_url.call_args.args[0]
    assert request.full_url == "https://api.github.com/repos/owner/repo/actions/artifacts/100"
    assert request.method == "DELETE"
    assert request.get_header("Authorization") == "Bearer test-token"
    assert open_url.call_args.kwargs["timeout"] == 30
