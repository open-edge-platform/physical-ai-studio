"""Tests for the command contract the runtime session consumes.

The load commands carry identifiers the API has resolved against the project,
never rows supplied by the browser — the session must not be handed a path it
did not derive itself. Phase B ships these over Zenoh as msgpack, so they must
also survive a round trip through plain JSON types.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from runtime.contract import CommandAdapter, LoadDatasetCommand, LoadModelCommand
from schemas import InferenceBackend


def _load_model_payload() -> dict[str, Any]:
    return {
        "command": "load_model",
        "model_id": str(uuid4()),
        # The UI picks a device out of InferenceDeviceInfo, so the extra keys
        # that come with it must be ignored rather than rejected.
        "inference_device": {
            "backend": "openvino",
            "device": "GPU.0",
            "type": "xpu",
            "name": "Intel Arc",
            "memory": 8000,
            "index": 0,
        },
    }


def test_load_model_carries_an_id_and_a_device() -> None:
    payload = _load_model_payload()

    command = CommandAdapter.validate_python(payload)

    assert isinstance(command, LoadModelCommand)
    assert str(command.model_id) == payload["model_id"]
    assert command.inference_device.backend == InferenceBackend.OPENVINO
    assert command.inference_device.device == "GPU.0"


def test_load_model_cannot_name_a_model_directory() -> None:
    """A row (and the path inside it) is not an accepted substitute for an id."""
    payload = _load_model_payload()
    del payload["model_id"]
    payload["model"] = {"id": str(uuid4()), "path": "/etc", "policy": "act"}

    with pytest.raises(ValidationError, match="model_id"):
        CommandAdapter.validate_python(payload)


def test_load_model_rejects_an_unknown_backend() -> None:
    payload = _load_model_payload()
    payload["inference_device"]["backend"] = "not_a_backend"

    with pytest.raises(ValidationError):
        CommandAdapter.validate_python(payload)


def test_load_dataset_carries_an_id() -> None:
    dataset_id = uuid4()

    command = CommandAdapter.validate_python({"command": "load_dataset", "dataset_id": str(dataset_id)})

    assert isinstance(command, LoadDatasetCommand)
    assert command.dataset_id == dataset_id


def test_commands_round_trip_through_json() -> None:
    """Phase B carries commands as msgpack, so they must dump to plain data."""
    command = CommandAdapter.validate_python(_load_model_payload())

    restored = CommandAdapter.validate_python(command.model_dump(mode="json"))

    assert restored == command
