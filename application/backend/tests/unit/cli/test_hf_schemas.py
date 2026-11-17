"""Structured Hub results retain the public JSON contract and reject invalid metadata."""

import pytest
from pydantic import TypeAdapter, ValidationError

from cli.hf.schemas import AssetMetadata, DatasetInfo, Inspection, ModelDetails, TransferResult, to_json


def test_inspection_serializes_the_existing_error_report_shape():
    report = Inspection(kind="model", errors=["No model found"])
    assert to_json(report) == {
        "schema_version": 1,
        "kind": "model",
        "subfolder": None,
        "validation_level": "metadata",
        "importable": False,
        "errors": ["No model found"],
        "warnings": [],
        "files": [],
        "size_bytes": 0,
        "size_complete": True,
    }
    report.errors.clear()
    assert report.importable
    assert to_json(report)["importable"] is True
    assert Inspection(kind="model").errors == []


def test_inspection_preserves_raw_manifest_details_and_nested_nulls():
    manifest = {"policy": {"name": "pi05"}, "extension": {"labels": ["custom", None]}}
    report = Inspection(kind="model", model=ModelDetails(policy=None, backends={"torch": manifest}, checkpoints=[]))
    assert to_json(report)["model"] == {"policy": None, "backends": {"torch": manifest}, "checkpoints": []}


def test_portable_metadata_round_trip_preserves_unknown_fields_and_explicit_nulls():
    raw = {
        "name": "My model",
        "schema_version": 1,
        "kind": "model",
        "status": "complete",
        "files": [],
        "properties": {"snapflow_enabled": True, "extension": {"labels": ["custom"]}},
        "source_dataset_id": None,
        "extension": {"extra": [1, None]},
    }
    metadata = AssetMetadata.model_validate(raw)
    assert metadata.version == 1
    assert metadata.source_dataset_id is None
    assert to_json(metadata) == raw


@pytest.mark.parametrize(
    "invalid",
    [
        {"version": True},
        {"version": 0},
        {"version": "1"},
        {"properties": {"lora_enabled": 1}},
        {"properties": {"snapflow_enabled": "true"}},
        {"name": " "},
        {"name": "x" * 256},
        {"default_task": None},
    ],
)
def test_portable_metadata_rejects_invalid_known_fields(invalid):
    with pytest.raises(ValidationError):
        AssetMetadata.model_validate(invalid)


@pytest.mark.parametrize("errors", [[], ["Invalid layout"]])
def test_saved_inspection_rejects_contradictory_importability(errors):
    raw = to_json(Inspection(kind="model", errors=errors))
    assert TypeAdapter(Inspection).validate_python(raw).errors == errors
    raw["importable"] = bool(errors)
    with pytest.raises(ValidationError, match="disagrees"):
        TypeAdapter(Inspection).validate_python(raw)


def test_dataset_info_identifies_cameras_without_changing_feature_metadata():
    features = {"camera": {"dtype": "video", "shape": [480, 640, 3]}, "action": {"dtype": "float32", "shape": [6]}}
    info = DatasetInfo.model_validate({"features": features, "fps": 30})
    assert info.cameras == ["camera"]
    assert info.features == features
    assert info.fps == 30


def test_transfer_result_omits_fields_not_applicable_to_the_operation():
    result = TransferResult(
        kind="model", id="model-id", repo_id="owner/model", revision="a" * 40, url="https://hub/model"
    )
    assert to_json(result) == {
        "kind": "model",
        "id": "model-id",
        "repo_id": "owner/model",
        "revision": "a" * 40,
        "url": "https://hub/model",
    }


def test_inspection_requires_a_pinned_revision_for_downloads():
    report = Inspection(kind="model")
    with pytest.raises(ValueError, match="pinned revision"):
        _ = report.pinned_revision
    report.revision = "a" * 40
    assert report.pinned_revision == "a" * 40
