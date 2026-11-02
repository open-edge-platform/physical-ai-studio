import json

from schemas.model import BackendExportDetail, BackendIOSpec, IOFeature


def test_io_feature_from_raw_list_parses_init_args() -> None:
    raw_features = [
        {
            "class_path": "physicalai.inference.data.features.InferenceFeature",
            "init_args": {
                "ftype": "STATE",
                "shape": [6],
                "name": "state",
                "dtype": "float32",
            },
        },
        {
            "class_path": "physicalai.inference.data.features.InferenceFeature",
            "init_args": {
                "ftype": "VISUAL",
                "shape": [3, 480, 640],
                "name": "images.wrist",
                "dtype": "float32",
            },
        },
    ]

    parsed = IOFeature.from_raw_list(raw_features)

    assert [feature.name for feature in parsed] == ["state", "images.wrist"]
    assert parsed[1].shape == [3, 480, 640]
    assert parsed[1].ftype == "VISUAL"
    assert parsed[1].dtype == "float32"


def test_io_feature_from_raw_list_ignores_invalid_items() -> None:
    raw_features = [
        "not-a-dict",
        {},
        {"init_args": {"shape": [1]}},
        {"name": "state", "shape": [1, "bad"], "ftype": 12, "dtype": None},
    ]

    parsed = IOFeature.from_raw_list(raw_features)

    assert len(parsed) == 1
    assert parsed[0].name == "state"
    assert parsed[0].shape is None
    assert parsed[0].ftype is None
    assert parsed[0].dtype is None


def test_backend_io_spec_from_manifest_returns_none_when_invalid() -> None:
    assert BackendIOSpec.from_manifest("invalid") is None
    assert BackendIOSpec.from_manifest({}) is None
    assert BackendIOSpec.from_manifest({"model": {"input_features": "not-a-list"}}) is None


def test_backend_io_spec_from_backend_dir_reads_manifest(tmp_path) -> None:
    manifest = {
        "model": {
            "input_features": [
                {
                    "init_args": {
                        "ftype": "STATE",
                        "shape": [6],
                        "name": "state",
                        "dtype": "float32",
                    }
                }
            ]
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    io_spec = BackendIOSpec.from_backend_dir(tmp_path)

    assert io_spec is not None
    assert io_spec.input_features[0].name == "state"


def test_backend_io_spec_from_backend_dir_returns_none_for_invalid_manifest(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{", encoding="utf-8")

    assert BackendIOSpec.from_backend_dir(tmp_path) is None


def test_backend_export_detail_from_backend_dir_includes_io_spec(tmp_path) -> None:
    backend_dir = tmp_path / "torch"
    backend_dir.mkdir(parents=True)
    (backend_dir / "model.pt").write_text("weights", encoding="utf-8")
    (backend_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": {
                    "input_features": [
                        {
                            "init_args": {
                                "ftype": "STATE",
                                "shape": [6],
                                "name": "state",
                                "dtype": "float32",
                            }
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    detail = BackendExportDetail.from_backend_dir(backend_dir)

    assert detail is not None
    assert detail.type == "torch"
    assert detail.file_count == 2
    assert detail.size_bytes > 0
    assert detail.io_spec is not None
    assert detail.io_spec.input_features[0].name == "state"


def test_backend_export_detail_from_backend_dir_returns_none_for_empty_dir(tmp_path) -> None:
    backend_dir = tmp_path / "openvino"
    backend_dir.mkdir(parents=True)

    assert BackendExportDetail.from_backend_dir(backend_dir) is None
