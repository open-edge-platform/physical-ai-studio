# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import zipfile
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from schemas.model import Model
from schemas.model_variant import ExportPackage, VariantId, VariantInfo
from services.model_download_service import ModelDownloadService
from services.model_service import ModelService


def write_package(directory: Path, info: VariantInfo | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.xml").write_text("graph")
    (directory / "model.bin").write_bytes(b"weights")
    manifest = {"model": {"artifacts": {"openvino": "model.xml"}}}
    if info is not None:
        manifest["studio"] = {"variant": info.model_dump(mode="json")}
    (directory / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("variant", ["original", "int8"])
def test_package_location_can_be_resolved_before_export(tmp_path: Path, variant: VariantId) -> None:
    package = ExportPackage.for_model(tmp_path, "openvino", variant)
    expected = tmp_path / "exports/openvino"
    if variant != "original":
        expected = expected / "variants" / variant
    assert package.directory == expected
    assert not expected.exists()


@pytest.mark.parametrize(
    "backend,variant", [("../openvino", "original"), ("openvino", "../int8"), ("openvino", "/int8")]
)
def test_package_rejects_path_injection(tmp_path: Path, backend: str, variant: str) -> None:
    with pytest.raises(ValueError):
        ExportPackage.for_model(tmp_path, backend, variant)


def test_export_and_dependency_symlinks_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "model"
    (root / "exports").mkdir(parents=True)
    (root / "exports/openvino").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        ExportPackage.for_model(root, "openvino")
    write_package(tmp_path / "package")
    (tmp_path / "package/model.bin").unlink()
    (tmp_path / "package/model.bin").symlink_to(tmp_path / "outside.bin")
    with pytest.raises(ValueError, match="symbolic"):
        _ = ExportPackage(tmp_path / "package", "openvino").manifest


def test_package_reuses_its_validated_manifest(tmp_path: Path) -> None:
    write_package(tmp_path, VariantInfo(id="int8", weight_compression="int8_sym"))
    package = ExportPackage(tmp_path, "openvino", "int8")
    with patch("schemas.model_variant.json.loads", wraps=json.loads) as parse:
        assert package.info.weight_compression == "int8_sym"
        assert package.artifact == tmp_path / "model.xml"
        assert len(package.files) == 3
        parse.assert_called_once()
    (tmp_path / "manifest.json").write_text("invalid JSON")
    assert package.info.id == "int8"
    with pytest.raises(ValueError):
        _ = ExportPackage(tmp_path, "openvino").manifest


@pytest.mark.parametrize("artifact", ["../outside.bin", "/outside.bin", "missing.bin"])
def test_package_rejects_unsafe_or_missing_dependencies(tmp_path: Path, artifact: str) -> None:
    write_package(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model"]["preprocessor"] = {"steps": [{"artifact": artifact}]}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Missing or unsafe"):
        _ = ExportPackage(tmp_path, "openvino").manifest


def test_package_requires_openvino_weights(tmp_path: Path) -> None:
    write_package(tmp_path)
    (tmp_path / "model.bin").unlink()
    with pytest.raises(ValueError, match="Missing weights"):
        _ = ExportPackage(tmp_path, "openvino").manifest


def test_variant_discovery_and_downloads_keep_packages_separate(tmp_path: Path) -> None:
    model = Model(path=str(tmp_path), name="Policy", policy="act", project_id=uuid4(), properties={})
    original = ExportPackage.for_model(tmp_path, "openvino")
    variant = ExportPackage.for_model(tmp_path, "openvino", "int8")
    write_package(original.directory)
    write_package(variant.directory, VariantInfo(id="int8", weight_compression="int8_sym"))
    (variant.directory / ".private").write_text("not exported")
    details = ModelService.get_backend_details(model)[0]
    assert [item.id for item in details.variants] == ["original", "int8"]
    assert details.size_bytes == details.variants[0].size_bytes
    assert ModelService.resolve_export(model, "openvino", "int8") == variant.directory
    archive_path = ModelDownloadService().create_backend_archive(original.directory, "openvino")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            assert set(archive.namelist()) == {"manifest.json", "model.xml", "model.bin"}
    finally:
        archive_path.unlink()
    (variant.directory / ".unpublished").write_text(str(uuid4()))
    assert set(model.export_packages()) == {("openvino", "original")}


def test_variant_only_backend_is_not_counted_as_an_original(tmp_path: Path) -> None:
    model = Model(path=str(tmp_path), name="Policy", policy="act", project_id=uuid4(), properties={})
    variant = ExportPackage.for_model(tmp_path, "openvino", "int8")
    write_package(variant.directory)
    assert model.available_backends == ["openvino"]
    assert set(model.export_packages()) == {("openvino", "int8")}
    assert not ExportPackage.for_model(tmp_path, "openvino").files
