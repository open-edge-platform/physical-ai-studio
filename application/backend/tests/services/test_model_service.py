"""Unit tests for ModelService."""

import importlib.util
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import yaml
from lightning.pytorch.core.saving import save_hparams_to_yaml

from schemas.model import Model
from services.model_service import ModelService


def _make_model(snapshot_id=None) -> Model:
    return Model.model_validate(
        {
            "id": str(uuid4()),
            "name": "test-model",
            "policy": "act",
            "path": "/tmp/test-model",
            "project_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "snapshot_id": str(snapshot_id) if snapshot_id else None,
            "properties": {},
        }
    )


def test_get_hparams_returns_none_when_missing(tmp_path) -> None:
    model = _make_model()
    model.path = str(tmp_path)

    assert ModelService.get_hparams(model) is None


def test_omegaconf_is_installed() -> None:
    """Guard against `omegaconf` silently disappearing from the dependency tree.

    `omegaconf` is only pulled in transitively (via `lightning`, itself
    pulled in through `physicalai-train`/`lerobot`) — it is not pinned
    directly in this project's `pyproject.toml`. If it stops resolving,
    Lightning's `save_hparams_to_yaml` silently falls back to writing
    `hparams.yaml` with unsafe `!!python/tuple` tags for tuple-valued
    hparams (see `test_get_hparams_raises_on_python_tuple_tag`), which then
    crashes `GET /api/models/{model_id}`.

    Unlike the other tests here, which only *simulate* the missing
    dependency via `use_omegaconf=False`, this test asserts the real,
    installed state of the environment and fails immediately, with a clear
    message pointing at the root cause, if `omegaconf` is ever actually
    missing.
    """
    assert importlib.util.find_spec("omegaconf") is not None, (
        "omegaconf is not installed. hparams.yaml writes will silently use "
        "the unsafe YAML dumper, emitting `!!python/tuple` tags that "
        "ModelService.get_hparams cannot parse. Ensure `omegaconf` remains "
        "a resolved dependency."
    )


def test_get_hparams_raises_on_python_tuple_tag(tmp_path) -> None:
    """Regression test for a hparams.yaml written without the omegaconf dependency.

    The reported traceback (``yaml.constructor.ConstructorError: could not
    determine a constructor for the tag 'tag:yaml.org,2002:python/tuple'``)
    happens on the *write* side. Lightning's ``save_hparams_to_yaml`` only
    avoids the unsafe ``!!python/tuple`` tag when ``omegaconf`` is installed:
    it uses ``OmegaConf`` to convert tuple-valued hparams (e.g.
    ``chunk_size: (384, 384)``) to plain lists before dumping. When
    ``omegaconf`` is missing from the training environment, Lightning falls
    back to plain ``yaml.dump``, which tags tuples as ``!!python/tuple``.
    ``ModelService.get_hparams`` correctly uses ``yaml.safe_load``, which
    cannot and should not construct that tag, so it surfaces the malformed
    file as a ``ConstructorError`` instead of silently mis-parsing it.
    """
    model = _make_model()
    model.path = str(tmp_path)
    version_dir = tmp_path / "version_0"
    version_dir.mkdir(parents=True)
    hparams_path = version_dir / "hparams.yaml"

    # `use_omegaconf=False` simulates a training environment where the
    # `omegaconf` dependency is missing/unavailable, reproducing exactly how
    # the malformed hparams.yaml gets written in production.
    save_hparams_to_yaml(
        str(hparams_path),
        {"chunk_size": (384, 384), "learning_rate": 0.0001},
        use_omegaconf=False,
    )
    assert "!!python/tuple" in hparams_path.read_text()

    with pytest.raises(yaml.constructor.ConstructorError):
        ModelService.get_hparams(model)


def test_get_hparams_parses_hparams_written_with_omegaconf(tmp_path) -> None:
    """Healthy-path counterpart: with omegaconf present, no unsafe tag is written.

    When ``omegaconf`` is available (the expected/healthy environment),
    Lightning converts tuple-valued hparams to plain lists via
    ``OmegaConf.save`` instead of falling back to ``yaml.dump``, so no
    ``!!python/tuple`` tag is emitted and ``get_hparams`` parses cleanly.
    """
    model = _make_model()
    model.path = str(tmp_path)
    version_dir = tmp_path / "version_0"
    version_dir.mkdir(parents=True)
    hparams_path = version_dir / "hparams.yaml"

    save_hparams_to_yaml(
        str(hparams_path),
        {"chunk_size": (384, 384), "learning_rate": 0.0001},
    )
    assert "python/tuple" not in hparams_path.read_text()

    hparams = ModelService.get_hparams(model)

    assert hparams == {"chunk_size": [384, 384], "learning_rate": 0.0001}


def test_get_backend_details_delegates_to_backend_export_detail(tmp_path) -> None:
    model = _make_model()
    model.path = str(tmp_path)
    backend_dir = tmp_path / "exports" / "torch"
    backend_dir.mkdir(parents=True)
    with patch(
        "services.model_service.BackendExportDetail.from_backend_dir", return_value=None
    ) as mock_from_backend_dir:
        ModelService.get_backend_details(model)

    mock_from_backend_dir.assert_called_once_with(backend_dir)


@pytest.mark.anyio
async def test_delete_model_deletes_snapshot_when_snapshot_id_set() -> None:
    """When model.snapshot_id is set, delete_model should also delete the snapshot row."""
    snapshot_id = uuid4()
    model = _make_model(snapshot_id=snapshot_id)

    mock_model_repo = AsyncMock()
    mock_snapshot_repo = AsyncMock()

    mock_session = AsyncMock()

    with (
        patch("services.model_service.ModelRepository", return_value=mock_model_repo),
        patch("services.model_service.SnapshotRepository", return_value=mock_snapshot_repo),
        patch("services.model_service.shutil.rmtree"),
    ):
        await ModelService(mock_session).delete_model(model)

    mock_model_repo.delete_by_id.assert_awaited_once_with(model.id)
    mock_snapshot_repo.delete_by_id.assert_awaited_once_with(model.snapshot_id)


@pytest.mark.anyio
async def test_delete_model_skips_snapshot_delete_when_no_snapshot_id() -> None:
    """When model.snapshot_id is None, snapshot repo delete should NOT be called."""
    model = _make_model(snapshot_id=None)

    mock_model_repo = AsyncMock()
    mock_snapshot_repo = AsyncMock()

    mock_session = AsyncMock()

    with (
        patch("services.model_service.ModelRepository", return_value=mock_model_repo),
        patch("services.model_service.SnapshotRepository", return_value=mock_snapshot_repo),
        patch("services.model_service.shutil.rmtree"),
    ):
        await ModelService(mock_session).delete_model(model)

    mock_model_repo.delete_by_id.assert_awaited_once_with(model.id)
    mock_snapshot_repo.delete_by_id.assert_not_awaited()
