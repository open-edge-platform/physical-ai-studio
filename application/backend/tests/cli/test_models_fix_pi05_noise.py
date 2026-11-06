import asyncio
import importlib
import json
import shutil
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import onnx
import openvino as ov
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cli import cli
from db.schema import Base, JobDB, ModelDB, ProjectDB
from services.model_import_service import ModelImportService

# cli/__init__.py binds `models` to the click group, so import the module itself.
models_cli = importlib.import_module("cli.models")


def _write_old_pi05_export(export_dir: Path) -> Path:
    """Tiny OpenVINO export that samples its noise in-graph like an old pi05 export: y = x + noise."""
    export_dir.mkdir(parents=True)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 3])
    nodes = [helper.make_node("RandomNormalLike", ["x"], ["n"]), helper.make_node("Add", ["x", "n"], ["y"])]
    onnx_path = export_dir / "tiny.onnx"
    onnx.save(helper.make_model(helper.make_graph(nodes, "g", [x], [y])), onnx_path)
    ov.save_model(ov.convert_model(str(onnx_path)), export_dir / "pi05.xml")
    onnx_path.unlink()
    manifest = {
        "format": "policy_package",
        "policy": {"name": "pi05"},
        "model": {"artifacts": {"openvino": "pi05.xml"}, "preprocessors": [{"type": "pi05"}]},
    }
    (export_dir / "manifest.json").write_text(json.dumps(manifest))
    return export_dir


def _input_names(export_dir: Path) -> set[str]:
    return {name for inp in ov.Core().read_model(export_dir / "pi05.xml").inputs for name in inp.get_names()}


def _studio_model(model_dir: Path, name: str, version: int = 1, created: datetime | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        name=name,
        version=version,
        created_at=created,
        path=str(model_dir),
        project_id=uuid4(),
        dataset_id=uuid4(),
    )


def _run(*args: str, stdin: str | None = None):
    # Through the root group, as users run it: invoking `models` alone makes click
    # cache its help option without the root's `-h`.
    return CliRunner().invoke(cli, ["models", "fix-pi05-noise", *args], input=stdin)


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    _write_old_pi05_export(model_dir / "exports" / "openvino")
    return model_dir


def test_fixes_studio_model_by_id(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_id = uuid4()
    monkeypatch.setattr(models_cli, "_get_model_dir", lambda mid: model_dir if mid == model_id else None)

    result = _run(str(model_id))

    assert result.exit_code == 0, result.output
    assert "Fixed:" in result.output
    export_dir = model_dir / "exports" / "openvino"
    assert "noise" in _input_names(export_dir)
    assert "noise" not in _input_names(model_dir / models_cli.NOISE_BACKUP_DIRNAME)
    # The backup stays out of exports/, where Studio would count it as part of the model.
    assert sorted(path.name for path in (model_dir / "exports").iterdir()) == ["openvino"]


def test_fixes_any_export_folder(tmp_path: Path) -> None:
    export_dir = _write_old_pi05_export(tmp_path / "my_export")

    result = _run(str(export_dir), "--no-backup")

    assert result.exit_code == 0, result.output
    assert "noise" in _input_names(export_dir)
    assert not (tmp_path / "my_export.orig").exists()


def test_out_leaves_studio_model_untouched(model_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "converted"

    result = _run(str(model_dir), "--out", str(out))

    assert result.exit_code == 0, result.output
    assert "noise" in _input_names(out)
    assert "noise" not in _input_names(model_dir / "exports" / "openvino")


def test_second_run_reports_already_fixed(model_dir: Path) -> None:
    _run(str(model_dir))

    result = _run(str(model_dir))

    assert result.exit_code == 0, result.output
    assert "Already takes the noise as input" in result.output


def test_restore_undoes_the_fix(model_dir: Path) -> None:
    original = (model_dir / "exports" / "openvino" / "pi05.xml").read_bytes()
    _run(str(model_dir))

    result = _run(str(model_dir), "--restore")

    assert result.exit_code == 0, result.output
    assert "Restored:" in result.output
    assert (model_dir / "exports" / "openvino" / "pi05.xml").read_bytes() == original
    assert not (model_dir / models_cli.NOISE_BACKUP_DIRNAME).exists()


def test_restore_without_backup(model_dir: Path) -> None:
    _run(str(model_dir), "--no-backup")

    result = _run(str(model_dir), "--restore")

    assert result.exit_code != 0
    assert "no complete backup" in result.output


def test_copy_adds_a_studio_model_and_keeps_the_original(
    model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _studio_model(model_dir, "model_07 (INT8)", version=3)
    models_dir = tmp_path / "models"
    persisted = []

    async def get_dataset(dataset_id: UUID) -> SimpleNamespace:
        return SimpleNamespace(id=dataset_id, project_id=original.project_id)

    async def persist_import(job, model):
        persisted.append(model)
        return model

    monkeypatch.setattr(models_cli, "_studio_models", lambda: [original])
    monkeypatch.setattr(models_cli, "_get_dataset", get_dataset)
    monkeypatch.setattr(models_cli, "_persist_import", persist_import)
    settings = SimpleNamespace(models_dir=models_dir)
    monkeypatch.setattr("settings.get_settings", lambda: settings)
    monkeypatch.setattr("services.model_import_service.get_settings", lambda: settings)

    result = _run(str(model_dir), "--copy")

    assert result.exit_code == 0, result.output
    [copy] = persisted
    assert copy.name == "model_07 (INT8) (noise fix)"
    assert (copy.project_id, copy.dataset_id, copy.version) == (original.project_id, original.dataset_id, 3)
    assert "noise" in _input_names(Path(copy.path) / "exports" / "openvino")
    assert "noise" not in _input_names(model_dir / "exports" / "openvino")
    assert [path.name for path in models_dir.iterdir()] == [str(copy.id)]


def test_copy_needs_a_studio_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    export_dir = _write_old_pi05_export(tmp_path / "my_export")

    result = _run(str(export_dir), "--copy")

    assert result.exit_code != 0
    assert "--copy needs a Studio model" in result.output


def test_options_are_exclusive(model_dir: Path) -> None:
    result = _run(str(model_dir), "--copy", "--restore")

    assert result.exit_code != 0
    assert "only one of --copy, --restore and --out" in result.output


def test_rejects_unknown_target(tmp_path: Path) -> None:
    result = _run(str(tmp_path / "missing"))

    assert result.exit_code != 0
    assert "Not a folder or a model UUID" in result.output


def test_rejects_folder_without_export(tmp_path: Path) -> None:
    result = _run(str(tmp_path))

    assert result.exit_code != 0
    assert "no manifest.json" in result.output


@pytest.fixture
def studio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[SimpleNamespace]:
    """A Studio with an old pi05 model, a fixed one, a restorable one and two that are not pi05 OpenVINO."""
    old, fixed, restorable = tmp_path / "old", tmp_path / "fixed", tmp_path / "restorable"
    for folder in (old, fixed, restorable):
        _write_old_pi05_export(folder / "exports" / "openvino")
    _run(str(fixed), "--no-backup")
    _run(str(restorable))
    act = tmp_path / "act"
    (act / "exports" / "openvino").mkdir(parents=True)
    (act / "exports" / "openvino" / "manifest.json").write_text(json.dumps({"model": {"preprocessors": []}}))
    (tmp_path / "torch_only").mkdir()
    rows = [
        _studio_model(old, "pi05 old", created=datetime(2026, 10, 28, tzinfo=UTC)),
        _studio_model(act, "act"),
        _studio_model(fixed, "pi05 fixed", version=2),
        _studio_model(tmp_path / "torch_only", "torch only"),
        _studio_model(restorable, "pi05 restorable"),
    ]
    monkeypatch.setattr(models_cli, "_studio_models", lambda: rows)
    return rows


def test_lists_studio_models_and_fixes_the_chosen_one(studio: list[SimpleNamespace]) -> None:
    old, _, fixed, _, restorable = studio

    result = _run(stdin="1\n")

    assert result.exit_code == 0, result.output
    assert f"1. pi05 old (v1, 2026-10-28)  [needs fix]  {old.id}" in result.output
    assert f"2. pi05 fixed (v2)  [fixed]  {fixed.id}" in result.output
    assert f"3. pi05 restorable (v1)  [fixed, restorable]  {restorable.id}" in result.output
    assert "act" not in result.output
    assert "torch only" not in result.output
    assert "noise" in _input_names(Path(old.path) / "exports" / "openvino")


def test_restore_lists_only_restorable_models(studio: list[SimpleNamespace]) -> None:
    restorable = studio[4]

    result = _run("--restore", stdin="1\n")

    assert result.exit_code == 0, result.output
    assert f"1. pi05 restorable (v1)  [fixed, restorable]  {restorable.id}" in result.output
    assert "pi05 old" not in result.output
    assert "noise" not in _input_names(Path(restorable.path) / "exports" / "openvino")


def test_restore_lists_an_interrupted_fix(studio: list[SimpleNamespace]) -> None:
    """A fix killed before the manifest was swapped still has its backup, so it can be restored."""
    restorable = studio[4]
    model_dir = Path(restorable.path)
    export_dir = model_dir / "exports" / "openvino"
    shutil.copy2(model_dir / "noise-input-backup" / "manifest.json", export_dir / "manifest.json")

    result = _run("--restore", stdin="1\n")

    assert result.exit_code == 0, result.output
    assert f"1. pi05 restorable (v1)  [needs fix, restorable]  {restorable.id}" in result.output
    assert "noise" not in _input_names(export_dir)


def test_list_without_pi05_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models_cli, "_studio_models", list)

    result = _run()

    assert result.exit_code != 0
    assert "No Studio pi05 model with an OpenVINO export" in result.output


def test_export_folder_target_is_treated_as_its_studio_model(model_dir: Path) -> None:
    export_dir = model_dir / "exports" / "openvino"

    fixed = _run(str(export_dir))
    restored = _run(str(export_dir), "--restore")

    assert fixed.exit_code == 0, fixed.output
    assert f"{models_cli.NOISE_BACKUP_DIRNAME}" in fixed.output
    assert sorted(path.name for path in (model_dir / "exports").iterdir()) == ["openvino"]
    assert restored.exit_code == 0, restored.output
    assert "noise" not in _input_names(export_dir)


@pytest.mark.parametrize(("cwd", "target"), [("exports", "openvino"), ("exports/openvino", ".")])
def test_relative_export_targets(model_dir: Path, monkeypatch: pytest.MonkeyPatch, cwd: str, target: str) -> None:
    monkeypatch.chdir(model_dir / cwd)

    fixed = _run(target)
    restored = _run(target, "--restore")

    assert fixed.exit_code == 0, fixed.output
    assert restored.exit_code == 0, restored.output
    assert sorted(path.name for path in model_dir.iterdir()) == ["exports"]
    assert sorted(path.name for path in (model_dir / "exports").iterdir()) == ["openvino"]


def test_failed_import_saves_neither_job_nor_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The job and the model go in one transaction: a model the database rejects leaves no job behind."""

    async def check() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")

        @event.listens_for(engine.sync_engine, "connect")
        def _foreign_keys(connection, _record) -> None:
            connection.execute("PRAGMA foreign_keys=ON")

        maker = async_sessionmaker(engine, expire_on_commit=False)

        @asynccontextmanager
        async def session_ctx():
            async with maker() as session:
                try:
                    yield session
                except Exception:
                    await session.rollback()
                    raise

        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            project_id = uuid4()
            async with maker() as session:
                session.add(ProjectDB(id=str(project_id), name="Project"))
                await session.commit()
            monkeypatch.setattr("db.get_async_db_session_ctx", session_ctx)

            async def get_dataset(dataset_id: UUID) -> SimpleNamespace:
                return SimpleNamespace(id=dataset_id, project_id=project_id)  # not in the database

            service = ModelImportService(get_dataset=get_dataset, persist_import=models_cli._persist_import)
            with pytest.raises(Exception, match="FOREIGN KEY"):
                await service._finalize_import(
                    model_dir=tmp_path / str(uuid4()),
                    dataset=await get_dataset(uuid4()),
                    model_name="copy",
                    policy="pi05",
                    base_model_id=None,
                    version=1,
                )

            async with maker() as session:
                assert await session.scalar(select(func.count()).select_from(JobDB)) == 0
                assert await session.scalar(select(func.count()).select_from(ModelDB)) == 0
        finally:
            await engine.dispose()

    asyncio.run(check())
