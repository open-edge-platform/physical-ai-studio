"""Standalone HF commands: no live network, hardware, or model loading."""

import asyncio
import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pandas as pd
import pytest
from click.testing import CliRunner
from huggingface_hub.errors import RepositoryNotFoundError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cli.hf import hf, inspection, storage, transfer
from cli.hf.inspection import token as resolve_token
from cli.hf.schemas import AssetMetadata, Inspection
from cli.hf.state import ExportSource, ExportState, ImportState
from db.schema import Base, DatasetDB, ModelDB, ProjectDB, ProjectEnvironmentDB
from settings import Settings


def model_files(*, prefix=""):
    manifest = {
        "format": "policy_package",
        "version": "1.0",
        "policy": {"name": "pi05"},
        "model": {"artifacts": {"torch": "pi05.pt"}},
    }
    return {
        f"{prefix}model.ckpt": b"checkpoint",
        f"{prefix}snapflow-model.ckpt": b"distilled checkpoint",
        f"{prefix}exports/torch/manifest.json": json.dumps(manifest).encode(),
        f"{prefix}exports/torch/pi05.pt": b"exported model",
    }


def put_files(root, files):
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def dataset_files(root):
    info = {
        "codebase_version": "v3.0",
        "total_episodes": 1,
        "total_frames": 2,
        "fps": 30,
        "features": {"action": {"shape": [6], "dtype": "float32"}},
    }
    put_files(root, {"meta/info.json": json.dumps(info).encode(), "meta/stats.json": b"{}"})
    pd.DataFrame({"task": ["clean up"]}).to_parquet(root / "meta/tasks.parquet")
    episodes = root / "meta/episodes/chunk-000/file-000.parquet"
    episodes.parent.mkdir(parents=True)
    pd.DataFrame({"episode_index": [0], "data/chunk_index": [0], "data/file_index": [0]}).to_parquet(episodes)
    data = root / "data/chunk-000/file-000.parquet"
    data.parent.mkdir(parents=True)
    pd.DataFrame({"action": [[1.0] * 6]}).to_parquet(data)
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


class FakeHub:
    def __init__(self):
        self.repos = {}
        self.creations = []
        self.downloaded = []
        self.fail_download = False
        self.fail_upload = False
        self.uploaded = []
        self.read_revisions = []

    def repo_info(self, repo_id, **kwargs):
        self.read_revisions.append(kwargs.get("revision"))
        return SimpleNamespace(
            sha="a" * 40,
            siblings=[
                SimpleNamespace(rfilename=name, size=len(content)) for name, content in self.repos[repo_id].items()
            ],
        )

    def list_repo_files(self, repo_id, **kwargs):
        if repo_id not in self.repos:
            raise RepositoryNotFoundError(
                "Repository does not exist",
                response=httpx.Response(404, request=httpx.Request("GET", "https://hf.test")),
            )
        return list(self.repos[repo_id])

    def create_repo(self, repo_id, **kwargs):
        self.creations.append(kwargs)
        self.repos.setdefault(repo_id, {".gitattributes": b""})

    def upload_file(self, repo_id, path_in_repo, path_or_fileobj, **kwargs):
        self.repos[repo_id][path_in_repo] = path_or_fileobj
        return SimpleNamespace(oid="b" * 40)

    def large_upload(self, repo_id, kind, payload, names):
        for name in names:
            content = (payload / name).read_bytes()
            if self.repos[repo_id].get(name) == content:
                continue
            self.repos[repo_id][name] = content
            self.uploaded.append(name)
            if self.fail_upload:
                self.fail_upload = False
                raise ConnectionError("Upload disconnected")

    def snapshot_download(self, repo_id, local_dir, allow_patterns, revision, **kwargs):
        assert revision == "a" * 40
        for name in allow_patterns:
            path = local_dir / name
            if not path.exists():
                put_files(local_dir, {name: self.repos[repo_id][name]})
                self.downloaded.append(name)
            if self.fail_download:
                self.fail_download = False
                raise ConnectionError("Download disconnected")
        return str(local_dir)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    # Customer-facing commands must not depend on running from the Studio checkout.
    monkeypatch.chdir(tmp_path)
    settings = Settings(STORAGE_DIR=tmp_path / "studio", SETTINGS_FILE=str(tmp_path / "settings.json"))
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr("settings.get_settings", lambda: settings)
    hub = FakeHub()
    monkeypatch.setattr(inspection, "hub_api", lambda: hub)
    monkeypatch.setattr(transfer, "hub_api", lambda: hub)
    monkeypatch.setattr(inspection, "token", lambda: False)
    monkeypatch.setattr(transfer, "token", lambda: False)
    monkeypatch.setattr("huggingface_hub.snapshot_download", hub.snapshot_download)
    monkeypatch.setattr(transfer, "_large_upload", hub.large_upload)

    def hf_download(repo_id, name, revision, **kwargs):
        assert revision == "a" * 40
        path = tmp_path / "metadata" / repo_id / name
        put_files(path.parent, {path.name: hub.repos[repo_id][name]})
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", hf_download)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def session_context():
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    monkeypatch.setattr("db.get_async_db_session_ctx", session_context)
    project_id, dataset_id, environment_id = (str(uuid4()) for _ in range(3))

    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with session_factory() as session:
            session.add(ProjectDB(id=project_id, name="Test project"))
            await session.flush()
            session.add(ProjectEnvironmentDB(id=environment_id, project_id=project_id, name="Existing environment"))
            await session.flush()
            session.add(
                DatasetDB(
                    id=dataset_id,
                    project_id=project_id,
                    environment_id=environment_id,
                    name="Existing dataset",
                    path=str(settings.datasets_dir / dataset_id),
                    default_task="",
                )
            )
            await session.commit()

    asyncio.run(setup())
    yield SimpleNamespace(
        settings=settings, hub=hub, project_id=project_id, dataset_id=dataset_id, sessions=session_factory, tmp=tmp_path
    )
    asyncio.run(engine.dispose())


def invoke(*args):
    return CliRunner().invoke(hf, list(args))


def transfer_state(env):
    return json.loads(next((env.settings.cache_dir / "hf/transfers").glob("*/transfer.json")).read_text())


@pytest.fixture(params=["model", "dataset"])
def imported_asset(environment, request):
    env = environment
    kind = request.param
    repo_id = f"owner/{kind}"
    env.hub.repos[repo_id] = model_files() if kind == "model" else dataset_files(env.tmp / "input")
    options = ["--dataset-id", env.dataset_id] if kind == "model" else []
    result = invoke(kind, "import", repo_id, "--project-id", env.project_id, *options, "--json")
    assert result.exit_code == 0, result.output
    return SimpleNamespace(**json.loads(result.stdout))


def test_help_has_singular_groups_and_no_network(monkeypatch):
    monkeypatch.setattr(inspection, "hub_api", lambda: pytest.fail("Help must not initialize Hub"))
    for args in [(), ("model",), ("dataset", "import"), ("model", "export")]:
        result = invoke(*args, "--help")
        assert result.exit_code == 0, result.output


def test_model_and_dataset_commands_have_independent_options():
    from cli.hf.dataset import export_dataset, import_dataset, inspect_dataset
    from cli.hf.model import export_model, import_model, inspect_model

    assert hf.commands["model"].commands == {
        "inspect": inspect_model,
        "import": import_model,
        "export": export_model,
    }
    assert hf.commands["dataset"].commands == {
        "inspect": inspect_dataset,
        "import": import_dataset,
        "export": export_dataset,
    }
    assert "--dataset-id" in invoke("model", "import", "--help").stdout
    assert "--dataset-id" not in invoke("dataset", "import", "--help").stdout
    assert "--model-id" in invoke("model", "export", "--help").stdout
    assert "--model-id" not in invoke("dataset", "export", "--help").stdout
    result = invoke("dataset", "import", "--dataset-id", str(uuid4()))
    assert result.exit_code == 2
    assert "No such option" in result.output


def test_inspect_commands_select_the_correct_inspector(monkeypatch):
    monkeypatch.setattr(inspection, "inspect_remote_model", lambda **kwargs: {"model_only": True})
    monkeypatch.setattr(inspection, "inspect_remote_dataset", lambda **kwargs: {"dataset_only": True})
    assert json.loads(invoke("model", "inspect", "owner/model", "--json").stdout) == {"model_only": True}
    assert json.loads(invoke("dataset", "inspect", "owner/data", "--json").stdout) == {"dataset_only": True}


@pytest.mark.parametrize(
    ("resource", "operation"),
    [
        ("model", "import"),
        ("model", "export"),
        ("dataset", "import"),
        ("dataset", "export"),
    ],
)
def test_transfer_commands_invoke_the_correct_workflow(resource, operation, monkeypatch):
    module = import_module(f"cli.hf.{resource}")
    workflow = f"_{operation}_{resource}"
    monkeypatch.setattr(module, workflow, lambda **options: {"workflow": workflow})
    result = invoke(resource, operation, "owner/repo", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"workflow": workflow}


def test_inspect_pins_revision_and_only_downloads_json(environment):
    env = environment
    env.hub.repos["owner/model"] = model_files(prefix="wrapper/")
    result = invoke("model", "inspect", "owner/model", "--json")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["importable"] is True
    assert report["subfolder"] == "wrapper"
    assert report["revision"] == "a" * 40
    assert not env.hub.downloaded
    assert report["model"]["policy"] == "pi05"


def test_ambiguous_roots_and_subfolder(environment):
    environment.hub.repos["owner/model"] = {**model_files(prefix="a/"), **model_files(prefix="b/")}
    report = json.loads(invoke("model", "inspect", "owner/model", "--json").stdout)
    assert not report["importable"]
    result = invoke("model", "inspect", "owner/model", "--subfolder", "a", "--json")
    assert json.loads(result.stdout)["importable"]


@pytest.mark.parametrize("bad", ["../model", "/model", "a\\b", "a/../b", "a/*", "C:/model"])
def test_unsafe_paths(bad):
    with pytest.raises(ValueError, match="Unsafe"):
        inspection.safe_path(bad)


def test_checks_all_backends_and_companions(environment):
    files = model_files()
    files["exports/openvino/manifest.json"] = json.dumps(
        {
            "format": "policy_package",
            "version": "1.0",
            "policy": {"name": "pi05"},
            "model": {"artifacts": {"openvino": "model.xml"}},
        }
    ).encode()
    files["exports/openvino/model.xml"] = b"xml"
    environment.hub.repos["owner/model"] = files
    result = invoke("model", "inspect", "owner/model", "--json")
    assert result.exit_code == 0
    assert "companion" in json.loads(result.stdout)["errors"][0]


@pytest.mark.parametrize(
    ("kind", "filename", "changes", "expected"),
    [
        ("model", "exports/torch/manifest.json", {"policy": []}, "'policy' section"),
        ("model", "exports/torch/manifest.json", {"policy": {"name": []}}, "'policy.name'"),
        ("model", "exports/torch/manifest.json", {"model": None}, "object field 'model'"),
        ("model", "exports/torch/manifest.json", {"model": {"artifacts": []}}, "'model.artifacts'"),
        ("dataset", "meta/info.json", {"features": []}, "feature names to objects"),
        ("dataset", "meta/info.json", {"features": {"camera": None}}, "feature names to objects"),
    ],
)
def test_inspection_reports_malformed_metadata(environment, kind, filename, changes, expected):
    env = environment
    files = model_files() if kind == "model" else dataset_files(env.tmp / "input")
    files[filename] = json.dumps({**json.loads(files[filename]), **changes}).encode()
    env.hub.repos["owner/asset"] = files
    result = invoke(kind, "inspect", "owner/asset", "--json")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert not report["importable"]
    assert any(expected in error for error in report["errors"])


@pytest.mark.parametrize("kind", ["model", "dataset"])
def test_inspection_checks_declared_payload_and_unknown_sizes(environment, kind):
    env = environment
    files = model_files() if kind == "model" else dataset_files(env.tmp / "input")
    metadata = {
        "schema_version": 1,
        "kind": kind,
        "name": "Asset",
        "status": "complete",
        "files": [*files, "private.env"],
    }
    files["studio.json"] = json.dumps(metadata).encode()
    sizes = dict.fromkeys(files)
    inspect_files = inspection.inspect_model_files if kind == "model" else inspection.inspect_dataset_files
    report = inspect_files(sizes, lambda name: json.loads(files[name]))
    assert not report.importable
    assert not report.size_complete
    assert report.size_bytes == 0
    assert any("missing ['private.env']" in error for error in report.errors)
    assert any("outside the supported layout" in error for error in report.errors)


def test_import_download_resume_and_duplicate_safe_registration(environment):
    env = environment
    env.hub.repos["owner/model"] = model_files()
    env.hub.fail_download = True
    result = invoke(
        "model", "import", "owner/model", "--project-id", env.project_id, "--dataset-id", env.dataset_id, "--json"
    )
    assert result.exit_code == 1
    assert "--resume" in json.loads(result.stdout)["error"]["message"]
    state = transfer_state(env)
    first_download = env.hub.downloaded[0]
    result = invoke("model", "import", "--resume", state["transfer_id"], "--json")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert Path(report["path"], "model.ckpt").exists()
    assert env.hub.downloaded.count(first_download) == 1
    # Simulate a crash after DB commit but before recording completion.
    state = ImportState.load(UUID(state["transfer_id"]), "model")
    state.phase = "downloaded"
    state.save()
    assert invoke("model", "import", "--resume", state.transfer_id, "--json").exit_code == 0

    async def check():
        async with env.sessions() as session:
            models = (await session.scalars(select(ModelDB))).all()
            assert len(models) == 1
            assert models[0].train_job_id is None

    asyncio.run(check())


def test_dataset_import_creates_empty_environment_and_round_trips(environment):
    env = environment
    env.hub.repos["owner/dataset"] = dataset_files(env.tmp / "input")
    result = invoke("dataset", "import", "owner/dataset", "--project-id", env.project_id, "--json")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)

    async def check():
        async with env.sessions() as session:
            environment = await session.get(ProjectEnvironmentDB, report["environment_id"])
            assert environment.project_id == env.project_id
            assert not environment.robot_links
            assert not environment.camera_links

    asyncio.run(check())
    exported = invoke("dataset", "export", "owner/copy", "--dataset-id", report["id"], "--json")
    assert exported.exit_code == 0, exported.output
    assert env.hub.creations[0]["private"] is True
    metadata = json.loads(env.hub.repos["owner/copy"]["studio.json"])
    assert metadata["status"] == "complete"
    assert metadata["kind"] == "dataset"
    assert "environment_id" not in metadata
    copied = invoke("dataset", "import", "owner/copy", "--project-id", env.project_id, "--json")
    assert copied.exit_code == 0, copied.output
    assert json.loads(copied.stdout)["id"] != report["id"]


def test_model_export_resume_preserves_properties(environment):
    env = environment
    model_id = str(uuid4())
    root = env.settings.models_dir / model_id
    put_files(root, model_files())
    put_files(root, {"snapshot_2026-10-24_00-00-00/private.mp4": b"not exported", ".env": b"secret"})

    async def setup():
        async with env.sessions() as session:
            session.add(
                ModelDB(
                    id=model_id,
                    name="SnapFlow model",
                    project_id=env.project_id,
                    dataset_id=env.dataset_id,
                    path=str(root),
                    policy="pi05",
                    properties={"snapflow_enabled": True, "lora_enabled": True},
                )
            )
            await session.commit()

    asyncio.run(setup())
    env.hub.fail_upload = True
    result = invoke("model", "export", "owner/model", "--model-id", model_id, "--json")
    assert result.exit_code == 1
    assert env.settings.cache_dir.exists(), result.output
    state = transfer_state(env)
    metadata = json.loads(env.hub.repos["owner/model"]["studio.json"])
    assert metadata["status"] == "uploading"
    assert not json.loads(invoke("model", "inspect", "owner/model", "--json").stdout)["importable"]
    resumed = invoke("model", "export", "--resume", state["transfer_id"], "--json")
    assert resumed.exit_code == 0, resumed.output
    assert len(env.hub.uploaded) == len(set(env.hub.uploaded))
    metadata = json.loads(env.hub.repos["owner/model"]["studio.json"])
    assert metadata["properties"]["snapflow_enabled"] is True
    assert not any("snapshot" in name or ".env" in name for name in env.hub.repos["owner/model"])
    imported = invoke(
        "model", "import", "owner/model", "--project-id", env.project_id, "--dataset-id", env.dataset_id, "--json"
    )
    assert imported.exit_code == 0, imported.output

    async def check():
        async with env.sessions() as session:
            model = await session.get(ModelDB, json.loads(imported.stdout)["id"])
            assert model.properties["snapflow_enabled"] is True
            assert model.name == "SnapFlow model"

    asyncio.run(check())


def test_completed_export_resume_uses_only_the_receipt(environment, imported_asset, monkeypatch):
    env, asset = environment, imported_asset
    result = invoke(asset.kind, "export", "owner/copy", f"--{asset.kind}-id", asset.id, "--json")
    assert result.exit_code == 0, result.output
    receipt = json.loads(result.stdout)
    state = ExportState.load(UUID(receipt["transfer_id"]), asset.kind)
    assert state.phase == "completed"
    assert not (state.directory / "payload").exists()
    state.phase = "prepared"
    with transfer.locked_transfer(state) as refreshed:
        assert refreshed.phase == "completed"

    def unexpected_check(*args, **kwargs):
        pytest.fail("Completed exports must not require their source or contact the Hub")

    monkeypatch.setattr(import_module(f"cli.hf.{asset.kind}"), f"_check_{asset.kind}_source", unexpected_check)
    monkeypatch.setattr(env.hub, "repo_info", unexpected_check)
    resumed = invoke(asset.kind, "export", "--resume", receipt["transfer_id"], "--json")
    assert resumed.exit_code == 0, resumed.output
    assert json.loads(resumed.stdout) == receipt


def test_export_preserves_remote_cards_and_dry_run_has_no_side_effects(environment, imported_asset):
    env, asset = environment, imported_asset
    put_files(Path(asset.path), {"README.md": b"local card", "LICENSE": b"local license"})
    remote = {"README.md": b"remote card", "LICENSE": b"remote license"}
    env.hub.repos["owner/copy"] = remote.copy()
    saved = set((env.settings.cache_dir / "hf/transfers").glob("*/transfer.json"))
    args = (asset.kind, "export", "owner/copy", f"--{asset.kind}-id", asset.id, "--public", "--json")
    preview = invoke(*args, "--dry-run")
    assert preview.exit_code == 0, preview.output
    report = json.loads(preview.stdout)
    assert not {"README.md", "LICENSE", "studio.json"} & set(report["files"])
    assert report["private_if_created"] is False
    assert env.hub.repos["owner/copy"] == remote
    assert not env.hub.creations
    assert set((env.settings.cache_dir / "hf/transfers").glob("*/transfer.json")) == saved

    exported = invoke(*args)
    assert exported.exit_code == 0, exported.output
    assert env.hub.creations[0]["private"] is False
    assert all(env.hub.repos["owner/copy"][name] == content for name, content in remote.items())
    metadata = json.loads(env.hub.repos["owner/copy"]["studio.json"])
    assert metadata["files"] == report["files"]


def test_resumed_export_rejects_source_changes(environment, imported_asset):
    env, asset = environment, imported_asset
    env.hub.fail_upload = True
    result = invoke(asset.kind, "export", "owner/copy", f"--{asset.kind}-id", asset.id, "--json")
    assert result.exit_code == 1, result.output
    transfer_id = json.loads(result.stdout)["error"]["transfer_id"]
    name = "model.ckpt" if asset.kind == "model" else "data/chunk-000/file-000.parquet"
    Path(asset.path, name).write_bytes(b"changed")
    uploaded = env.hub.uploaded.copy()
    resumed = invoke(asset.kind, "export", "--resume", transfer_id, "--json")
    assert resumed.exit_code == 1
    assert "source changed" in resumed.stdout
    assert env.hub.uploaded == uploaded
    assert json.loads(env.hub.repos["owner/copy"]["studio.json"])["status"] == "uploading"


@pytest.mark.parametrize(
    ("update", "remote_status", "stale", "expected"),
    [
        (False, "complete", False, "nonempty"),
        (True, "complete", True, "stale"),
        (True, "uploading", False, "Another transfer"),
        (True, "complete", False, None),
        (False, None, False, "nonempty"),
    ],
)
def test_resumed_export_rechecks_remote_ownership(environment, imported_asset, update, remote_status, stale, expected):
    env, asset = environment, imported_asset
    env.hub.fail_upload = True
    options = ["--update"] if update else []
    result = invoke(asset.kind, "export", "owner/copy", f"--{asset.kind}-id", asset.id, *options, "--json")
    assert result.exit_code == 1, result.output
    transfer_id = json.loads(result.stdout)["error"]["transfer_id"]
    state = ExportState.load(UUID(transfer_id), asset.kind)
    assert state.phase == "uploading"

    remote = env.hub.repos["owner/copy"]
    if remote_status is None:
        del remote["studio.json"]
    else:
        metadata = json.loads(remote["studio.json"])
        metadata.update(transfer_id=str(uuid4()), status=remote_status)
        remote["studio.json"] = json.dumps(metadata).encode()
    if stale:
        remote["stale-payload.bin"] = b"old"
    before = remote.copy()
    uploaded = env.hub.uploaded.copy()

    resumed = invoke(asset.kind, "export", "--resume", transfer_id, "--json")
    if expected:
        assert resumed.exit_code == 1, resumed.output
        assert expected in resumed.stdout
        assert remote == before
        assert env.hub.uploaded == uploaded
    else:
        assert resumed.exit_code == 0, resumed.output
        metadata = json.loads(remote["studio.json"])
        assert metadata["transfer_id"] == transfer_id
        assert metadata["status"] == "complete"


def test_saved_export_retains_version_one_fingerprint_arrays(environment, imported_asset):
    asset = imported_asset
    result = invoke(asset.kind, "export", "owner/copy", f"--{asset.kind}-id", asset.id, "--json")
    assert result.exit_code == 0, result.output
    transfer_id = UUID(json.loads(result.stdout)["transfer_id"])
    state = ExportState.load(transfer_id, asset.kind)
    saved = json.loads((state.directory / "transfer.json").read_text())
    assert saved["schema_version"] == 1
    for name, fingerprint in state.fingerprints.items():
        stat = Path(asset.path, name).stat()
        assert fingerprint.size == stat.st_size
        assert saved["fingerprints"][name] == [stat.st_size, stat.st_mtime_ns, stat.st_ino]

    # Load a version-one document directly, without using the new serializer.
    saved["url"] = state.url
    (state.directory / "transfer.json").write_text(json.dumps(saved))
    reloaded = ExportState.load(transfer_id, asset.kind)
    assert reloaded == state


def test_state_round_trip_preserves_null_portable_metadata(environment):
    state = ExportState(
        kind="model",
        repo_id="owner/model",
        asset_id=str(uuid4()),
        source=ExportSource(environment.tmp, AssetMetadata(source_dataset_id=None)),
        files=[],
        fingerprints={},
        public=False,
        update=False,
    )
    state.save()
    saved = json.loads((state.directory / "transfer.json").read_text())
    assert saved["source"]["metadata"] == {"source_dataset_id": None}
    assert saved["source"]["path"] == str(environment.tmp)
    assert ExportState.load(UUID(state.transfer_id), "model") == state


@pytest.mark.parametrize("field", ["schema_version", "transfer_id", "operation", "phase"])
def test_saved_transfer_requires_explicit_headers(environment, imported_asset, field):
    asset = imported_asset
    state = ImportState.load(UUID(asset.transfer_id), asset.kind)
    path = state.directory / "transfer.json"
    saved = json.loads(path.read_text())
    del saved[field]
    path.write_text(json.dumps(saved))
    reads = environment.hub.read_revisions.copy()
    result = invoke(asset.kind, "import", "--resume", asset.transfer_id, "--json")
    assert result.exit_code == 1
    assert "Cannot read transfer" in result.stdout
    assert environment.hub.read_revisions == reads


@pytest.mark.parametrize("invalid", [[], {"schema_version": 1}, {"operation": "export", "phase": "downloaded"}])
def test_malformed_transfer_state_fails_before_network(environment, invalid):
    transfer_id = uuid4()
    path = environment.settings.cache_dir / "hf/transfers" / str(transfer_id) / "transfer.json"
    put_files(path.parent, {path.name: json.dumps(invalid).encode()})
    result = invoke("model", "export", "--resume", str(transfer_id), "--json")
    assert result.exit_code == 1
    assert "Cannot read transfer" in result.stdout
    assert not environment.hub.read_revisions


def test_missing_associations_fail_before_network(environment):
    result = invoke("model", "import", "owner/model", "--project-id", environment.project_id, "--json")
    assert result.exit_code == 1
    assert "--dataset-id" in result.stdout
    assert not environment.hub.read_revisions


def test_resume_rejects_overrides(environment):
    result = invoke("dataset", "import", "owner/repo", "--resume", str(uuid4()), "--json")
    assert result.exit_code == 1
    assert "cannot be combined" in result.stdout


def test_dry_run_does_not_create_transfer_or_placeholder(environment):
    env = environment
    env.hub.repos["owner/dataset"] = dataset_files(env.tmp / "input")
    result = invoke("dataset", "import", "owner/dataset", "--project-id", env.project_id, "--dry-run", "--json")
    assert result.exit_code == 0, result.output
    assert not env.settings.cache_dir.exists()
    assert not env.hub.downloaded


def test_stale_remote_shards_are_not_silently_merged(environment):
    environment.hub.repos["owner/data"] = {"data/chunk-001/file-000.parquet": b"old"}
    with pytest.raises(ValueError, match="nonempty"):
        transfer._remote_destination(environment.hub, "dataset", "owner/data", [], False)
    with pytest.raises(ValueError, match="stale"):
        transfer._remote_destination(environment.hub, "dataset", "owner/data", [], True)


def test_source_mutation_is_rejected(environment):
    env = environment
    root = env.tmp / "model"
    put_files(root, model_files())
    state = {"files": sorted(model_files()), "fingerprints": transfer.fingerprint_files(root, sorted(model_files()))}
    (root / "model.ckpt").write_bytes(b"changed")
    assert transfer.fingerprint_files(root, state["files"]) != state["fingerprints"]


def test_tokens_never_appear_in_json_errors(environment, monkeypatch):
    monkeypatch.setattr(inspection, "token", lambda: "hf_topsecret")
    monkeypatch.setattr(inspection, "hub_api", lambda: (_ for _ in ()).throw(ValueError("token=hf_topsecret")))
    result = invoke("model", "inspect", "owner/model", "--json")
    assert result.exit_code == 1
    assert "hf_topsecret" not in result.output
    assert "REDACTED" in result.stdout


def test_dataset_rejects_missing_referenced_shards(tmp_path):
    dataset_files(tmp_path)
    (tmp_path / "data/chunk-000/file-000.parquet").unlink()
    with pytest.raises(ValueError, match="missing payload"):
        storage.validate_dataset_payload(tmp_path)


def test_exports_only_import_normalizes_and_resumes_after_rename(environment, monkeypatch):
    env = environment
    files = {
        name.removeprefix("exports/"): content for name, content in model_files().items() if name.startswith("exports/")
    }
    env.hub.repos["owner/model"] = files
    model_cli = import_module("cli.hf.model")
    original = model_cli._finish_normalization
    monkeypatch.setattr(
        model_cli,
        "_finish_normalization",
        lambda destination: (_ for _ in ()).throw(RuntimeError("Crash after rename")),
    )
    result = invoke(
        "model", "import", "owner/model", "--project-id", env.project_id, "--dataset-id", env.dataset_id, "--json"
    )
    assert result.exit_code == 1
    state = transfer_state(env)
    monkeypatch.setattr(model_cli, "_finish_normalization", original)
    resumed = invoke("model", "import", "--resume", state["transfer_id"], "--json")
    assert resumed.exit_code == 0, resumed.output
    assert (Path(json.loads(resumed.stdout)["path"]) / "exports/torch/pi05.pt").exists()


def test_exports_normalization_recovers_after_moving_files(tmp_path, monkeypatch):
    model_cli = import_module("cli.hf.model")
    files = {
        name.removeprefix("exports/"): content for name, content in model_files().items() if name.startswith("exports/")
    }
    metadata = {"schema_version": 1, "kind": "model", "name": "Exports", "status": "complete", "files": sorted(files)}
    put_files(tmp_path, {**files, "studio.json": json.dumps(metadata).encode()})
    atomic_json = model_cli.atomic_json

    def fail_write(*args, **kwargs):
        raise OSError("Crash before descriptor replacement")

    monkeypatch.setattr(model_cli, "atomic_json", fail_write)
    with pytest.raises(OSError, match="descriptor replacement"):
        model_cli._finish_normalization(tmp_path)
    assert (tmp_path / "exports/torch/pi05.pt").exists()
    assert json.loads((tmp_path / "studio.json").read_text()) == metadata

    monkeypatch.setattr(model_cli, "atomic_json", atomic_json)
    model_cli._finish_normalization(tmp_path)
    normalized = json.loads((tmp_path / "studio.json").read_text())
    assert normalized["files"] == [f"exports/{name}" for name in sorted(files)]
    model_cli._finish_normalization(tmp_path)
    assert json.loads((tmp_path / "studio.json").read_text()) == normalized
    assert inspection.inspect_local_model(tmp_path).importable


def test_placeholder_environment_rolls_back_on_registration_failure(environment, monkeypatch):
    from sqlalchemy.ext.asyncio import AsyncSession

    env = environment
    env.hub.repos["owner/dataset"] = dataset_files(env.tmp / "input")
    commit = AsyncSession.commit

    async def fail_commit(session):
        raise RuntimeError("Database unavailable")

    monkeypatch.setattr(AsyncSession, "commit", fail_commit)
    result = invoke("dataset", "import", "owner/dataset", "--project-id", env.project_id, "--json")
    assert result.exit_code == 1

    async def check():
        async with env.sessions() as session:
            assert len((await session.scalars(select(ProjectEnvironmentDB))).all()) == 1
            assert len((await session.scalars(select(DatasetDB))).all()) == 1

    asyncio.run(check())
    monkeypatch.setattr(AsyncSession, "commit", commit)
    state = transfer_state(env)
    resumed = invoke("dataset", "import", "--resume", state["transfer_id"], "--json")
    assert resumed.exit_code == 0, resumed.output


def test_resume_is_locked(environment):
    from huggingface_hub.utils import WeakFileLock

    state = ImportState(
        kind="dataset",
        repo_id="owner/data",
        asset_id=str(uuid4()),
        project_id=environment.project_id,
        revision="a" * 40,
        inspection=Inspection(kind="dataset"),
        environment_id=str(uuid4()),
    )
    state.save()
    with WeakFileLock(state.directory / "transfer.lock"):
        result = invoke("dataset", "import", "--resume", state.transfer_id, "--json")
    assert result.exit_code == 1
    assert "lock" in json.loads(result.stdout)["error"]["message"]


def test_insufficient_space_prevents_payload_download(environment, monkeypatch):
    env = environment
    env.hub.repos["owner/model"] = model_files()
    monkeypatch.setattr(transfer.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    result = invoke(
        "model", "import", "owner/model", "--project-id", env.project_id, "--dataset-id", env.dataset_id, "--json"
    )
    assert result.exit_code == 1
    assert "Insufficient disk" in result.stdout
    assert not env.hub.downloaded


def test_hub_metadata_size_is_bounded(environment):
    environment.hub.repos["owner/model"] = {
        **model_files(),
        "exports/torch/manifest.json": b" " * (inspection.MAX_METADATA_BYTES + 1),
    }
    result = invoke("model", "inspect", "owner/model", "--json")
    report = json.loads(result.stdout)
    assert not report["importable"]
    assert "oversized" in report["errors"][0]


def test_saved_token_precedes_environment_and_anonymous_is_explicit(environment, monkeypatch):
    from pydantic import SecretStr

    environment.settings.huggingface.hf_token = SecretStr("hf_saved")
    monkeypatch.setenv("HF_TOKEN", "hf_environment")
    assert resolve_token() == "hf_saved"
    environment.settings.huggingface.hf_token = None
    assert resolve_token() == "hf_environment"
    monkeypatch.delenv("HF_TOKEN")
    assert resolve_token() is False


def test_upload_worker_diagnostics_stay_on_stderr(monkeypatch, capsys):
    def upload(*args, **kwargs):
        print("SDK progress")

    monkeypatch.setattr("huggingface_hub.HfApi", lambda **kwargs: SimpleNamespace(upload_large_folder=upload))
    messages = []
    transfer._upload_worker(
        "owner/model", "model", "/unused", [], False, SimpleNamespace(send=messages.append, close=lambda: None)
    )
    assert messages == [None]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "SDK progress" in captured.err


def test_cancellation_terminates_sdk_worker(monkeypatch, tmp_path):
    class Receiver:
        def recv(self):
            raise KeyboardInterrupt

        def close(self):
            pass

    calls = []
    process = SimpleNamespace(
        start=lambda: None,
        is_alive=lambda: True,
        terminate=lambda: calls.append("terminate"),
        join=lambda: calls.append("join"),
    )
    context = SimpleNamespace(
        Pipe=lambda **kwargs: (Receiver(), SimpleNamespace(close=lambda: None)), Process=lambda **kwargs: process
    )
    monkeypatch.setattr(transfer.multiprocessing, "get_context", lambda method: context)
    monkeypatch.setattr(transfer, "token", lambda: False)
    with pytest.raises(KeyboardInterrupt):
        transfer._large_upload("owner/model", "model", tmp_path, [])
    assert calls == ["terminate", "join"]


def _fake_upload_worker(repo_id, kind, payload, names, credential, pipe):
    Path(payload, "worker-started").write_text(f"{kind}:{repo_id}")
    pipe.send(None)
    pipe.close()


def test_upload_process_spawn_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr(transfer, "_upload_worker", _fake_upload_worker)
    monkeypatch.setattr(transfer, "token", lambda: False)
    transfer._large_upload("owner/model", "model", tmp_path, [])
    assert (tmp_path / "worker-started").read_text() == "model:owner/model"
