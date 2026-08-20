import importlib
from pathlib import Path

import robots.catalog.assets as assets

sync_module = importlib.import_module("robots.catalog.sync_robot_assets")


def test_sync_robot_assets_uses_default_assets_root(tmp_path, monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(assets, "BUILTIN_ROBOT_ASSETS_ROOT", tmp_path)

    def fake_run_git(args: list[str], cwd: Path | None = None) -> None:
        calls.append((args, cwd))
        if args[:2] == ["clone", "--depth"] and args[-2] == sync_module.SO101_REPO_URL:
            (Path(args[-1]) / "Simulation" / "SO101").mkdir(parents=True)
        if args[:2] == ["clone", "--depth"] and args[-2] == sync_module.WIDOWX_REPO_URL:
            Path(args[-1]).mkdir(parents=True)

    monkeypatch.setattr(sync_module, "_run_git", fake_run_git)

    sync_module.sync_robot_assets()

    assert (tmp_path / "SO101").is_dir()
    assert (tmp_path / "widowx").is_dir()
    assert any(
        args[:5] == ["clone", "--depth", "1", "--filter=blob:none", "--no-checkout"]
        and "--sparse" in args
        and args[-2] == sync_module.SO101_REPO_URL
        for args, _ in calls
    )
    assert any(
        args[:5] == ["clone", "--depth", "1", "--filter=blob:none", "--no-checkout"]
        and "--sparse" not in args
        and args[-2] == sync_module.WIDOWX_REPO_URL
        for args, _ in calls
    )
    assert any(
        args == ["fetch", "--depth", "1", "origin", sync_module.SO101_REPO_REVISION]
        and cwd is not None
        and cwd.name == "so101-repo"
        for args, cwd in calls
    )
    assert any(
        args == ["checkout", "--detach", sync_module.SO101_REPO_REVISION]
        and cwd is not None
        and cwd.name == "so101-repo"
        for args, cwd in calls
    )
    assert any(
        args == ["fetch", "--depth", "1", "origin", sync_module.WIDOWX_REPO_REVISION]
        and cwd is not None
        and cwd.name == "widowx-repo"
        for args, cwd in calls
    )
    assert any(
        args == ["checkout", "--detach", sync_module.WIDOWX_REPO_REVISION]
        and cwd is not None
        and cwd.name == "widowx-repo"
        for args, cwd in calls
    )
    assert len(calls) == 7
