from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from robots.catalog.assets import get_builtin_robot_assets_root

SO101_REPO_URL = "https://github.com/TheRobotStudio/SO-ARM100.git"
SO101_REPO_REVISION = "fda892cba81032c46c40976a48c9ceadbf40a9ca"
WIDOWX_REPO_URL = "https://github.com/TrossenRobotics/trossen_arm_description.git"
WIDOWX_REPO_REVISION = "21d8b360c211c2ad8a065d8f462cbec0207626e7"


def sync_robot_assets(target_dir: Path | None = None) -> None:
    """Sync SO101 and WidowX assets into backend static storage."""
    target_root = target_dir or get_builtin_robot_assets_root()
    target_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)

        so101_repo_path = tmp_root / "so101-repo"
        _clone_pinned_revision(SO101_REPO_URL, SO101_REPO_REVISION, so101_repo_path, sparse=True)
        _run_git(["sparse-checkout", "set", "--no-cone", "Simulation/SO101"], cwd=so101_repo_path)

        so101_target = target_root / "SO101"
        if so101_target.exists():
            shutil.rmtree(so101_target)
        shutil.copytree(so101_repo_path / "Simulation" / "SO101", so101_target)

        widowx_repo_path = tmp_root / "widowx-repo"
        _clone_pinned_revision(WIDOWX_REPO_URL, WIDOWX_REPO_REVISION, widowx_repo_path)

        widowx_target = target_root / "widowx"
        if widowx_target.exists():
            shutil.rmtree(widowx_target)
        shutil.copytree(widowx_repo_path, widowx_target, ignore=shutil.ignore_patterns(".git"))


def _clone_pinned_revision(repo_url: str, revision: str, target_path: Path, sparse: bool = False) -> None:
    args = ["clone", "--depth", "1", "--filter=blob:none", "--no-checkout"]
    if sparse:
        args.append("--sparse")
    args.extend([repo_url, str(target_path)])
    _run_git(args)
    _run_git(["fetch", "--depth", "1", "origin", revision], cwd=target_path)
    _run_git(["checkout", "--detach", revision], cwd=target_path)


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git executable was not found")
    subprocess.run([git, *args], cwd=cwd, check=True)  # noqa: S603
