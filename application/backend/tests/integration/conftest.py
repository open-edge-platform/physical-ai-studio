"""Fixtures shared by the backend end-to-end integration tests.

These tests exercise the real FastAPI app, real services, and a real
(isolated) SQLite database instead of stubbing services out, so they need a
migrated schema and a small on-disk LeRobot v3 dataset to upload.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

_INTEGRATION_TESTS_DIRECTORY = Path(__file__).parent
_exitstatus: int | None = None
_hard_exit = False


def pytest_collection_finish(session: pytest.Session) -> None:
    """Enable the shutdown workaround only for selected integration tests."""
    global _hard_exit  # noqa: PLW0603
    _hard_exit = any(item.path.is_relative_to(_INTEGRATION_TESTS_DIRECTORY) for item in session.items)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Stash the exit status for `pytest_unconfigure` to hard-exit with.

    Reporting (terminal summary, warnings, junit-xml, ...) happens in other
    `pytest_sessionfinish`/`pytest_terminal_summary` implementations that run
    after this one, so we can't force-exit here without truncating it.
    """
    global _exitstatus  # noqa: PLW0603
    _exitstatus = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    """Hard-exit once pytest is fully done, instead of hanging forever.

    The training/export path pulls in pyarrow, HuggingFace `datasets`, and
    PyTorch, which can leave native worker threads running that were never
    designed to be joined (e.g. pyarrow's global CPU thread pool). CPython's
    normal interpreter shutdown blocks forever trying to join them, so we
    bypass it. `pytest_unconfigure` is pytest's last hook, called strictly
    after the terminal summary is printed, so this can't truncate any output.
    """
    if _hard_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_exitstatus if _exitstatus is not None else 1)


@pytest.fixture(scope="session")
def migrated_db() -> None:
    """Run Alembic migrations once against the session's isolated STORAGE_DIR.

    `application/backend/tests/conftest.py` points STORAGE_DIR at a fresh temp
    directory before any other module is imported, so the sqlite file backing
    `db.engine` is empty until migrations create the schema.
    """
    from db.migration import MigrationManager
    from settings import get_settings

    manager = MigrationManager(get_settings())
    assert manager.initialize_database(), "Failed to initialize the isolated test database"


def _build_synthetic_lerobot_v3_dataset(
    destination: Path,
    *,
    num_episodes: int = 2,
    frames_per_episode: int = 8,
    fps: int = 10,
    image_size: int = 64,
    task: str = "do the thing",
) -> Path:
    """Write a tiny, fully-synthetic LeRobot v3 dataset to *destination*.

    Uses the real `lerobot` dataset writer (`LeRobotDataset.create` /
    `add_frame` / `save_episode` / `finalize`) so the on-disk layout matches
    exactly what both the backend's dataset-import adapter (`meta/info.json`,
    `meta/tasks.parquet`, `data/chunk-*/file-*.parquet`) and
    `physicalai.data.lerobot.LeRobotDataModule` expect - no hand-authored
    parquet/JSON fixtures to keep in sync with either format.

    Images are stored uncompressed (``use_videos=False``) rather than encoded
    to video, since encoding is unnecessary overhead for a 2-episode fixture
    and ACT resizes to its configured `image_size` regardless of source
    resolution.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if destination.exists():
        shutil.rmtree(destination)

    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        "observation.images.top": {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id="physicalai-studio-tests/synthetic-e2e",
        fps=fps,
        features=features,
        root=destination,
        robot_type="test_robot",
        use_videos=False,
    )

    rng = np.random.default_rng(seed=0)
    for _episode in range(num_episodes):
        for _frame in range(frames_per_episode):
            dataset.add_frame(
                {
                    "observation.state": rng.random(2).astype(np.float32),
                    "action": rng.random(2).astype(np.float32),
                    "observation.images.top": rng.integers(0, 255, size=(image_size, image_size, 3), dtype=np.uint8),
                    "task": task,
                }
            )
        dataset.save_episode()

    dataset.finalize()
    return destination


def _zip_directory(directory: Path) -> bytes:
    """Zip *directory*'s contents (relative paths, no enclosing folder)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(directory)))
    return buffer.getvalue()


@pytest.fixture
def synthetic_dataset_archive_bytes(tmp_path: Path) -> bytes:
    """Zip bytes for a tiny synthetic LeRobot v3 dataset (2 episodes, 8 frames each)."""
    dataset_dir = _build_synthetic_lerobot_v3_dataset(tmp_path / "synthetic_dataset")
    return _zip_directory(dataset_dir)
