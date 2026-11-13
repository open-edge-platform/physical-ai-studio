import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tests.lerobot_datasets import FRAMES_PER_EPISODE, record_dataset

from internal_datasets.lerobot import episode_task
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from internal_datasets.mutations.edit_episode_task_mutation import EditEpisodeTaskMutation

TASKS = ["pick the cube", "plcae the cube", "pick the cube"]
"""Episode 1 was recorded with a typo."""

DATA_FILE = "data/chunk-000/file-000.parquet"
EPISODES_FILE = "meta/episodes/chunk-000/file-000.parquet"


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return record_dataset(tmp_path_factory.mktemp("recorded") / "dataset", TASKS)


@pytest.fixture
def root(recorded: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(recorded, tmp_path / "dataset"))


def _tasks(root: Path) -> dict[str, int]:
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    return {str(task): int(index) for task, index in tasks["task_index"].items()}


def _frame_task_indices(root: Path) -> dict[int, list[int]]:
    frames = pq.read_table(root / DATA_FILE, columns=["episode_index", "task_index"]).to_pandas()
    return {int(episode): group["task_index"].tolist() for episode, group in frames.groupby("episode_index")}


def _episode_rows(root: Path) -> pd.DataFrame:
    return pq.read_table(root / EPISODES_FILE).to_pandas().set_index("episode_index")


def _digests(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _edit(root: Path, episode_index: int, task: str):
    return EditEpisodeTaskMutation(InternalLeRobotDataset(root)).edit_task(episode_index, task)


def test_new_task_is_appended_and_only_the_episode_uses_it(root: Path) -> None:
    episode = _edit(root, 1, "  place the cube ")

    assert episode is not None
    assert episode.episode_index == 1
    assert episode.tasks == ["place the cube"]
    # The typo stays: dropping it would renumber the tasks after it.
    assert _tasks(root) == {"pick the cube": 0, "plcae the cube": 1, "place the cube": 2}
    assert _frame_task_indices(root) == {0: [0] * FRAMES_PER_EPISODE, 1: [2] * FRAMES_PER_EPISODE, 2: [0, 0, 0]}
    assert _episode_rows(root)["tasks"].map(list).to_dict() == {
        0: ["pick the cube"],
        1: ["place the cube"],
        2: ["pick the cube"],
    }
    assert json.loads((root / "meta/info.json").read_text())["total_tasks"] == 3


def test_existing_task_is_reused(root: Path) -> None:
    episode = _edit(root, 1, "pick the cube")

    assert episode is not None
    assert episode.tasks == ["pick the cube"]
    assert _tasks(root) == {"pick the cube": 0, "plcae the cube": 1}
    assert _frame_task_indices(root) == {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}
    assert json.loads((root / "meta/info.json").read_text())["total_tasks"] == 2


def test_other_episodes_frames_and_videos_are_untouched(root: Path) -> None:
    before_frames = pq.read_table(root / DATA_FILE).to_pandas()
    before = _digests(root)

    _edit(root, 2, "stack the cube")

    after_frames = pq.read_table(root / DATA_FILE).to_pandas()
    after = _digests(root)
    pd.testing.assert_frame_equal(before_frames.drop(columns="task_index"), after_frames.drop(columns="task_index"))
    assert after_frames["task_index"][after_frames["episode_index"] != 2].tolist() == [0, 0, 0, 1, 1, 1]
    changed = {path for path in before if before[path] != after.get(path)}
    assert changed == {"meta/tasks.parquet", DATA_FILE, EPISODES_FILE, "meta/stats.json", "meta/info.json"}
    assert set(after) == set(before), "no temporary files are left behind"


def test_task_index_stats_follow_the_frames(root: Path) -> None:
    _edit(root, 1, "place the cube")

    row = _episode_rows(root).loc[1]
    for stat in ("min", "max", "mean", "q01", "q50", "q99"):
        assert list(row[f"stats/task_index/{stat}"]) == [2.0]
    assert list(row["stats/task_index/std"]) == [0.0]
    assert list(row["stats/task_index/count"]) == [FRAMES_PER_EPISODE]

    stats = json.loads((root / "meta/stats.json").read_text())["task_index"]
    frames = [index for indices in _frame_task_indices(root).values() for index in indices]
    assert stats["min"] == [min(frames)]
    assert stats["max"] == [max(frames)]
    assert stats["mean"] == pytest.approx([sum(frames) / len(frames)])
    assert stats["count"] == [len(frames)]


def test_lerobot_reads_the_edited_dataset(root: Path) -> None:
    _edit(root, 1, "place the cube")

    dataset = LeRobotDataset("edited", root)

    assert dataset.meta.total_tasks == 3
    assert [episode["tasks"] for episode in dataset.meta.episodes] == [
        ["pick the cube"],
        ["place the cube"],
        ["pick the cube"],
    ]
    # LeRobot's own lookup (DatasetReader.get_item), on the raw rows: decoding the videos needs FFmpeg.
    frame_tasks = [
        dataset.meta.tasks.iloc[int(dataset.get_raw_item(index)["task_index"])].name for index in range(len(dataset))
    ]
    assert frame_tasks == (
        ["pick the cube"] * FRAMES_PER_EPISODE
        + ["place the cube"] * FRAMES_PER_EPISODE
        + ["pick the cube"] * FRAMES_PER_EPISODE
    )


def test_client_serves_the_edit_without_reopening(root: Path) -> None:
    client = InternalLeRobotDataset(root)

    EditEpisodeTaskMutation(client).edit_task(1, "place the cube")

    assert [info.tasks for info in client.get_episode_infos()] == [
        ["pick the cube"],
        ["place the cube"],
        ["pick the cube"],
    ]
    assert client.get_tasks() == ["pick the cube", "plcae the cube", "place the cube"]


def test_unchanged_task_writes_nothing(root: Path) -> None:
    before = {path: path.stat().st_mtime_ns for path in root.rglob("*") if path.is_file()}

    episode = _edit(root, 0, "pick the cube")

    assert episode is not None
    assert episode.tasks == ["pick the cube"]
    assert {path: path.stat().st_mtime_ns for path in root.rglob("*") if path.is_file()} == before


def test_missing_episode_returns_none(root: Path) -> None:
    before = _digests(root)

    assert _edit(root, 7, "place the cube") is None
    assert _digests(root) == before


@pytest.mark.parametrize("task", ["", "   "])
def test_empty_task_is_refused(root: Path, task: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _edit(root, 1, task)


def test_failed_write_leaves_the_dataset_as_it_was(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before = _digests(root)
    write_table = pq.write_table

    def fail_on_episodes_metadata(table, where, *args, **kwargs):
        if "episodes" in str(Path(where).parent):
            raise OSError("disk full")
        return write_table(table, where, *args, **kwargs)

    monkeypatch.setattr(episode_task.pq, "write_table", fail_on_episodes_metadata)

    with pytest.raises(OSError, match="disk full"):
        _edit(root, 1, "place the cube")

    assert _digests(root) == before


@pytest.mark.parametrize(
    "failing", ["meta/tasks.parquet", DATA_FILE, EPISODES_FILE, "meta/stats.json", "meta/info.json"]
)
def test_failed_swap_puts_back_what_was_swapped(root: Path, monkeypatch: pytest.MonkeyPatch, failing: str) -> None:
    before = _digests(root)
    replace = Path.replace

    def fail_on(self: Path, target: Path) -> Path:
        if self.name.endswith(".tmp") and Path(target) == root / failing:
            raise OSError("swap failed")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_on)

    with pytest.raises(OSError, match="swap failed"):
        _edit(root, 1, "place the cube")

    assert _digests(root) == before  # no staged or backup files left either


def test_dataset_with_nothing_recorded_has_no_episode(tmp_path: Path) -> None:
    assert _edit(tmp_path, 0, "place the cube") is None


def test_failed_backup_copy_leaves_no_partial_backup(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before = _digests(root)

    def no_links(*_args: object) -> None:
        raise OSError("hard links not supported")

    def partial_copy(source: Path, destination: Path) -> None:
        Path(destination).write_bytes(Path(source).read_bytes()[:10])
        raise OSError("disk full")

    monkeypatch.setattr(episode_task.os, "link", no_links)
    monkeypatch.setattr(episode_task.shutil, "copy2", partial_copy)

    with pytest.raises(OSError, match="disk full"):
        _edit(root, 1, "place the cube")

    assert _digests(root) == before
