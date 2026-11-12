"""Change the task of one recorded episode of a LeRobot v3 dataset, in place.

Only task metadata changes; videos are never touched:

- ``meta/tasks.parquet`` gains the task when it is new.
- The episode's frames in its ``data/chunk-*/file-*.parquet`` get the task's ``task_index``.
- The episode's row in ``meta/episodes/chunk-*/file-*.parquet`` gets the new ``tasks`` list and
  ``stats/task_index/*``, and ``meta/stats.json`` gets ``task_index`` aggregated again.
- ``meta/info.json`` gets ``total_tasks``. It is always rewritten, which also tells Studio's
  read cache (keyed on its mtime) that the dataset changed.

Every file is first written next to the one it replaces. Only when all of them are written are
they swapped in with ``os.replace``, ``info.json`` last. A failure while writing or swapping
leaves the dataset as it was: the swapped files are put back from hard-linked backups (best
effort: a failure while putting one back is not retried). A crash in the middle of the swaps can
still leave part of the edit.

A task no episode uses any more stays in ``meta/tasks.parquet``: LeRobot looks a frame's task up
by position (``tasks.iloc[task_index]``), so dropping one would renumber the frames of every
episode recorded with a later task.
"""

import json
import os
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_feature_stats
from lerobot.datasets.utils import DEFAULT_TASKS_PATH, EPISODES_DIR, INFO_PATH, STATS_PATH

from internal_datasets.dataset_client import EpisodeNotFoundError

TASK_INDEX = "task_index"
_TASK_STATS_PREFIX = f"stats/{TASK_INDEX}/"
_EPISODE_INDEX = "episode_index"


def set_episode_task(root: Path, episode_index: int, task: str) -> bool:
    """Make ``task`` the only task of episode ``episode_index``; return whether anything changed.

    Raises:
        ValueError: ``task`` is empty, or the episode's frames disagree with its metadata.
        EpisodeNotFoundError: The dataset has no such episode.
    """
    if not task.strip():
        raise ValueError("The task must not be empty")

    if not (root / INFO_PATH).is_file():  # nothing recorded yet
        raise EpisodeNotFoundError(f"Episode {episode_index} not found")
    info: dict[str, Any] = json.loads((root / INFO_PATH).read_text())
    episodes_path, episodes = _find_episodes_file(root, episode_index)
    row = episodes[_EPISODE_INDEX].to_pylist().index(episode_index)
    episode = {name: episodes[name][row].as_py() for name in ("tasks", "length", "data/chunk_index", "data/file_index")}

    tasks = pd.read_parquet(root / DEFAULT_TASKS_PATH)
    known: dict[str, int] = dict(zip(tasks.index.tolist(), tasks[TASK_INDEX].tolist(), strict=True))
    is_new_task = task not in known
    # LeRobot appends a new task at position len(tasks) too (LeRobotDatasetMetadata.save_episode_tasks).
    task_index = len(tasks) if is_new_task else known[task]

    data_path = root / info["data_path"].format(
        chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"]
    )
    data = pq.read_table(data_path)
    in_episode = data[_EPISODE_INDEX].to_numpy() == episode_index
    frame_task_indices = data[TASK_INDEX].to_numpy()
    frames = int(in_episode.sum())
    if frames != episode["length"]:
        raise ValueError(
            f"Episode {episode_index} has {frames} frames in {data_path.name}, "
            f"but its metadata says {episode['length']}"
        )

    already_set = not is_new_task and episode["tasks"] == [task]
    if already_set and bool((frame_task_indices[in_episode] == task_index).all()):
        return False

    staged: dict[Path, Path] = {}
    try:
        if is_new_task:
            new_task = pd.DataFrame({TASK_INDEX: [task_index]}, index=pd.Index([task], name=tasks.index.name))
            tasks = pd.concat([tasks, new_task])
            staged[root / DEFAULT_TASKS_PATH] = _stage(root / DEFAULT_TASKS_PATH, tasks.to_parquet)

        data = _set_column(data, TASK_INDEX, pa.array(np.where(in_episode, task_index, frame_task_indices)))
        staged[data_path] = _stage(data_path, partial(pq.write_table, data, compression="snappy"))

        episodes = _set_row(episodes, row, "tasks", [task])
        for name in episodes.column_names:
            if name.startswith(_TASK_STATS_PREFIX) and not name.endswith("/count"):
                # Every frame of the episode now has the same task_index.
                value = 0.0 if name.endswith("/std") else float(task_index)
                episodes = _set_row(episodes, row, name, [value])
        staged[episodes_path] = _stage(episodes_path, partial(pq.write_table, episodes))

        stats_path = root / STATS_PATH
        if stats_path.is_file():
            stats = json.loads(stats_path.read_text())
            if TASK_INDEX in stats and (aggregated := _aggregate_task_stats(root, episodes_path, episodes)):
                stats[TASK_INDEX] = aggregated
                staged[stats_path] = _stage(stats_path, partial(_write_json, stats))

        info["total_tasks"] = len(tasks)
        staged[root / INFO_PATH] = _stage(root / INFO_PATH, partial(_write_json, info))
    except BaseException:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        raise

    _swap(staged)
    return True


def _swap(staged: dict[Path, Path]) -> None:
    """Swap the staged files in, in order (dicts keep it: info.json last); on a failure put back those swapped."""
    swapped: list[tuple[Path, Path]] = []
    try:
        for target, temporary in staged.items():
            backup = target.with_name(f".{target.name}.{uuid4().hex}.bak")
            _link_or_copy(target, backup)
            try:
                temporary.replace(target)
            except BaseException:
                backup.unlink(missing_ok=True)
                raise
            swapped.append((target, backup))
    except BaseException:
        for target, backup in reversed(swapped):
            with suppress(OSError):
                backup.replace(target)
        for temporary in staged.values():
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        raise
    for _target, backup in swapped:
        with suppress(OSError):
            backup.unlink()


def _link_or_copy(source: Path, destination: Path) -> None:
    """A backup of ``source`` that survives replacing it: a hard link, or a copy where links are not supported."""
    try:
        os.link(source, destination)
    except OSError:
        with _removed_on_error(destination):
            shutil.copy2(source, destination)


def _find_episodes_file(root: Path, episode_index: int) -> tuple[Path, pa.Table]:
    for path in sorted((root / EPISODES_DIR).glob("*/*.parquet")):
        if episode_index in pq.read_table(path, columns=[_EPISODE_INDEX])[_EPISODE_INDEX].to_pylist():
            return path, pq.read_table(path)
    raise EpisodeNotFoundError(f"Episode {episode_index} not found")


def _aggregate_task_stats(root: Path, edited_path: Path, edited: pa.Table) -> dict[str, list] | None:
    """``task_index`` stats over every episode, as LeRobot aggregates them into ``meta/stats.json``."""
    # Typed loosely: LeRobot annotates the per-episode stats as dicts, but they are arrays.
    per_episode: list[dict[str, Any]] = []
    for path in sorted((root / EPISODES_DIR).glob("*/*.parquet")):
        table = edited if path == edited_path else pq.read_table(path)
        columns = [name for name in table.column_names if name.startswith(_TASK_STATS_PREFIX)]
        if not columns:
            return None
        rows = table.select(columns).to_pylist()
        per_episode.extend(
            {name.removeprefix(_TASK_STATS_PREFIX): np.asarray(value) for name, value in row.items()} for row in rows
        )
    if not per_episode:
        return None
    return {name: np.asarray(value).tolist() for name, value in aggregate_feature_stats(per_episode).items()}


def _set_column(table: pa.Table, name: str, values: pa.Array | pa.ChunkedArray) -> pa.Table:
    position = table.schema.get_field_index(name)
    field = table.schema.field(position)
    return table.set_column(position, field, values.cast(field.type))


def _set_row(table: pa.Table, row: int, name: str, value: object) -> pa.Table:
    values = table[name].to_pylist()
    values[row] = value
    return _set_column(table, name, pa.array(values, type=table.schema.field(name).type))


def _write_json(data: dict[str, Any], path: Path) -> None:
    # Same layout as lerobot.utils.io_utils.write_json.
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False))


def _stage(target: Path, write: Callable[[Path], object]) -> Path:
    """Write ``target``'s replacement to a hidden file next to it; LeRobot's ``*.parquet`` globs skip it."""
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    with _removed_on_error(temporary):
        write(temporary)
        _fsync(temporary)
    return temporary


@contextmanager
def _removed_on_error(path: Path) -> Iterator[None]:
    try:
        yield
    except BaseException:
        with suppress(OSError):
            path.unlink(missing_ok=True)
        raise


def _fsync(path: Path) -> None:
    """Flush to disk before the swap, so a power loss cannot leave an empty file in its place."""
    # Windows flushes only a handle opened for writing.
    descriptor = os.open(path, os.O_RDWR)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
