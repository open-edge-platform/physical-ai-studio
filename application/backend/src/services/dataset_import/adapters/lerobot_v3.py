from __future__ import annotations

import re
import shutil
from io import BytesIO
from uuid import UUID, uuid4

import pandas as pd
from loguru import logger

from exceptions import InvalidArchiveError
from schemas import Dataset
from schemas.dataset_import_job import (
    DatasetImportJobPayload,
    DatasetImportSource,
    DatasetManifest,
    DatasetManifestStatistics,
    ImportValidationReport,
)
from services.archive_safety import SafeZipArchive, check_disk_headroom, flatten_single_root_directory
from services.dataset_service import DatasetService
from settings import get_settings

from .base import DatasetImportAdapter
from .recording_schema import extract_recording_schema


class LeRobotV3Adapter(DatasetImportAdapter):
    source = DatasetImportSource.LEROBOT_V3
    _EPISODE_PARQUET_PATTERN = re.compile(r"(?:^|/)meta/episodes/chunk-(\d+)/file-(\d+)\.parquet$")

    def _load_info(self, archive: SafeZipArchive, report: ImportValidationReport) -> dict:
        raw_info = archive.read_json("meta/info.json")
        if raw_info is None:
            report.add_error("Could not read required metadata file 'meta/info.json'.")
            return {}
        return raw_info

    def _load_episode_counts(  # noqa: C901, PLR0912, PLR0915
        self,
        archive: SafeZipArchive,
        report: ImportValidationReport,
        info: dict | None = None,
    ) -> tuple[int, int]:
        episode_count = 0
        frame_count = 0

        episode_shards: list[tuple[int, int, str]] = []
        for name in archive.iter_normalized_names():
            if match := self._EPISODE_PARQUET_PATTERN.search(name):
                episode_shards.append((int(match.group(1)), int(match.group(2)), name))

        if not episode_shards:
            report.add_error("No episode parquet found under 'meta/episodes/chunk-*/file-*.parquet'.")
            return episode_count, frame_count

        episode_shards.sort(key=lambda item: (item[0], item[1], item[2]))

        readable_shard_count = 0
        shard_count_with_episode_index = 0
        shard_count_without_episode_index = 0
        shard_count_without_length = 0
        episode_rows_total = 0
        episode_rows_without_episode_index = 0
        unique_episode_indices: set[int] = set()

        for _chunk_index, _file_index, shard_name in episode_shards:
            shard_bytes = archive.read_bytes(shard_name)
            if shard_bytes is None:
                report.add_warning(f"Could not read episode parquet '{shard_name}'.")
                continue

            try:
                episodes_df = pd.read_parquet(BytesIO(shard_bytes))
            except Exception as error:
                report.add_warning(
                    f"Could not parse episode parquet '{shard_name}' ({type(error).__name__}); skipping this shard."
                )
                continue

            readable_shard_count += 1
            row_count = len(episodes_df)
            episode_rows_total += row_count

            if "episode_index" in episodes_df.columns:
                shard_count_with_episode_index += 1
                episode_indexes = pd.to_numeric(episodes_df["episode_index"], errors="coerce").dropna()
                unique_episode_indices.update(episode_indexes.astype("int64").tolist())
            else:
                shard_count_without_episode_index += 1
                episode_rows_without_episode_index += row_count

            if "length" in episodes_df.columns:
                lengths = pd.to_numeric(episodes_df["length"], errors="coerce").fillna(0)
                frame_count += int(lengths.sum())
            else:
                shard_count_without_length += 1

        if readable_shard_count == 0:
            report.add_error("No readable episode parquet found under 'meta/episodes/chunk-*/file-*.parquet'.")
            return 0, 0

        if shard_count_with_episode_index == 0:
            episode_count = episode_rows_total
            report.add_warning(
                "Episode parquet shards are missing 'episode_index'; episode count is based on row count."
            )
        elif shard_count_without_episode_index == 0:
            episode_count = len(unique_episode_indices)
        else:
            episode_count = len(unique_episode_indices) + episode_rows_without_episode_index
            report.add_warning(
                "Some episode parquet shards are missing 'episode_index'; episode count may be approximate."
            )

        if shard_count_without_length > 0:
            report.add_warning(
                f"{shard_count_without_length} episode parquet shard(s) are missing 'length';"
                " frame count may be incomplete."
            )

        if info:
            expected_episode_count = info.get("total_episodes")
            if isinstance(expected_episode_count, int) and expected_episode_count != episode_count:
                report.add_warning(
                    "Episode count mismatch: metadata reports "
                    f"{expected_episode_count}, parsed episode shards report {episode_count}."
                )

            expected_frame_count = info.get("total_frames")
            if isinstance(expected_frame_count, int) and expected_frame_count != frame_count:
                report.add_warning(
                    "Frame count mismatch: metadata reports "
                    f"{expected_frame_count}, parsed episode shards report {frame_count}."
                )

        return episode_count, frame_count

    def detect(self, archive: SafeZipArchive) -> tuple[bool, ImportValidationReport]:
        """Return (matched, report) for LeRobot v3 archives.

        v3 markers (all required):
          - ``meta/info.json``
          - ``meta/tasks.parquet``         (v3 task index; v2 uses tasks.jsonl)
          - ``data/chunk-*/file-*.parquet`` (v3 data layout; v2 uses episode_*.parquet)

        Explicitly reject if v2-specific markers are found:
          - ``meta/episodes.jsonl``  (v2 only)
          - ``meta/tasks.jsonl``     (v2 only)
        """
        report = ImportValidationReport()
        has_info = False
        has_tasks_parquet = False
        has_v3_data = False
        rejected_v2_markers: list[str] = []

        for name in archive.iter_normalized_names():
            # Reject v2-only markers immediately (handle both flat and nested archives)
            if name == "meta/episodes.jsonl" or name.endswith("/meta/episodes.jsonl"):
                rejected_v2_markers.append("'meta/episodes.jsonl'")
                break
            if name == "meta/tasks.jsonl" or name.endswith("/meta/tasks.jsonl"):
                rejected_v2_markers.append("'meta/tasks.jsonl'")
                break

            if not has_info and (name == "meta/info.json" or name.endswith("/meta/info.json")):
                has_info = True
            if not has_tasks_parquet and (name == "meta/tasks.parquet" or name.endswith("/meta/tasks.parquet")):
                has_tasks_parquet = True
            if (
                not has_v3_data
                and ("data/" in name or name.startswith("data/"))
                and "/file-" in name
                and name.endswith(".parquet")
                and "episode_" not in name
            ):
                has_v3_data = True

            if has_info and has_tasks_parquet and has_v3_data:
                return True, report

        if rejected_v2_markers:
            report.add_error(
                f"Archive contains LeRobot v2 markers ({', '.join(rejected_v2_markers)}); "
                "use the lerobot_v2 dataset format instead."
            )
            return False, report

        missing = []
        if not has_info:
            missing.append("'meta/info.json'")
        if not has_tasks_parquet:
            missing.append("'meta/tasks.parquet'")
        if not has_v3_data:
            missing.append("v3 data files ('data/chunk-*/file-*.parquet')")

        if missing:
            report.add_error(f"LeRobot v3 markers not found: {', '.join(missing)}.")

        return False, report

    def build_draft(
        self,
        archive: SafeZipArchive,
        payload: DatasetImportJobPayload,  # noqa: ARG002
    ) -> tuple[DatasetManifest, ImportValidationReport]:
        report = ImportValidationReport()
        info: dict = {}
        episode_count = 0
        frame_count = 0

        try:
            info = self._load_info(archive=archive, report=report)
            episode_count, frame_count = self._load_episode_counts(archive=archive, report=report, info=info)

            if archive.read_json("meta/stats.json") is None:
                report.add_warning("No global stats metadata found in 'meta/stats.json'.")

            if archive.read_bytes("meta/tasks.parquet") is None:
                report.add_warning("No tasks index found in 'meta/tasks.parquet'.")

        except (ValueError, InvalidArchiveError) as error:
            logger.debug("Could not read LeRobot v3 info from '{}': {}", archive.path, error)
            report.add_error(f"Unable to parse core dataset metadata from archive ('{type(error).__name__}').")

        if episode_count == 0:
            report.add_warning("Detected 0 episodes from metadata.")
        if frame_count == 0 and episode_count > 0:
            report.add_warning("Detected episodes but total frame count is 0.")

        recording_schema = extract_recording_schema(info)
        if not recording_schema.cameras:
            report.add_warning("No camera streams inferred from dataset metadata.")
        if not recording_schema.robots:
            report.add_warning("No robot schema inferred from dataset metadata.")

        logger.info(
            "LeRobotV3Adapter manifest draft parsed: archive='{}'",
            archive.path,
        )

        manifest = DatasetManifest(
            source_type=DatasetImportSource.LEROBOT_V3,
            statistics=DatasetManifestStatistics(
                episode_count=episode_count,
                frame_count=frame_count,
            ),
            dataset_schema=recording_schema,
        )

        return manifest, report

    def validate_pre_commit(self, payload: DatasetImportJobPayload) -> ImportValidationReport:
        report = ImportValidationReport()

        if payload.finalize_input is None:
            report.add_error("Finalize input is required before commit")

        return report

    async def commit(self, payload: DatasetImportJobPayload, project_id: UUID, archive: SafeZipArchive) -> Dataset:
        if payload.finalize_input is None:
            raise ValueError("Cannot commit dataset import without finalize input")

        settings = get_settings()
        settings.datasets_dir.mkdir(parents=True, exist_ok=True)

        dataset_id = uuid4()
        destination_dir = settings.datasets_dir / str(dataset_id)
        destination_dir.mkdir(parents=True, exist_ok=False)
        logger.info(
            "LeRobotV3Adapter commit destination prepared: dataset_id='{}', destination_dir='{}'",
            dataset_id,
            destination_dir,
        )

        check_disk_headroom(
            settings.datasets_dir,
            required_bytes=archive.estimated_uncompressed_size(),
            min_free_bytes=settings.data_import_min_free_bytes,
        )

        try:
            extracted_count = archive.extract_to(
                destination_dir,
                min_free_bytes=settings.data_import_min_free_bytes,
            )

            # Allow users to upload a zip with the dataset either at archive root
            # or inside a single top-level folder.
            flatten_single_root_directory(destination_dir)

            logger.info(
                "LeRobotV3Adapter extracted archive: archive='{}', destination_dir='{}', file_count={}",
                archive.path,
                destination_dir,
                extracted_count,
            )

            dataset = Dataset(
                id=dataset_id,
                name=payload.dataset_name or "Imported Dataset",
                path=str(destination_dir),
                default_task=payload.finalize_input.default_task,
                project_id=project_id,
                environment_id=payload.finalize_input.environment_id,
            )

            saved = await DatasetService.create_dataset(dataset)
            logger.info(
                "LeRobotV3Adapter dataset persisted: dataset_id='{}', project_id='{}', environment_id='{}', path='{}'",
                saved.id,
                saved.project_id,
                saved.environment_id,
                saved.path,
            )

            return saved
        except Exception:
            shutil.rmtree(destination_dir, ignore_errors=True)
            raise
