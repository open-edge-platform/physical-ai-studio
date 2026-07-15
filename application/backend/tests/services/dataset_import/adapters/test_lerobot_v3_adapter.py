import json
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
from physicalai.data.archive_safety import SafeZipArchive

from schemas.dataset_import_job import ImportValidationSeverity
from services.dataset_import.adapters.lerobot_v3 import LeRobotV3Adapter


def _write_zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _episodes_parquet_bytes(
    lengths: list[int],
    *,
    episode_start: int = 0,
    include_episode_index: bool = True,
    include_length: bool = True,
) -> bytes:
    data: dict[str, list[int]] = {}
    if include_episode_index:
        data["episode_index"] = list(range(episode_start, episode_start + len(lengths)))
    if include_length:
        data["length"] = lengths
    df = pd.DataFrame(data)
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


def test_lerobot_v3_detect_returns_true_for_v3_layout(tmp_path: Path) -> None:
    """v3 archive with tasks.parquet and file-* data layout is detected."""
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "v3.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": json.dumps({"codebase_version": "v3.0"}).encode("utf-8"),
            "meta/tasks.parquet": b"PAR1",
            "data/chunk-000/file-000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    matched, _report = adapter.detect(safe_archive)
    assert matched is True


def test_lerobot_v3_detect_returns_true_for_nested_v3_layout(tmp_path: Path) -> None:
    """v3 archive with dataset root inside single top-level folder is detected."""
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "v3-nested.zip"
    _write_zip(
        archive_path,
        {
            "dataset/meta/info.json": json.dumps({"codebase_version": "v3.0"}).encode("utf-8"),
            "dataset/meta/tasks.parquet": b"PAR1",
            "dataset/data/chunk-000/file-000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    matched, _report = adapter.detect(safe_archive)
    assert matched is True


def test_lerobot_v3_detect_returns_false_for_v2_layout(tmp_path: Path) -> None:
    """v2 archive with tasks.jsonl and episode_*.parquet is NOT detected as v3."""
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "v2.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": b'{"codebase_version":"v2.1"}',
            "meta/tasks.jsonl": b'{"task_index":0,"task":"pick"}\n',
            "meta/episodes.jsonl": b'{"episode_index":0,"tasks":["pick"],"length":1}\n',
            "data/chunk-000/episode_000000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    matched, _report = adapter.detect(safe_archive)
    assert matched is False


def test_lerobot_v3_detect_returns_false_without_tasks_parquet(tmp_path: Path) -> None:
    """Archive missing meta/tasks.parquet is NOT detected as v3."""
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "no-tasks.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": b'{"codebase_version":"v3.0"}',
            "data/chunk-000/file-000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    matched, _report = adapter.detect(safe_archive)
    assert matched is False


def test_lerobot_v3_detect_returns_false_without_v3_data_files(tmp_path: Path) -> None:
    """Archive with tasks.parquet but no file-* data is NOT detected as v3."""
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "no-data.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": b'{"codebase_version":"v3.0"}',
            "meta/tasks.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    matched, _report = adapter.detect(safe_archive)
    assert matched is False


def test_lerobot_v3_parse_manifest_reads_stats_from_nested_root_zip(tmp_path: Path) -> None:
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "v3-nested.zip"
    episodes_bytes = _episodes_parquet_bytes([11, 7, 5])
    _write_zip(
        archive_path,
        {
            "dataset/meta/info.json": json.dumps({"fps": 20}).encode("utf-8"),
            "dataset/meta/episodes/chunk-000/file-000.parquet": episodes_bytes,
            "dataset/data/chunk-000/file-000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    manifest, _report = adapter.build_draft(safe_archive, payload=MagicMock())

    assert manifest.statistics.episode_count == 3
    assert manifest.statistics.frame_count == 23


def test_lerobot_v3_parse_manifest_counts_all_episode_parquet_shards(tmp_path: Path) -> None:
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "v3-multi-episodes.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": json.dumps({"codebase_version": "v3.0", "fps": 20}).encode("utf-8"),
            "meta/tasks.parquet": b"PAR1",
            "meta/episodes/chunk-000/file-000.parquet": _episodes_parquet_bytes([11, 7], episode_start=0),
            "meta/episodes/chunk-000/file-001.parquet": _episodes_parquet_bytes([5], episode_start=2),
            "meta/episodes/chunk-001/file-000.parquet": _episodes_parquet_bytes([13, 9], episode_start=3),
            "data/chunk-000/file-000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    manifest, _report = adapter.build_draft(safe_archive, payload=MagicMock())

    assert manifest.statistics.episode_count == 5
    assert manifest.statistics.frame_count == 45


def test_lerobot_v3_parse_manifest_counts_episode_shards_when_file_000_is_missing(tmp_path: Path) -> None:
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "v3-missing-first-shard.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": json.dumps({"codebase_version": "v3.0", "fps": 20}).encode("utf-8"),
            "meta/tasks.parquet": b"PAR1",
            "meta/episodes/chunk-000/file-001.parquet": _episodes_parquet_bytes([3, 4], episode_start=0),
            "data/chunk-000/file-000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    manifest, _report = adapter.build_draft(safe_archive, payload=MagicMock())

    assert manifest.statistics.episode_count == 2
    assert manifest.statistics.frame_count == 7


def test_lerobot_v3_parse_manifest_schema_empty_when_no_features(tmp_path: Path) -> None:
    """When info.json has no features the recording schema is empty."""
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "v3-no-features.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": json.dumps({"fps": 10}).encode("utf-8"),
            "meta/tasks.parquet": b"PAR1",
            "data/chunk-000/file-000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    manifest, _report = adapter.build_draft(safe_archive, payload=MagicMock())

    assert manifest.dataset_schema.cameras == []
    assert manifest.dataset_schema.robots == []


def test_lerobot_v3_parse_manifest_schema_two_cameras_six_joints(tmp_path: Path) -> None:
    """Realistic v3 feature layout yields two cameras and six joints on one robot."""
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "v3-full.zip"

    info = {
        "fps": 30,
        "robot_type": "so100",
        "features": {
            "observation.images.top": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "info": {"video": {"fps": 30, "width": 640, "height": 480}},
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "info": {"video": {"fps": 30, "width": 640, "height": 480}},
            },
            "action": {
                "dtype": "float32",
                "shape": [6],
                "names": [
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                    "gripper.pos",
                ],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [6],
                "names": [
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                    "gripper.pos",
                ],
            },
        },
    }

    _write_zip(
        archive_path,
        {
            "meta/info.json": json.dumps(info).encode("utf-8"),
            "meta/tasks.parquet": b"PAR1",
            "data/chunk-000/file-000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    manifest, _report = adapter.build_draft(safe_archive, payload=MagicMock())

    schema = manifest.dataset_schema

    # Two cameras
    assert len(schema.cameras) == 2
    camera_names = {c.name for c in schema.cameras}
    assert camera_names == {"top", "wrist"}
    for cam in schema.cameras:
        assert cam.width == 640
        assert cam.height == 480
        assert cam.fps == 30

    # One robot with six joints (set equality - ordering not contractual)
    assert len(schema.robots) == 1
    robot = schema.robots[0]
    assert robot.type == "so100"
    assert set(robot.joints) == {
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    }
    assert len(robot.joints) == 6


def test_lerobot_v3_parse_manifest_schema_joints_deduped_across_action_and_state(tmp_path: Path) -> None:
    """Joint names appearing in both action and observation.state are deduplicated."""
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "v3-dedup.zip"

    info = {
        "fps": 10,
        "robot_type": "myrobot",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [2],
                "names": ["joint_a.pos", "joint_b.pos"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [2],
                "names": ["joint_a.pos", "joint_b.pos"],
            },
        },
    }

    _write_zip(
        archive_path,
        {
            "meta/info.json": json.dumps(info).encode("utf-8"),
            "meta/tasks.parquet": b"PAR1",
            "data/chunk-000/file-000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    manifest, _report = adapter.build_draft(safe_archive, payload=MagicMock())

    assert set(manifest.dataset_schema.robots[0].joints) == {"joint_a", "joint_b"}
    assert len(manifest.dataset_schema.robots[0].joints) == 2


def test_lerobot_v3_parse_manifest_schema_camera_falls_back_to_shape_dims(tmp_path: Path) -> None:
    """When video.info is absent, camera dimensions come from the feature shape."""
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "v3-shape-fallback.zip"

    info = {
        "fps": 15,
        "features": {
            "observation.images.front": {
                "dtype": "video",
                "shape": [240, 320, 3],
            },
        },
    }

    _write_zip(
        archive_path,
        {
            "meta/info.json": json.dumps(info).encode("utf-8"),
            "meta/tasks.parquet": b"PAR1",
            "data/chunk-000/file-000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    manifest, _report = adapter.build_draft(safe_archive, payload=MagicMock())

    cameras = manifest.dataset_schema.cameras
    assert len(cameras) == 1
    cam = cameras[0]
    assert cam.name == "front"
    assert cam.height == 240
    assert cam.width == 320
    assert cam.fps == 15  # fallback to dataset fps


def test_lerobot_v3_parse_manifest_does_not_include_suggested_name(tmp_path: Path) -> None:
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "opaque-staging-id.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": b'{"fps":30}',
            "meta/tasks.parquet": b"PAR1",
            "data/chunk-000/file-000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    manifest, _report = adapter.build_draft(safe_archive, payload=MagicMock())

    assert "suggested_name" not in manifest.model_dump()


def test_lerobot_v3_build_draft_reports_error_when_episode_parquet_missing(tmp_path: Path) -> None:
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "v3-no-episodes.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": json.dumps({"codebase_version": "v3.0", "fps": 20}).encode("utf-8"),
            "meta/tasks.parquet": b"PAR1",
            "data/chunk-000/file-000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    _manifest, report = adapter.build_draft(safe_archive, payload=MagicMock())

    assert any(msg.severity == ImportValidationSeverity.ERROR for msg in report.messages)
    assert any("episodes" in msg.message for msg in report.messages)


def test_lerobot_v3_build_draft_reports_warning_when_stats_missing(tmp_path: Path) -> None:
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "v3-no-stats.zip"
    episodes_bytes = _episodes_parquet_bytes([3])
    _write_zip(
        archive_path,
        {
            "meta/info.json": json.dumps({"codebase_version": "v3.0", "fps": 20}).encode("utf-8"),
            "meta/tasks.parquet": b"PAR1",
            "meta/episodes/chunk-000/file-000.parquet": episodes_bytes,
            "data/chunk-000/file-000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    _manifest, report = adapter.build_draft(safe_archive, payload=MagicMock())

    warning_messages = [msg.message for msg in report.messages if msg.severity == ImportValidationSeverity.WARNING]
    assert any("stats" in message for message in warning_messages)


def test_lerobot_v3_build_draft_reports_warning_when_counts_mismatch_info_json(tmp_path: Path) -> None:
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "v3-count-mismatch.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": json.dumps(
                {
                    "codebase_version": "v3.0",
                    "fps": 20,
                    "total_episodes": 99,
                    "total_frames": 999,
                }
            ).encode("utf-8"),
            "meta/tasks.parquet": b"PAR1",
            "meta/episodes/chunk-000/file-000.parquet": _episodes_parquet_bytes([3, 5], episode_start=0),
            "meta/episodes/chunk-000/file-001.parquet": _episodes_parquet_bytes([7], episode_start=2),
            "data/chunk-000/file-000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    _manifest, report = adapter.build_draft(safe_archive, payload=MagicMock())

    warning_messages = [msg.message for msg in report.messages if msg.severity == ImportValidationSeverity.WARNING]
    assert any("Episode count mismatch" in message for message in warning_messages)
    assert any("Frame count mismatch" in message for message in warning_messages)


def test_lerobot_v3_detect_report_includes_v2_marker_error(tmp_path: Path) -> None:
    """When detect returns False due to v2 markers, report should describe them."""
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "v2-for-v3-check.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": b'{"codebase_version":"v2.1"}',
            "meta/tasks.jsonl": b'{"task_index":0,"task":"pick"}\n',
            "meta/episodes.jsonl": b'{"episode_index":0,"tasks":["pick"],"length":1}\n',
            "data/chunk-000/episode_000000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    matched, report = adapter.detect(safe_archive)

    assert matched is False
    error_messages = [msg.message for msg in report.messages if msg.severity == ImportValidationSeverity.ERROR]
    assert error_messages, "Expected at least one error in detect report for v2 archive"
    combined = " ".join(error_messages)
    assert "v2" in combined.lower() or "lerobot" in combined.lower()


def test_lerobot_v3_detect_report_includes_missing_markers_error(tmp_path: Path) -> None:
    """When detect returns False due to missing markers, report should name them."""
    adapter = LeRobotV3Adapter()
    archive_path = tmp_path / "incomplete-v3.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": b'{"codebase_version":"v3.0"}',
            # missing tasks.parquet and data files
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    matched, report = adapter.detect(safe_archive)

    assert matched is False
    error_messages = [msg.message for msg in report.messages if msg.severity == ImportValidationSeverity.ERROR]
    assert error_messages
    combined = " ".join(error_messages)
    assert "tasks.parquet" in combined
