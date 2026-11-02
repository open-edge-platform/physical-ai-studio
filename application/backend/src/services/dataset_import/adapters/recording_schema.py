"""Helpers for inferring recording schema (cameras, robots) from LeRobot info.json features."""

from __future__ import annotations

from loguru import logger

from schemas.dataset_import_job import DatasetManifestRecordingSchema, ManifestCameraEntry, ManifestRobotEntry


def extract_recording_schema(info: dict) -> DatasetManifestRecordingSchema:
    """Parse LeRobot ``meta/info.json`` content and return a :class:`DatasetManifestRecordingSchema`.

    Extraction is best-effort: any missing or malformed fields are silently
    skipped so that a bad feature entry never blocks the import flow.

    Camera detection
    ----------------
    Any feature whose key matches ``observation.images.<camera_name>`` and
    whose ``dtype`` is ``"video"`` is treated as a camera stream.  Width,
    height and fps are read from the nested ``info.video.*`` sub-dict when
    present; as a fallback the spatial dimensions are taken from ``shape``
    and fps from the top-level ``fps`` field.

    Robot / joint detection
    -----------------------
    A single robot entry is produced when ``action`` or
    ``observation.state`` features contain ``names`` lists.  Joint names are
    derived by stripping everything from the first ``.`` onward
    (e.g. ``shoulder_pan.pos`` → ``shoulder_pan``), with duplicates removed
    while preserving insertion order.

    The ``robot_type`` top-level field is used as ``type`` when present.
    """
    try:
        features: dict = {}
        if isinstance(info.get("features"), dict):
            features = info["features"]

        dataset_fps: int | None = None
        fps_raw = info.get("fps")
        if fps_raw is not None:
            dataset_fps = int(float(fps_raw))

        robot_type_raw = info.get("robot_type")
        robot_type: str | None = str(robot_type_raw).strip() if robot_type_raw else None

        cameras = _extract_cameras(features, dataset_fps)
        robots = _extract_robots(features, robot_type)
        return DatasetManifestRecordingSchema(cameras=cameras, robots=robots)
    except Exception as exc:
        logger.debug("recording_schema: unexpected error during extraction, returning empty schema: {}", exc)
        return DatasetManifestRecordingSchema()


def _extract_cameras(features: dict, dataset_fps: int | None) -> list[ManifestCameraEntry]:
    cameras: list[ManifestCameraEntry] = []
    prefix = "observation.images."

    for key, feature in features.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        if not isinstance(feature, dict):
            continue
        if feature.get("dtype") != "video":
            continue

        camera_name = key[len(prefix) :]
        if not camera_name:
            continue

        width: int | None = None
        height: int | None = None
        fps: int | None = dataset_fps

        # Prefer info.video.* for precise values
        video_info = feature.get("info", {})
        if isinstance(video_info, dict):
            video_section = video_info.get("video", {})
            if isinstance(video_section, dict):
                width = _safe_int(video_section.get("width")) or width
                height = _safe_int(video_section.get("height")) or height
                fps = _safe_int(video_section.get("fps")) or fps

        # Fallback: derive from shape [height, width, channels] if available
        if width is None or height is None:
            shape = feature.get("shape")
            if isinstance(shape, list) and len(shape) >= 2:
                height = height or _safe_int(shape[0])
                width = width or _safe_int(shape[1])

        cameras.append(ManifestCameraEntry(name=camera_name, width=width, height=height, fps=fps))

    return cameras


def _extract_robots(features: dict, robot_type: str | None) -> list[ManifestRobotEntry]:
    joints: list[str] = []
    seen: set[str] = set()

    # Collect joint names from action and observation.state features
    for key in ("action", "observation.state"):
        feature = features.get(key)
        if not isinstance(feature, dict):
            continue
        names = feature.get("names")
        if not isinstance(names, list):
            continue
        for raw_name in names:
            if not isinstance(raw_name, str):
                continue
            joint = raw_name.split(".")[0].strip()
            if joint and joint not in seen:
                seen.add(joint)
                joints.append(joint)

    if not joints and robot_type is None:
        # Nothing to describe - return empty list rather than a placeholder robot
        return []

    robot = ManifestRobotEntry(
        type=robot_type,
        joints=joints,
    )
    return [robot]


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
