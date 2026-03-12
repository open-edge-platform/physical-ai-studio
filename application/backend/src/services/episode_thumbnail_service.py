import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import format_datetime
from uuid import UUID

from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset


@dataclass(frozen=True)
class EpisodeThumbnail:
    content: bytes
    etag: str
    last_modified: str


class EpisodeThumbnailService:
    def get_thumbnail(
        self,
        dataset_id: UUID,
        dataset: InternalLeRobotDataset,
        episode_index: int,
        camera: str | None = None,
        width: int = 320,
        height: int = 240,
    ) -> EpisodeThumbnail | None:
        video_key = self._resolve_video_key(dataset, camera)
        if video_key is None:
            return None

        result = dataset.get_episode_thumbnail_png(episode_index, video_key, width, height)
        if result is None:
            return None

        thumbnail_bytes, video_path = result
        video_stat = video_path.stat()
        etag_payload = (
            f"{dataset_id}:{episode_index}:{video_key}:{width}x{height}:{video_stat.st_mtime_ns}:{video_stat.st_size}"
        )
        etag = hashlib.sha256(etag_payload.encode()).hexdigest()
        last_modified = format_datetime(datetime.fromtimestamp(video_stat.st_mtime, tz=UTC), usegmt=True)
        return EpisodeThumbnail(content=thumbnail_bytes, etag=f'"{etag}"', last_modified=last_modified)

    def _resolve_video_key(self, dataset: InternalLeRobotDataset, camera: str | None) -> str | None:
        video_keys = dataset.get_video_keys()
        if len(video_keys) == 0:
            return None

        if camera is None:
            return video_keys[0]

        if camera in video_keys:
            return camera

        prefixed_camera = f"observation.images.{camera}"
        if prefixed_camera in video_keys:
            return prefixed_camera

        return None
