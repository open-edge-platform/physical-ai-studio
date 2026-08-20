from internal_datasets.utils import get_internal_read_dataset
from schemas import Project

from .episode_thumbnail_service import EpisodeThumbnail, EpisodeThumbnailService


class ProjectThumbnailService:
    def __init__(self, episode_thumbnail_service: EpisodeThumbnailService) -> None:
        self._episode_thumbnail_service = episode_thumbnail_service

    def get_thumbnail(self, project: Project, width: int = 320, height: int = 240) -> EpisodeThumbnail | None:
        if len(project.datasets) == 0:
            return None

        dataset = project.datasets[0]
        internal_dataset = get_internal_read_dataset(dataset)
        episode_infos = internal_dataset.get_episode_infos()

        if len(episode_infos) == 0:
            return None

        episode_index = episode_infos[0].episode_index
        return self._episode_thumbnail_service.get_thumbnail(
            dataset_id=dataset.id,
            dataset=internal_dataset,
            episode_index=episode_index,
            width=width,
            height=height,
        )
