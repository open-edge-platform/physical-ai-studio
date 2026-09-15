import { useCallback, useEffect } from 'react';

import { useNavigate, useParams } from 'react-router';

import { paths } from '../../router';
import { useDataset } from './dataset-provider';

export function useActiveEpisode(): [number | null, (idx: number) => void] {
    const navigate = useNavigate();
    const { project_id, dataset_id, episode_index } = useParams<{
        project_id: string;
        dataset_id: string;
        episode_index: string;
    }>();
    const { episodes } = useDataset();

    if (project_id === undefined) {
        throw new Error('Unknown project_id parameter');
    }
    if (dataset_id === undefined) {
        throw new Error('Unknown dataset_id parameter');
    }

    const parsedIndex = episode_index === undefined ? null : Number(episode_index);
    const isValid =
        parsedIndex !== null &&
        !Number.isNaN(parsedIndex) &&
        episodes.some((episode) => episode.episode_index === parsedIndex);

    const activeEpisodeIndex = isValid ? parsedIndex : null;

    useEffect(() => {
        if (episodes.length > 0 && !isValid) {
            navigate(
                paths.project.datasets.showEpisode({
                    project_id,
                    dataset_id,
                    episode_index: String(episodes[0].episode_index),
                }),
                { replace: true }
            );
        }
    }, [episodes, isValid, project_id, dataset_id, navigate]);

    const setActiveEpisodeIndex = useCallback(
        (idx: number) => {
            navigate(
                paths.project.datasets.showEpisode({
                    project_id,
                    dataset_id,
                    episode_index: String(idx),
                })
            );
        },
        [navigate, project_id, dataset_id]
    );

    return [activeEpisodeIndex, setActiveEpisodeIndex];
}
