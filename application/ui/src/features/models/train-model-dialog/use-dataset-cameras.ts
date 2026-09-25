import { useMemo } from 'react';

import { $api } from '../../../api/client';

/** LeRobot prefixes every camera feature with this. */
export const IMAGE_KEY_PREFIX = 'observation.images.';

export interface DatasetCamera {
    /** Feature key as stored in the dataset, e.g. `observation.images.top`. */
    key: string;
    /** The camera's name without the LeRobot prefix, e.g. `top`. */
    name: string;
}

interface UseDatasetCamerasResult {
    cameras: DatasetCamera[];
    isLoading: boolean;
    /** The dataset holds no episodes yet, so its cameras are unknown. */
    isEmpty: boolean;
}

/**
 * The cameras a dataset was recorded with, in the order they appear in the dataset.
 *
 * There is no endpoint for a dataset's features, but every episode carries its
 * video keys, so the first episode is what the camera list is read from.
 */
export const useDatasetCameras = (datasetId: string | undefined): UseDatasetCamerasResult => {
    const { data: episodes = [], isLoading: isLoadingEpisodes } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}/episodes',
        { params: { path: { dataset_id: datasetId ?? '' } } },
        { enabled: datasetId !== undefined }
    );

    const firstEpisodeIndex = episodes.at(0)?.episode_index;

    const { data: episode, isLoading: isLoadingEpisode } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}/episodes/{episode_index}',
        { params: { path: { dataset_id: datasetId ?? '', episode_index: firstEpisodeIndex ?? 0 } } },
        { enabled: datasetId !== undefined && firstEpisodeIndex !== undefined }
    );

    const cameras = useMemo(
        () =>
            Object.keys(episode?.videos ?? {}).map((key) => ({
                key,
                name: key.startsWith(IMAGE_KEY_PREFIX) ? key.slice(IMAGE_KEY_PREFIX.length) : key,
            })),
        [episode]
    );

    return {
        cameras,
        isLoading: isLoadingEpisodes || (firstEpisodeIndex !== undefined && isLoadingEpisode),
        isEmpty: !isLoadingEpisodes && episodes.length === 0,
    };
};
