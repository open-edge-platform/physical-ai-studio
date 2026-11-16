import { useQueryClient } from '@tanstack/react-query';

import { $api } from '../../../api/client';
import { getQueryKey } from '../../../query-client/query-client.interface';
import { getEpisodesQueryKey } from './episodes-query-key';

export const useUpdateEpisodeTask = (dataset_id: string, episode_index: number) => {
    const queryClient = useQueryClient();
    const episodeQueryKey = getQueryKey([
        'get',
        '/api/dataset/{dataset_id}/episodes/{episode_index}',
        { params: { path: { dataset_id, episode_index } } },
    ]);

    return $api.useMutation('patch', '/api/dataset/{dataset_id}/episodes/{episode_index}', {
        meta: {
            // Invalidating the episode cancels a fetch that started before the save finished, which would bring
            // back the old task, and starts a new one.
            invalidates: [getEpisodesQueryKey(dataset_id), episodeQueryKey],
            // Resolve only once the viewer shows the new task, so the editor closes onto it.
            awaits: [episodeQueryKey],
        },
        // Runs after the refetch: if that failed, the viewer still shows the saved task.
        onSuccess: (episode) => queryClient.setQueryData(episodeQueryKey, episode),
    });
};
