import { useQueryClient } from '@tanstack/react-query';

import { $api } from '../../../api/client';
import { getEpisodesQueryKey } from './episodes-query-key';

export const useDeleteEpisodeQuery = (dataset_id: string) => {
    const queryClient = useQueryClient();

    const mutation = $api.useMutation('delete', '/api/dataset/{dataset_id}/episodes', {
        meta: {
            invalidates: [
                getEpisodesQueryKey(dataset_id),
                ['get', '/api/dataset/{dataset_id}', { params: { path: { dataset_id } } }],
            ],
        },
        onSuccess: (data) => {
            const query_key = getEpisodesQueryKey(dataset_id);
            queryClient.setQueryData(query_key, data);
        },
    });

    const deleteEpisodes = (episodeIndices: number[]) => {
        return mutation.mutateAsync({
            params: {
                path: {
                    dataset_id,
                },
            },
            body: episodeIndices,
        });
    };

    return {
        ...mutation,
        deleteEpisodes,
    };
};
