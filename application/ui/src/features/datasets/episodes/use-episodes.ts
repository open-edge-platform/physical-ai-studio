import { useQueryClient } from '@tanstack/react-query';

import { $api } from '../../../api/client';

export const useDeleteEpisodeQuery = (dataset_id: string) => {
    const queryClient = useQueryClient();

    const mutation = $api.useMutation('delete', '/api/dataset/{dataset_id}/episodes', {
        meta: {
            invalidates: [
                ['get', '/api/dataset/{dataset_id}/episodes', { params: { path: { dataset_id } } }],
                ['get', '/api/dataset/{dataset_id}', { params: { path: { dataset_id } } }],
            ],
        },
        onSuccess: (data) => {
            const query_key = [
                'get',
                '/api/dataset/{dataset_id}/episodes',
                {
                    params: {
                        path: {
                            dataset_id,
                        },
                    },
                },
            ];
            queryClient.setQueryData(query_key, data);
        },
    });

    const deleteEpisodes = (episodeIndices: number[]) => {
        mutation.mutate({
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
