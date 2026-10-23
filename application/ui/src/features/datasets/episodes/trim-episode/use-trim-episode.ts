import { $api } from '../../../../api/client';

export const useTrimEpisodeQuery = (dataset_id: string, episode_index: number) => {
    const mutation = $api.useMutation('put', '/api/dataset/{dataset_id}/episodes/{episode_index}', {
        meta: {
            invalidates: [
                ['get', '/api/dataset/{dataset_id}/episodes', { params: { path: { dataset_id } } }],
                [
                    'get',
                    '/api/dataset/{dataset_id}/episodes/{episode_index}',
                    { params: { path: { dataset_id, episode_index } } },
                ],
            ],
        },
    });

    const trimEpisode = (start_frame: number, end_frame: number) =>
        mutation.mutateAsync({
            params: { path: { dataset_id, episode_index } },
            body: { start_frame, end_frame },
        });

    return { ...mutation, trimEpisode };
};
