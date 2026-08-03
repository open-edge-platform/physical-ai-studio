import { $api } from '../../api/client';

const REMOTE_UNAVAILABLE_POLL_MS = 5000;

/**
 * Reads a remote trainer's health and retries unavailable trainers until they recover.
 */
export const useRemoteTrainerHealth = (remoteTrainerId: string | null) => {
    const query = $api.useQuery(
        'get',
        '/api/remote-trainers/{remote_trainer_id}/health',
        { params: { path: { remote_trainer_id: remoteTrainerId ?? '' } } },
        {
            enabled: remoteTrainerId !== null,
            refetchOnMount: 'always',
            refetchInterval: (healthQuery) =>
                healthQuery.state.data?.status === 'unreachable' ? REMOTE_UNAVAILABLE_POLL_MS : false,
        }
    );

    return {
        health: query.data,
        isChecking: query.isFetching,
        hasError: query.isError,
        checkHealth: async () => {
            const result = await query.refetch();
            return result.isError ? null : (result.data ?? null);
        },
    };
};
