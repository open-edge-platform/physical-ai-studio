import { useQueries } from '@tanstack/react-query';

import { $api } from '../../../api/client';
import { SchemaRemoteTrainerHealth } from '../../../api/openapi-spec';

const REMOTE_UNAVAILABLE_POLL_MS = 5000;

type RemoteTrainerHealthEntry = {
    health?: SchemaRemoteTrainerHealth;
    isChecking: boolean;
    hasError: boolean;
    checkHealth: () => Promise<SchemaRemoteTrainerHealth | null>;
};

/**
 * Batch-fetches health for a set of remote trainers, sharing the query cache with
 * `useRemoteTrainerHealth` (identical query key), and retries unreachable trainers until they recover.
 */
export const useRemoteTrainersHealth = (remoteTrainerIds: string[]): Map<string, RemoteTrainerHealthEntry> => {
    const queries = useQueries({
        queries: remoteTrainerIds.map((remoteTrainerId) =>
            $api.queryOptions(
                'get',
                '/api/remote-trainers/{remote_trainer_id}/health',
                { params: { path: { remote_trainer_id: remoteTrainerId } } },
                {
                    refetchOnMount: 'always',
                    refetchInterval: (healthQuery) =>
                        healthQuery.state.data?.status === 'unreachable' ? REMOTE_UNAVAILABLE_POLL_MS : false,
                }
            )
        ),
    });

    return new Map(
        remoteTrainerIds.map((remoteTrainerId, index) => {
            const query = queries[index];
            return [
                remoteTrainerId,
                {
                    health: query.data,
                    isChecking: query.isPending,
                    hasError: query.isError,
                    checkHealth: async () => {
                        const result = await query.refetch();
                        return result.isError ? null : (result.data ?? null);
                    },
                },
            ];
        })
    );
};
