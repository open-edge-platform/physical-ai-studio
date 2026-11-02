import { useEffect, useRef } from 'react';

import { $api } from '../../api/client';
import { issueToasts } from '../../components/notification/issue-toasts';

const REMOTE_UNAVAILABLE_POLL_MS = 15000;
const REMOTE_STARTING_POLL_MS = 5000;

/**
 * Reads a remote trainer's health and retries unavailable trainers until they recover.
 */
export const useRemoteTrainerHealth = (remoteTrainerId: string | null) => {
    const observer = useRef<ReturnType<typeof issueToasts.observe> | undefined>(undefined);
    const query = $api.useQuery(
        'get',
        '/api/remote-trainers/{remote_trainer_id}/health',
        { params: { path: { remote_trainer_id: remoteTrainerId ?? '' } } },
        {
            enabled: remoteTrainerId !== null,
            refetchOnMount: 'always',
            refetchInterval: (healthQuery) => {
                const status = healthQuery.state.data?.status;
                if (status === 'starting') return REMOTE_STARTING_POLL_MS;
                return status === 'unreachable' ||
                    healthQuery.state.data?.issues?.length ||
                    issueToasts.hasIssues(`trainer:${remoteTrainerId}`)
                    ? REMOTE_UNAVAILABLE_POLL_MS
                    : false;
            },
        }
    );
    useEffect(() => {
        if (remoteTrainerId === null) return;
        const subscription = issueToasts.observe(`trainer:${remoteTrainerId}`);
        observer.current = subscription;
        return () => {
            subscription.unsubscribe();
            observer.current = undefined;
        };
    }, [remoteTrainerId]);
    useEffect(() => {
        if (query.data?.remote_trainer_id === remoteTrainerId) observer.current?.update(query.data.issues);
    }, [query.data, remoteTrainerId]);

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
