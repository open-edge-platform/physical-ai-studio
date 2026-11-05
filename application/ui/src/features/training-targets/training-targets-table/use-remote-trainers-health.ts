import { useEffect, useRef } from 'react';

import { useQueries } from '@tanstack/react-query';

import { $api } from '../../../api/client';
import { SchemaRemoteTrainerHealth } from '../../../api/openapi-spec';
import { issueToasts } from '../../../components/notification/issue-toasts';

const REMOTE_UNAVAILABLE_POLL_MS = 15000;
const REMOTE_STARTING_POLL_MS = 5000;

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
    const observers = useRef(new Map<string, ReturnType<typeof issueToasts.observe>>());
    const queries = useQueries({
        queries: remoteTrainerIds.map((remoteTrainerId) =>
            $api.queryOptions(
                'get',
                '/api/remote-trainers/{remote_trainer_id}/health',
                { params: { path: { remote_trainer_id: remoteTrainerId } } },
                {
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
            )
        ),
    });
    useEffect(() => {
        const active = new Set(remoteTrainerIds);
        for (const [id, observer] of observers.current) {
            if (!active.has(id)) {
                observer.unsubscribe();
                observers.current.delete(id);
            }
        }
        queries.forEach((query, index) => {
            const id = remoteTrainerIds[index];
            let observer = observers.current.get(id);
            if (!observer) {
                observer = issueToasts.observe(`trainer:${id}`);
                observers.current.set(id, observer);
            }
            if (query.data?.remote_trainer_id === id) observer.update(query.data.issues);
        });
    }, [queries, remoteTrainerIds]);
    useEffect(() => {
        const subscriptions = observers.current;
        return () => {
            for (const observer of subscriptions.values()) observer.unsubscribe();
            subscriptions.clear();
        };
    }, []);

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
