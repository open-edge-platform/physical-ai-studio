import { useQueries } from '@tanstack/react-query';

import { $api } from '../../../api/client';
import { SchemaRemoteServerStatus } from '../../../api/openapi-spec';

const REMOTE_UNAVAILABLE_POLL_MS = 5000;

type RemoteServerStatusEntry = {
    status?: SchemaRemoteServerStatus;
    isChecking: boolean;
    hasError: boolean;
    checkStatus: () => Promise<SchemaRemoteServerStatus | null>;
};

/**
 * Batch-polls Tier 1 status for a set of SSH-provisioned servers, retrying
 * unreachable/degraded servers until they recover. Never triggers Tier 2 — that
 * is only ever invoked from `useRemoteServerCheckMutation` via an explicit action.
 */
export const useRemoteServersStatus = (remoteServerIds: string[]): Map<string, RemoteServerStatusEntry> => {
    const queries = useQueries({
        queries: remoteServerIds.map((remoteServerId) =>
            $api.queryOptions(
                'get',
                '/api/remote-servers/{remote_server_id}/status',
                { params: { path: { remote_server_id: remoteServerId } } },
                {
                    refetchOnMount: 'always',
                    refetchInterval: (statusQuery) =>
                        statusQuery.state.data?.status !== 'healthy' ? REMOTE_UNAVAILABLE_POLL_MS : false,
                }
            )
        ),
    });

    return new Map(
        remoteServerIds.map((remoteServerId, index) => {
            const query = queries[index];
            return [
                remoteServerId,
                {
                    status: query.data,
                    isChecking: query.isFetching,
                    hasError: query.isError,
                    checkStatus: async () => {
                        const result = await query.refetch();
                        return result.isError ? null : (result.data ?? null);
                    },
                },
            ];
        })
    );
};
