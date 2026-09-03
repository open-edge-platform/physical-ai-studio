import { useIsMutating } from '@tanstack/react-query';

/**
 * Whether a Tier 2 "Pull & verify image" check is currently in flight for a
 * given server, *regardless of which component instance fired it*.
 *
 * `useRemoteServerCheckMutation` is called from multiple places - the
 * training-targets table row, its expanded detail panel, and the
 * `VerifyAfterSaveDialog` - each getting its own `useMutation` instance with
 * its own local `isPending`. That dialog in particular fires the mutation and
 * then immediately unmounts, so nothing in the table would otherwise reflect
 * that a pull/verify is still running in the background: the row would look
 * idle even though the mutation is actively in flight, until it eventually
 * resolves and invalidates the server list.
 *
 * `useIsMutating` reads directly from the shared `MutationCache`, so it sees
 * every in-flight mutation for this endpoint no matter which component (or
 * how many re-renders) started it, and keeps working after the originating
 * component has unmounted.
 */
export const useIsRemoteServerCheckRunning = (remoteServerId: string): boolean => {
    const runningCount = useIsMutating({
        mutationKey: ['post', '/api/remote-servers/{remote_server_id}/check'],
        predicate: (mutation) => {
            const variables = mutation.state.variables as
                { params?: { path?: { remote_server_id?: string } } } | undefined;

            return variables?.params?.path?.remote_server_id === remoteServerId;
        },
    });

    return runningCount > 0;
};
