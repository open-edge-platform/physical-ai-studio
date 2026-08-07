import { $api } from '../../../api/client';

/**
 * Runs the explicit Tier 2 preflight ("Test connection") for one SSH server.
 * A mutation, never a query — this pulls a multi-gigabyte trainer image and
 * launches a one-shot container, so it must only ever fire from a deliberate
 * user action, never on mount or a status poll.
 */
export const useRemoteServerCheckMutation = () =>
    $api.useMutation('post', '/api/remote-servers/{remote_server_id}/check');
