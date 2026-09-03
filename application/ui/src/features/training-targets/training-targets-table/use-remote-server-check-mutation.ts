import { $api } from '../../../api/client';

/**
 * Runs the explicit Tier 2 preflight ("Pull & verify image") for one SSH server.
 * A mutation, never a query — this pulls a multi-gigabyte trainer image and
 * launches a one-shot container, so it must only ever fire from a deliberate
 * user action, never on mount or a status poll.
 *
 * The backend persists this result's rolled-up outcome onto the server's
 * `last_check_status` (see `record_check_result`), which is what gates job
 * submission and the "Train model" dialog's readiness banner. Invalidating
 * the server list here is required so both pick up the new status without
 * a manual page reload.
 */
export const useRemoteServerCheckMutation = () =>
    $api.useMutation('post', '/api/remote-servers/{remote_server_id}/check', {
        meta: { invalidates: [['get', '/api/remote-servers']] },
    });
