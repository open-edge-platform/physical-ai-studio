import { $api } from '../../../api/client';
import { getSshHostKeyFingerprint } from '../../../api/errors';
import { SchemaSshHostAliasCreate, SchemaSshHostAliasOption } from '../../../api/openapi-spec';

/**
 * Appends a new Host entry to the user's ~/.ssh/config via
 * `POST /api/remote-servers/aliases`, the same host-key-confirmation retry
 * shape `useRemoteTrainerFormMutation` already uses for a tunneled trainer:
 * a genuinely new host has no ``known_hosts`` entry yet, so the first
 * attempt raises a 428 the caller re-submits once the fingerprint it
 * surfaced is confirmed.
 */
export const useAddSshHostMutation = () => {
    const mutation = $api.useMutation('post', '/api/remote-servers/aliases', {
        meta: { invalidates: [['get', '/api/remote-servers/aliases']] },
    });

    const save = (
        values: SchemaSshHostAliasCreate,
        {
            onSuccess,
            acceptedHostKeyFingerprint,
            onHostKeyConfirmationRequired,
        }: {
            onSuccess: (option: SchemaSshHostAliasOption) => void;
            acceptedHostKeyFingerprint?: string;
            onHostKeyConfirmationRequired: (fingerprint: string) => void;
        }
    ): void => {
        const headers = acceptedHostKeyFingerprint
            ? { 'accepted-host-key-fingerprint': acceptedHostKeyFingerprint }
            : undefined;
        mutation.mutate(
            { body: values, params: { header: headers } },
            {
                onSuccess,
                onError: (error) => {
                    const fingerprint = getSshHostKeyFingerprint(error);
                    if (fingerprint !== undefined) {
                        onHostKeyConfirmationRequired(fingerprint);
                    }
                },
            }
        );
    };

    return { save, reset: mutation.reset, isPending: mutation.isPending, error: mutation.error };
};
