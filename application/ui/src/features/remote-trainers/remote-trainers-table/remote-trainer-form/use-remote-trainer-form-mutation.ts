import { $api } from '../../../../api/client';
import { getSshHostKeyFingerprint } from '../../../../api/errors';
import { SchemaRemoteTrainer, SchemaRemoteTrainerCreate } from '../../../../api/openapi-spec';

export type RemoteTrainerFormValues = SchemaRemoteTrainerCreate;

export const useRemoteTrainerFormMutation = (remoteTrainer: SchemaRemoteTrainer | undefined) => {
    const createRemoteTrainer = $api.useMutation('post', '/api/remote-trainers', {
        meta: { invalidates: [['get', '/api/remote-trainers']] },
    });
    const updateRemoteTrainer = $api.useMutation('patch', '/api/remote-trainers/{remote_trainer_id}', {
        meta: { invalidates: [['get', '/api/remote-trainers']] },
    });

    const save = (
        values: RemoteTrainerFormValues,
        {
            onSuccess,
            acceptedHostKeyFingerprint,
            onHostKeyConfirmationRequired,
        }: {
            onSuccess: () => void;
            acceptedHostKeyFingerprint?: string;
            onHostKeyConfirmationRequired: (fingerprint: string) => void;
        }
    ): void => {
        const headers = acceptedHostKeyFingerprint
            ? { 'accepted-host-key-fingerprint': acceptedHostKeyFingerprint }
            : undefined;
        const onError = (error: unknown) => {
            const fingerprint = getSshHostKeyFingerprint(error);
            if (fingerprint !== undefined) {
                onHostKeyConfirmationRequired(fingerprint);
            }
        };
        if (remoteTrainer === undefined) {
            createRemoteTrainer.mutate({ body: values, params: { header: headers } }, { onSuccess, onError });

            return;
        }

        updateRemoteTrainer.mutate(
            {
                params: { path: { remote_trainer_id: remoteTrainer.id }, header: headers },
                body: values,
            },
            { onSuccess, onError }
        );
    };

    const activeMutation = remoteTrainer === undefined ? createRemoteTrainer : updateRemoteTrainer;
    return {
        save,
        reset: activeMutation.reset,
        isPending: activeMutation.isPending,
        error: activeMutation.error,
    };
};
