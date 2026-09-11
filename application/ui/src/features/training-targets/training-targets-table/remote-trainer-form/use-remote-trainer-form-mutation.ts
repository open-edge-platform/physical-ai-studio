import { $api } from '../../../../api/client';
import { SchemaRemoteTrainer, SchemaRemoteTrainerCreate } from '../../../../api/openapi-spec';

export type RemoteTrainerFormValues = SchemaRemoteTrainerCreate;

export const useRemoteTrainerFormMutation = (remoteTrainer: SchemaRemoteTrainer | undefined) => {
    const createRemoteTrainer = $api.useMutation('post', '/api/remote-trainers', {
        meta: { invalidates: [['get', '/api/remote-trainers']] },
    });
    const updateRemoteTrainer = $api.useMutation('patch', '/api/remote-trainers/{remote_trainer_id}', {
        meta: { invalidates: [['get', '/api/remote-trainers']] },
    });

    const save = (values: RemoteTrainerFormValues, { onSuccess }: { onSuccess: () => void }): void => {
        if (remoteTrainer === undefined) {
            createRemoteTrainer.mutate({ body: values }, { onSuccess });

            return;
        }

        updateRemoteTrainer.mutate(
            {
                params: { path: { remote_trainer_id: remoteTrainer.id } },
                body: values,
            },
            { onSuccess }
        );
    };

    const activeMutation = remoteTrainer === undefined ? createRemoteTrainer : updateRemoteTrainer;
    const error: unknown = activeMutation.error;

    return {
        save,
        isPending: activeMutation.isPending,
        error,
    };
};
