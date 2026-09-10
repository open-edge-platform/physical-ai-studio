import { $api } from '../../../../api/client';
import { SchemaRemoteTrainer, SchemaRemoteTrainerCreate, SchemaSshHostAliasCreate } from '../../../../api/openapi-spec';

export type RemoteTrainerFormValues = SchemaRemoteTrainerCreate;

export const useRemoteTrainerFormMutation = (remoteTrainer: SchemaRemoteTrainer | undefined) => {
    const createSshHostAlias = $api.useMutation('post', '/api/remote-servers/aliases', {
        meta: { invalidates: [['get', '/api/remote-servers/aliases']] },
    });
    const createRemoteTrainer = $api.useMutation('post', '/api/remote-trainers', {
        meta: { invalidates: [['get', '/api/remote-trainers']] },
    });
    const updateRemoteTrainer = $api.useMutation('patch', '/api/remote-trainers/{remote_trainer_id}', {
        meta: { invalidates: [['get', '/api/remote-trainers']] },
    });

    const save = async (
        values: RemoteTrainerFormValues,
        newSshHost: SchemaSshHostAliasCreate | undefined,
        { onSuccess }: { onSuccess: () => void }
    ): Promise<void> => {
        // A new host is created before the trainer that references its alias, so a
        // failed alias creation (e.g. it already exists) never leaves the trainer
        // half-saved with an alias that was never actually added.
        const resolvedValues =
            newSshHost === undefined
                ? values
                : { ...values, ssh_host_alias: (await createSshHostAlias.mutateAsync({ body: newSshHost })).alias };

        if (remoteTrainer === undefined) {
            createRemoteTrainer.mutate({ body: resolvedValues }, { onSuccess });

            return;
        }

        updateRemoteTrainer.mutate(
            {
                params: { path: { remote_trainer_id: remoteTrainer.id } },
                body: resolvedValues,
            },
            { onSuccess }
        );
    };

    const activeMutation = remoteTrainer === undefined ? createRemoteTrainer : updateRemoteTrainer;
    const error: unknown = createSshHostAlias.error ?? activeMutation.error;

    return {
        save,
        isPending: createSshHostAlias.isPending || activeMutation.isPending,
        error,
    };
};
