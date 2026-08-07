import { $api } from '../../../../api/client';
import { SchemaRemoteServer, SchemaRemoteServerCreate } from '../../../../api/openapi-spec';

export type RemoteServerFormValues = SchemaRemoteServerCreate;

export const useRemoteServerFormMutation = (remoteServer: SchemaRemoteServer | undefined) => {
    const createRemoteServer = $api.useMutation('post', '/api/remote-servers', {
        meta: { invalidates: [['get', '/api/remote-servers']] },
    });
    const updateRemoteServer = $api.useMutation('patch', '/api/remote-servers/{remote_server_id}', {
        meta: { invalidates: [['get', '/api/remote-servers']] },
    });

    const save = (values: RemoteServerFormValues, { onSuccess }: { onSuccess: () => void }): void => {
        if (remoteServer === undefined) {
            createRemoteServer.mutate({ body: values }, { onSuccess });

            return;
        }

        updateRemoteServer.mutate(
            {
                params: { path: { remote_server_id: remoteServer.id } },
                body: values,
            },
            { onSuccess }
        );
    };

    const activeMutation = remoteServer === undefined ? createRemoteServer : updateRemoteServer;
    const error: unknown = activeMutation.error;

    return {
        save,
        isPending: activeMutation.isPending,
        error,
    };
};
