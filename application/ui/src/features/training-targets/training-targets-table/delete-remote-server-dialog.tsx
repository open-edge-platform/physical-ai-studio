import { AlertDialog, Flex, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { SchemaRemoteServer } from '../../../api/openapi-spec';

import classes from './remote-trainer-form/remote-trainer-form.module.css';

type DeleteRemoteServerDialogProps = {
    remoteServer: SchemaRemoteServer;
    onDeleted: () => void;
    onCancel: () => void;
};

export const DeleteRemoteServerDialog = ({ remoteServer, onDeleted, onCancel }: DeleteRemoteServerDialogProps) => {
    const deleteMutation = $api.useMutation('delete', '/api/remote-servers/{remote_server_id}', {
        meta: {
            invalidates: [['get', '/api/remote-servers']],
        },
    });

    const error = deleteMutation.isError
        ? (getApiErrorMessage(deleteMutation.error) ?? 'The training target could not be deleted. Try again.')
        : undefined;

    const remove = async () => {
        deleteMutation.mutate(
            {
                params: {
                    path: { remote_server_id: remoteServer.id },
                },
            },
            {
                onSuccess: onDeleted,
            }
        );
    };

    return (
        <AlertDialog
            title='Delete training target'
            variant='warning'
            primaryActionLabel='Delete'
            cancelLabel='Close'
            onCancel={onCancel}
            onPrimaryAction={remove}
            isPrimaryActionDisabled={deleteMutation.isPending}
        >
            <Flex direction='column' gap='size-150'>
                <Text>Delete {remoteServer.name}? This does not remove the Host entry from your SSH config.</Text>
                {error && <Text UNSAFE_className={classes.errorMessage}>{error}</Text>}
            </Flex>
        </AlertDialog>
    );
};
