import { AlertDialog, Flex, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { SchemaRemoteTrainer } from '../../../api/openapi-spec';

import classes from './remote-trainer-form/remote-trainer-form.module.css';

type DeleteRemoteTrainerDialogProps = {
    remoteTrainer: SchemaRemoteTrainer;
    onDeleted: () => void;
    onCancel: () => void;
};

export const DeleteRemoteTrainerDialog = ({ remoteTrainer, onDeleted, onCancel }: DeleteRemoteTrainerDialogProps) => {
    const deleteMutation = $api.useMutation('delete', '/api/remote-trainers/{remote_trainer_id}', {
        meta: {
            invalidates: [['get', '/api/remote-trainers']],
        },
    });

    const error = deleteMutation.isError
        ? (getApiErrorMessage(deleteMutation.error) ?? "'The remote trainer could not be deleted. Try again.'")
        : undefined;

    const remove = async () => {
        deleteMutation.mutate(
            {
                params: {
                    path: { remote_trainer_id: remoteTrainer.id },
                },
            },
            {
                onSuccess: onDeleted,
            }
        );
    };

    return (
        <AlertDialog
            title='Delete remote trainer'
            variant='warning'
            primaryActionLabel='Delete'
            cancelLabel='Close'
            onCancel={onCancel}
            onPrimaryAction={remove}
            isPrimaryActionDisabled={deleteMutation.isPending}
        >
            <Flex direction='column' gap='size-150'>
                <Text>
                    Delete {remoteTrainer.name}? Submitted remote jobs retain their pinned endpoint URL and are not
                    changed.
                </Text>
                {error && <Text UNSAFE_className={classes.errorMessage}>{error}</Text>}
            </Flex>
        </AlertDialog>
    );
};
