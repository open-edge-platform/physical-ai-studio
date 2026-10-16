import { useState } from 'react';

import { AlertDialog, Checkbox, Flex, Text } from '@geti-ui/ui';

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
    const [forgetLocally, setForgetLocally] = useState(false);
    const deleteMutation = $api.useMutation('delete', '/api/remote-trainers/{remote_trainer_id}', {
        meta: {
            invalidates: [['get', '/api/remote-trainers']],
        },
    });

    const error = deleteMutation.isError
        ? (getApiErrorMessage(deleteMutation.error) ?? 'The remote trainer could not be deleted. Try again.')
        : undefined;

    const remove = async () => {
        deleteMutation.mutate(
            {
                params: {
                    path: { remote_trainer_id: remoteTrainer.id },
                    query: { forget_locally: forgetLocally },
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
            primaryActionLabel={forgetLocally ? 'Force delete' : 'Delete'}
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
                {deleteMutation.isPending && (
                    <Text>
                        Deleting trainer…{' '}
                        {remoteTrainer.connection_mode === 'ssh' &&
                            !forgetLocally &&
                            'SSH cleanup may take a while if the host is unreachable.'}
                    </Text>
                )}
                {error && <Text UNSAFE_className={classes.errorMessage}>{error}</Text>}
                {remoteTrainer.connection_mode === 'ssh' && (
                    <Checkbox
                        isSelected={forgetLocally}
                        onChange={setForgetLocally}
                        isDisabled={deleteMutation.isPending}
                    >
                        Force delete locally without cleaning up the remote container and volume. Stop pending or
                        running jobs first.
                    </Checkbox>
                )}
            </Flex>
        </AlertDialog>
    );
};
