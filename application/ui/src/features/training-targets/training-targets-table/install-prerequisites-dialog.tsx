import { AlertDialog, Flex, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { SchemaRemoteTrainer } from '../../../api/openapi-spec';

import classes from './remote-trainer-form/remote-trainer-form.module.css';

type Props = {
    trainer: SchemaRemoteTrainer;
    reboot: boolean;
    onClose: () => void;
};

export const InstallPrerequisitesDialog = ({ trainer, reboot, onClose }: Props) => {
    const healthKey: [
        'get',
        '/api/remote-trainers/{remote_trainer_id}/health',
        { params: { path: { remote_trainer_id: string } } },
    ] = [
        'get',
        '/api/remote-trainers/{remote_trainer_id}/health',
        { params: { path: { remote_trainer_id: trainer.id } } },
    ];
    const install = $api.useMutation('post', '/api/remote-trainers/{remote_trainer_id}/install-prerequisites', {
        meta: { invalidates: [healthKey] },
    });
    const restart = $api.useMutation('post', '/api/remote-trainers/{remote_trainer_id}/reboot-after-install', {
        meta: { invalidates: [healthKey] },
    });
    const mutation = reboot ? restart : install;
    const error = mutation.isError
        ? (getApiErrorMessage(mutation.error) ?? 'The request failed. Try again.')
        : undefined;
    const message = reboot
        ? `Reboot ${trainer.name} to activate installed GPU prerequisites? ` +
          'All workloads on this host will be interrupted.'
        : `Install Docker and GPU prerequisites on ${trainer.name}? ` +
          'This changes packages on the SSH host and may require a reboot.';

    return (
        <AlertDialog
            title={reboot ? 'Reboot SSH host' : 'Install host prerequisites'}
            variant='warning'
            primaryActionLabel={reboot ? 'Reboot host' : 'Install'}
            cancelLabel='Cancel'
            onCancel={onClose}
            onPrimaryAction={() =>
                mutation.mutate({ params: { path: { remote_trainer_id: trainer.id } } }, { onSuccess: onClose })
            }
            isPrimaryActionDisabled={mutation.isPending}
        >
            <Flex direction='column' gap='size-150'>
                <Text>{message}</Text>
                {error && <Text UNSAFE_className={classes.errorMessage}>{error}</Text>}
            </Flex>
        </AlertDialog>
    );
};
