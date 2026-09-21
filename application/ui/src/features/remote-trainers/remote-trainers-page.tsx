import { useState } from 'react';

import { Button, DialogContainer, Flex, Heading, Text, View } from '@geti-ui/ui';
import { Add } from '@geti-ui/ui/icons';

import { $api } from '../../api/client';
import { SchemaRemoteTrainer } from '../../api/openapi-spec';
import { DeleteRemoteTrainerDialog } from './remote-trainers-table/delete-remote-trainer-dialog';
import { RemoteTrainerForm } from './remote-trainers-table/remote-trainer-form/remote-trainer-form';
import { RemoteTrainersTable } from './remote-trainers-table/remote-trainers-table';
import {
    SshHostKeyConfirmation,
    SshHostKeyConfirmationDialog,
} from './remote-trainers-table/ssh-host-key-confirmation-dialog';

import classes from './remote-trainers-page.module.css';

type RemoteTrainerAction =
    | { type: 'create' }
    | { type: 'edit'; remoteTrainer: SchemaRemoteTrainer }
    | { type: 'delete'; remoteTrainer: SchemaRemoteTrainer }
    | undefined;

export const RemoteTrainersPage = () => {
    const { data: remoteTrainers } = $api.useSuspenseQuery('get', '/api/remote-trainers');
    const [action, setAction] = useState<RemoteTrainerAction>();
    const [hostKeyConfirmation, setHostKeyConfirmation] = useState<SshHostKeyConfirmation>();

    const closeForm = () => {
        setAction(undefined);
        setHostKeyConfirmation(undefined);
    };

    const dismissHostKeyConfirmation = () => {
        hostKeyConfirmation?.onCancel();
        setHostKeyConfirmation(undefined);
    };

    return (
        <View padding='size-400' height='100%' maxWidth='240ch' marginX='auto'>
            <Flex marginBottom={'size-250'} justifyContent={'space-between'} alignItems={'center'}>
                <View>
                    <Heading level={1}>Remote Trainers</Heading>
                    <Text>Configure and monitor trainer endpoints for remote training jobs.</Text>
                </View>

                <Button
                    variant='accent'
                    UNSAFE_className={classes.addButton}
                    onPress={() => setAction({ type: 'create' })}
                >
                    <Add />
                    <Text>New remote trainer</Text>
                </Button>
            </Flex>

            {remoteTrainers.length === 0 ? (
                <View UNSAFE_className={classes.container}>
                    <Text UNSAFE_className={classes.emptyList}>No remote trainers are configured.</Text>
                </View>
            ) : (
                <RemoteTrainersTable
                    remoteTrainers={remoteTrainers}
                    onEdit={(remoteTrainer) => setAction({ type: 'edit', remoteTrainer })}
                    onDelete={(remoteTrainer) => setAction({ type: 'delete', remoteTrainer })}
                />
            )}

            <DialogContainer onDismiss={closeForm}>
                {(action?.type === 'create' || action?.type === 'edit') && (
                    <RemoteTrainerForm
                        remoteTrainer={action.type === 'edit' ? action.remoteTrainer : undefined}
                        close={closeForm}
                        requestHostKeyConfirmation={setHostKeyConfirmation}
                    />
                )}
                {action?.type === 'delete' && (
                    <DeleteRemoteTrainerDialog
                        remoteTrainer={action.remoteTrainer}
                        onCancel={() => setAction(undefined)}
                        onDeleted={() => setAction(undefined)}
                    />
                )}
            </DialogContainer>
            <DialogContainer onDismiss={dismissHostKeyConfirmation}>
                {hostKeyConfirmation !== undefined && (
                    <SshHostKeyConfirmationDialog
                        host={hostKeyConfirmation.host}
                        fingerprint={hostKeyConfirmation.fingerprint}
                        onCancel={dismissHostKeyConfirmation}
                        onConfirm={() => {
                            setHostKeyConfirmation(undefined);
                            hostKeyConfirmation.onConfirm();
                        }}
                    />
                )}
            </DialogContainer>
        </View>
    );
};
