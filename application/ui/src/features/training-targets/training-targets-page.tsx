import { useState } from 'react';

import { Button, DialogContainer, Flex, Icon, Text, View } from '@geti-ui/ui';
import { Add } from '@geti-ui/ui/icons';

import { $api } from '../../api/client';
import { TrainingTargetForm } from './training-target-form/training-target-form';
import { DeleteRemoteTrainerDialog } from './training-targets-table/delete-remote-trainer-dialog';
import { RemoteTrainerForm } from './training-targets-table/remote-trainer-form/remote-trainer-form';
import {
    SshHostKeyConfirmation,
    SshHostKeyConfirmationDialog,
} from './training-targets-table/ssh-host-key-confirmation-dialog';
import { TrainingTargetRow } from './training-targets-table/training-target-row';
import { TrainingTargetsTable } from './training-targets-table/training-targets-table';

import classes from './training-targets-page.module.css';

type TrainingTargetAction =
    | { type: 'create' }
    | { type: 'edit'; row: TrainingTargetRow }
    | { type: 'delete'; row: TrainingTargetRow }
    | undefined;

export const TrainingTargetsPage = () => {
    const { data: remoteTrainers } = $api.useSuspenseQuery('get', '/api/remote-trainers');
    const { data: sshFeature } = $api.useQuery('get', '/api/remote-servers/feature-status', {}, { retry: false });
    const sshAvailable = sshFeature?.network_exposed === false;
    const [action, setAction] = useState<TrainingTargetAction>();
    const [hostKeyConfirmation, setHostKeyConfirmation] = useState<SshHostKeyConfirmation>();

    const closeForm = () => {
        setAction(undefined);
        setHostKeyConfirmation(undefined);
    };

    const dismissHostKeyConfirmation = () => {
        hostKeyConfirmation?.onCancel();
        setHostKeyConfirmation(undefined);
    };

    const rows: TrainingTargetRow[] = remoteTrainers.map((trainer): TrainingTargetRow => ({
        kind: 'direct-url',
        trainer,
    }));

    return (
        <View padding='size-400' height='100%' maxWidth='240ch' marginX='auto'>
            <Flex marginBottom={'size-250'} justifyContent={'space-between'} alignItems={'center'}>
                <Text>Configure and monitor where training jobs run.</Text>

                <Button
                    variant='secondary'
                    UNSAFE_className={classes.addButton}
                    onPress={() => setAction({ type: 'create' })}
                >
                    <Icon marginEnd='size-50'>
                        <Add />
                    </Icon>
                    New training target
                </Button>
            </Flex>

            {sshFeature?.network_exposed && (
                <Text UNSAFE_className={classes.notice}>
                    SSH training targets are unavailable in this environment. Direct-URL trainers are unaffected.
                </Text>
            )}

            {rows.length === 0 ? (
                <View UNSAFE_className={classes.container}>
                    <Text UNSAFE_className={classes.emptyList}>No training targets are configured.</Text>
                </View>
            ) : (
                <TrainingTargetsTable
                    rows={rows}
                    onEdit={(row) => setAction({ type: 'edit', row })}
                    onDelete={(row) => setAction({ type: 'delete', row })}
                />
            )}

            <DialogContainer onDismiss={closeForm}>
                {action?.type === 'create' && (
                    <TrainingTargetForm
                        close={closeForm}
                        requestHostKeyConfirmation={setHostKeyConfirmation}
                        sshAvailable={sshAvailable}
                    />
                )}
                {action?.type === 'edit' && action.row.kind === 'direct-url' && (
                    <RemoteTrainerForm
                        remoteTrainer={action.row.trainer}
                        close={closeForm}
                        requestHostKeyConfirmation={setHostKeyConfirmation}
                        sshAvailable={sshAvailable}
                    />
                )}
                {action?.type === 'delete' && action.row.kind === 'direct-url' && (
                    <DeleteRemoteTrainerDialog
                        remoteTrainer={action.row.trainer}
                        onCancel={closeForm}
                        onDeleted={closeForm}
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
