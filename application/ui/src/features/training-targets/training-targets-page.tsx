import { useState } from 'react';

import { DialogContainer, Heading, Text, View } from '@geti-ui/ui';
import { Deployments, Lock } from '@geti-ui/ui/icons';

import { $api } from '../../api/client';
import { ReactComponent as AwsIcon } from '../../assets/icons/aws-icon.svg';
import { CloudProviderForm } from './training-target-form/cloud-provider-form';
import { DeleteRemoteTrainerDialog } from './training-targets-table/delete-remote-trainer-dialog';
import { InstallPrerequisitesDialog } from './training-targets-table/install-prerequisites-dialog';
import { RemoteTrainerForm } from './training-targets-table/remote-trainer-form/remote-trainer-form';
import {
    SshHostKeyConfirmation,
    SshHostKeyConfirmationDialog,
} from './training-targets-table/ssh-host-key-confirmation-dialog';
import { INSTANCE_TYPES, InstanceType, TrainingTargetRow } from './training-targets-table/training-target-row';
import { TrainingTargetsTable } from './training-targets-table/training-targets-table';

import classes from './training-targets-page.module.css';

type TrainingTargetAction =
    | { type: 'create'; instanceType: InstanceType }
    | { type: 'edit'; row: TrainingTargetRow }
    | { type: 'delete'; row: TrainingTargetRow }
    | { type: 'setup'; row: TrainingTargetRow; reboot: boolean }
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
            <Heading level={2}>Add a training target</Heading>
            <div className={classes.creationGrid}>
                {INSTANCE_TYPES.map(({ id, label, description }) => (
                    <button
                        key={id}
                        type='button'
                        className={classes.creationCard}
                        aria-label={`Add training target: ${label}`}
                        aria-describedby={`training-target-${id}-description`}
                        disabled={id === 'ssh' && !sshAvailable}
                        onClick={() => setAction({ type: 'create', instanceType: id })}
                    >
                        <span className={classes.typeIcon} aria-hidden='true'>
                            {id === 'direct' ? <Deployments /> : id === 'ssh' ? <Lock /> : <AwsIcon />}
                        </span>
                        <span className={classes.typeCopy}>
                            <span className={classes.typeName}>{label}</span>
                            <span id={`training-target-${id}-description`} className={classes.typeDescription}>
                                {description}
                            </span>
                        </span>
                    </button>
                ))}
            </div>

            {sshFeature?.network_exposed && (
                <Text UNSAFE_className={classes.notice}>
                    SSH training targets are unavailable in this environment. Direct-URL trainers are unaffected.
                </Text>
            )}

            <Heading level={2}>Training targets</Heading>
            {rows.length === 0 ? (
                <View UNSAFE_className={classes.container}>
                    <Text UNSAFE_className={classes.emptyList}>No training targets added</Text>
                </View>
            ) : (
                <div className={classes.tableScroll}>
                    <TrainingTargetsTable
                        rows={rows}
                        onEdit={(row) => setAction({ type: 'edit', row })}
                        onDelete={(row) => setAction({ type: 'delete', row })}
                        onSetup={sshAvailable ? (row, reboot) => setAction({ type: 'setup', row, reboot }) : undefined}
                    />
                </div>
            )}

            <DialogContainer onDismiss={closeForm}>
                {action?.type === 'create' && action.instanceType === 'aws_batch' && (
                    <CloudProviderForm close={closeForm} />
                )}
                {action?.type === 'create' && action.instanceType !== 'aws_batch' && (
                    <RemoteTrainerForm
                        instanceType={action.instanceType}
                        close={closeForm}
                        requestHostKeyConfirmation={setHostKeyConfirmation}
                        sshAvailable={sshAvailable}
                    />
                )}
                {action?.type === 'edit' && action.row.trainer.connection_mode === 'aws_batch' && (
                    <CloudProviderForm remoteTrainer={action.row.trainer} close={closeForm} />
                )}
                {action?.type === 'edit' && action.row.trainer.connection_mode !== 'aws_batch' && (
                    <RemoteTrainerForm
                        instanceType={action.row.trainer.connection_mode}
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
                {action?.type === 'setup' && (
                    <InstallPrerequisitesDialog
                        trainer={action.row.trainer}
                        reboot={action.reboot}
                        onClose={closeForm}
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
