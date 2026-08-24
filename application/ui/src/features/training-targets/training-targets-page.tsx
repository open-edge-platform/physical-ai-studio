import { useState } from 'react';

import { Button, DialogContainer, Flex, Icon, Text, View } from '@geti-ui/ui';
import { Add } from '@geti-ui/ui/icons';

import { $api } from '../../api/client';
import { isSshFeatureUnavailableError } from '../../api/errors';
import { TrainingTargetForm } from './training-target-form/training-target-form';
import { DeleteRemoteServerDialog } from './training-targets-table/delete-remote-server-dialog';
import { DeleteRemoteTrainerDialog } from './training-targets-table/delete-remote-trainer-dialog';
import { RemoteServerForm } from './training-targets-table/remote-server-form/remote-server-form';
import { RemoteTrainerForm } from './training-targets-table/remote-trainer-form/remote-trainer-form';
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
    // SSH-provisioned servers are gated behind a backend feature switch that
    // fails closed (503 `ssh_feature_unavailable`) whenever this Studio
    // instance is not eligible to run the feature (e.g. it is bound to more
    // than loopback). That is an expected, often-permanent environment state,
    // not a page-breaking error, so this is a plain query the page degrades
    // gracefully around rather than a suspense query that would crash to the
    // nearest error boundary.
    const { data: remoteServers, error: remoteServersError } = $api.useQuery('get', '/api/remote-servers');
    const sshUnavailable = isSshFeatureUnavailableError(remoteServersError);
    const [action, setAction] = useState<TrainingTargetAction>();

    const rows: TrainingTargetRow[] = [
        ...remoteTrainers.map((trainer): TrainingTargetRow => ({ kind: 'direct-url', trainer })),
        ...(remoteServers ?? []).map((server): TrainingTargetRow => ({ kind: 'ssh', server })),
    ];

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

            {sshUnavailable && (
                <Text UNSAFE_className={classes.notice}>
                    SSH-provisioned training targets are not available in this environment. Direct-URL trainers below
                    are unaffected.
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

            <DialogContainer onDismiss={() => setAction(undefined)}>
                {action?.type === 'create' && <TrainingTargetForm close={() => setAction(undefined)} />}
                {action?.type === 'edit' && action.row.kind === 'direct-url' && (
                    <RemoteTrainerForm remoteTrainer={action.row.trainer} close={() => setAction(undefined)} />
                )}
                {action?.type === 'edit' && action.row.kind === 'ssh' && (
                    <RemoteServerForm remoteServer={action.row.server} close={() => setAction(undefined)} />
                )}
                {action?.type === 'delete' && action.row.kind === 'direct-url' && (
                    <DeleteRemoteTrainerDialog
                        remoteTrainer={action.row.trainer}
                        onCancel={() => setAction(undefined)}
                        onDeleted={() => setAction(undefined)}
                    />
                )}
                {action?.type === 'delete' && action.row.kind === 'ssh' && (
                    <DeleteRemoteServerDialog
                        remoteServer={action.row.server}
                        onCancel={() => setAction(undefined)}
                        onDeleted={() => setAction(undefined)}
                    />
                )}
            </DialogContainer>
        </View>
    );
};
