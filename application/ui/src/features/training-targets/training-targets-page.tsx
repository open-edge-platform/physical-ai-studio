import { useState } from 'react';

import { Button, DialogContainer, Flex, Heading, Item, Key, Menu, MenuTrigger, Text, View } from '@geti-ui/ui';
import { Add } from '@geti-ui/ui/icons';

import { $api } from '../../api/client';
import { DeleteRemoteServerDialog } from './training-targets-table/delete-remote-server-dialog';
import { DeleteRemoteTrainerDialog } from './training-targets-table/delete-remote-trainer-dialog';
import { RemoteServerForm } from './training-targets-table/remote-server-form/remote-server-form';
import { RemoteTrainerForm } from './training-targets-table/remote-trainer-form/remote-trainer-form';
import { TrainingTargetRow } from './training-targets-table/training-target-row';
import { TrainingTargetsTable } from './training-targets-table/training-targets-table';

import classes from './training-targets-page.module.css';

type TrainingTargetAction =
    | { type: 'create-direct-url' }
    | { type: 'create-ssh' }
    | { type: 'edit'; row: TrainingTargetRow }
    | { type: 'delete'; row: TrainingTargetRow }
    | undefined;

const NEW_TARGET_MENU_ITEMS = {
    DIRECT_URL: 'direct-url',
    SSH: 'ssh',
};

export const TrainingTargetsPage = () => {
    const { data: remoteTrainers } = $api.useSuspenseQuery('get', '/api/remote-trainers');
    const { data: remoteServers } = $api.useSuspenseQuery('get', '/api/remote-servers');
    const [action, setAction] = useState<TrainingTargetAction>();

    const rows: TrainingTargetRow[] = [
        ...remoteTrainers.map((trainer): TrainingTargetRow => ({ kind: 'direct-url', trainer })),
        ...remoteServers.map((server): TrainingTargetRow => ({ kind: 'ssh', server })),
    ];

    const handleNewTargetAction = (key: Key) => {
        if (key === NEW_TARGET_MENU_ITEMS.DIRECT_URL) {
            setAction({ type: 'create-direct-url' });
        } else if (key === NEW_TARGET_MENU_ITEMS.SSH) {
            setAction({ type: 'create-ssh' });
        }
    };

    return (
        <View padding='size-400' height='100%' maxWidth='240ch' marginX='auto'>
            <Flex marginBottom={'size-250'} justifyContent={'space-between'} alignItems={'center'}>
                <View>
                    <Heading level={1}>Training Targets</Heading>
                    <Text>Configure and monitor where training jobs run.</Text>
                </View>

                <MenuTrigger>
                    <Button variant='accent' UNSAFE_className={classes.addButton}>
                        <Add />
                        <Text>New training target</Text>
                    </Button>
                    <Menu onAction={handleNewTargetAction}>
                        <Item key={NEW_TARGET_MENU_ITEMS.DIRECT_URL}>Direct URL trainer</Item>
                        <Item key={NEW_TARGET_MENU_ITEMS.SSH}>SSH server</Item>
                    </Menu>
                </MenuTrigger>
            </Flex>

            <View UNSAFE_className={classes.container}>
                {rows.length === 0 ? (
                    <Text UNSAFE_className={classes.emptyList}>No training targets are configured.</Text>
                ) : (
                    <TrainingTargetsTable
                        rows={rows}
                        onEdit={(row) => setAction({ type: 'edit', row })}
                        onDelete={(row) => setAction({ type: 'delete', row })}
                    />
                )}
            </View>

            <DialogContainer onDismiss={() => setAction(undefined)}>
                {action?.type === 'create-direct-url' && <RemoteTrainerForm close={() => setAction(undefined)} />}
                {action?.type === 'create-ssh' && <RemoteServerForm close={() => setAction(undefined)} />}
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
