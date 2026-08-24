import { useState } from 'react';

import { ActionButton, Badge, Flex, Item, Key, Menu, MenuTrigger, StatusLight, Text } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { SchemaRemoteTrainer, SchemaRemoteTrainerHealth } from '../../../api/openapi-spec';
import { Table, TableColumn } from '../../../components/table/table';
import { deviceTypes, getDisplayHealth, healthLabel, healthVariant } from '../remote-trainer-health-utils';
import { RemoteTrainerDetail } from './remote-trainer-detail/remote-trainer-detail';
import { useRemoteTrainersHealth } from './use-remote-trainers-health';

import classes from './remote-trainers-table.module.css';

const REMOTE_TRAINER_COLUMNS: TableColumn[] = [
    { width: 'max-content' },
    { width: '1fr', header: 'Name' },
    { width: '1fr', header: 'Trainer URL' },
    { width: '1fr', header: 'Status' },
    { width: '1fr', header: 'Compute' },
    { width: 'auto', align: 'end' },
];

const DEVICE_BADGE_CLASSES: Record<string, string> = {
    CUDA: classes.cudaBadge,
    XPU: classes.xpuBadge,
};

const REMOTE_TRAINERS_MENU_ACTION_ITEMS = {
    CHECK_STATUS: 'check_status',
    EDIT: 'Edit',
    DELETE: 'Delete',
};

type RemoteTrainersMenuActionsProps = {
    remoteTrainerName: string;
    onCheck: () => void;
    onEdit: () => void;
    onDelete: () => void;
    isChecking: boolean;
};

const RemoteTrainersMenuActions = ({
    remoteTrainerName,
    onCheck,
    onEdit,
    onDelete,
    isChecking,
}: RemoteTrainersMenuActionsProps) => {
    const handleAction = (action: Key) => {
        if (action === REMOTE_TRAINERS_MENU_ACTION_ITEMS.CHECK_STATUS) {
            onCheck();
        } else if (action === REMOTE_TRAINERS_MENU_ACTION_ITEMS.EDIT) {
            onEdit();
        } else if (action === REMOTE_TRAINERS_MENU_ACTION_ITEMS.DELETE) {
            onDelete();
        }
    };

    return (
        <MenuTrigger>
            <ActionButton aria-label={`More actions ${remoteTrainerName}`} isQuiet>
                <MoreMenu />
            </ActionButton>
            <Menu
                onAction={handleAction}
                disabledKeys={isChecking ? [REMOTE_TRAINERS_MENU_ACTION_ITEMS.CHECK_STATUS] : undefined}
            >
                <Item key={REMOTE_TRAINERS_MENU_ACTION_ITEMS.EDIT}>Edit</Item>
                <Item key={REMOTE_TRAINERS_MENU_ACTION_ITEMS.DELETE}>Delete</Item>
                <Item key={REMOTE_TRAINERS_MENU_ACTION_ITEMS.CHECK_STATUS}>Check status</Item>
            </Menu>
        </MenuTrigger>
    );
};

type RemoteTrainerRowProps = {
    remoteTrainer: SchemaRemoteTrainer;
    health?: SchemaRemoteTrainerHealth;
    isChecking: boolean;
    isExpanded: boolean;
    onToggleExpanded: () => void;
    onCheck: () => void;
    onEdit: () => void;
    onDelete: () => void;
};

const RemoteTrainerRow = ({
    remoteTrainer,
    health,
    isChecking,
    isExpanded,
    onToggleExpanded,
    onCheck,
    onEdit,
    onDelete,
}: RemoteTrainerRowProps) => {
    const types = deviceTypes(health);

    return (
        <Table.ExpandableRow
            id={`remote-trainer-row-${remoteTrainer.id}`}
            label={remoteTrainer.name}
            isExpanded={isExpanded}
            onExpandedChange={onToggleExpanded}
            detail={<RemoteTrainerDetail remoteTrainer={remoteTrainer} health={health} isChecking={isChecking} />}
        >
            <Text>{remoteTrainer.name}</Text>

            <Text UNSAFE_className={classes.trainerUrl}>{remoteTrainer.url}</Text>

            <StatusLight variant={healthVariant(health, isChecking)} UNSAFE_className={classes.healthStatus}>
                {healthLabel(health, isChecking)}
            </StatusLight>

            <Flex gap='size-100' alignItems='center' wrap>
                {types.map((type) => (
                    <Badge key={type} variant='neutral' UNSAFE_className={DEVICE_BADGE_CLASSES[type]}>
                        {type}
                    </Badge>
                ))}
                <Text UNSAFE_className={classes.cardMetaText}>
                    {health?.devices?.at(0)?.name ?? (isChecking ? 'Checking capability…' : 'Not reported')}
                </Text>
            </Flex>

            <Flex gap='size-100' wrap>
                <RemoteTrainersMenuActions
                    remoteTrainerName={remoteTrainer.name}
                    onCheck={onCheck}
                    onEdit={onEdit}
                    onDelete={onDelete}
                    isChecking={isChecking}
                />
            </Flex>
        </Table.ExpandableRow>
    );
};

type RemoteTrainersTableProps = {
    remoteTrainers: SchemaRemoteTrainer[];
    onEdit: (remoteTrainer: SchemaRemoteTrainer) => void;
    onDelete: (remoteTrainer: SchemaRemoteTrainer) => void;
};

export const RemoteTrainersTable = ({ remoteTrainers, onEdit, onDelete }: RemoteTrainersTableProps) => {
    const [expandedRemoteTrainerId, setExpandedRemoteTrainerId] = useState<string | undefined>(remoteTrainers[0]?.id);

    const health = useRemoteTrainersHealth(remoteTrainers.map((remoteTrainer) => remoteTrainer.id));

    return (
        <Table columns={REMOTE_TRAINER_COLUMNS} isEmphasized>
            {remoteTrainers.map((remoteTrainer) => {
                const entry = health.get(remoteTrainer.id);
                const displayHealth = getDisplayHealth(remoteTrainer.id, entry?.health, entry?.hasError ?? false);

                return (
                    <RemoteTrainerRow
                        key={remoteTrainer.id}
                        remoteTrainer={remoteTrainer}
                        health={displayHealth}
                        isChecking={entry?.isChecking ?? false}
                        isExpanded={expandedRemoteTrainerId === remoteTrainer.id}
                        onToggleExpanded={() =>
                            setExpandedRemoteTrainerId((current) =>
                                current === remoteTrainer.id ? undefined : remoteTrainer.id
                            )
                        }
                        onCheck={() => {
                            void entry?.checkHealth();
                            setExpandedRemoteTrainerId(remoteTrainer.id);
                        }}
                        onEdit={() => onEdit(remoteTrainer)}
                        onDelete={() => onDelete(remoteTrainer)}
                    />
                );
            })}
        </Table>
    );
};
