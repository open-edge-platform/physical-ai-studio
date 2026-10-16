import { useState } from 'react';

import { ActionButton, Flex, Item, Key, Menu, MenuTrigger, Text } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { SchemaRemoteTrainer } from '../../../api/openapi-spec';
import { Table, TableColumn } from '../../../components/table/table';
import { getDisplayHealth } from '../remote-trainer-health-utils';
import { RemoteTrainerDetail } from './remote-trainer-detail/remote-trainer-detail';
import { instanceTypeLabel, TrainingTargetRow, trainingTargetRowId } from './training-target-row';
import { useRemoteTrainersHealth } from './use-remote-trainers-health';

import classes from './training-targets-table.module.css';

export const TRAINING_TARGET_COLUMNS: TableColumn[] = [
    { width: 'max-content' },
    { width: 'minmax(120px, 1fr)', header: 'Name' },
    { width: 'minmax(140px, 1fr)', header: 'Type' },
    { width: 'auto', header: 'Actions', align: 'end' },
];

const TARGET_MENU_ACTION_ITEMS = {
    CHECK_STATUS: 'check_status',
    REBOOT: 'reboot_after_install',
    INSTALL: 'install_prerequisites',
};

type TargetMenuActionsProps = {
    targetName: string;
    onCheck?: () => void;
    onReboot?: () => void;
    onInstall?: () => void;
    isChecking: boolean;
    isStarting: boolean;
};

const TargetMenuActions = ({
    targetName,
    onCheck,
    onReboot,
    onInstall,
    isChecking,
    isStarting,
}: TargetMenuActionsProps) => {
    const items = [
        { key: TARGET_MENU_ACTION_ITEMS.CHECK_STATUS, label: 'Check status' },
        ...(onReboot ? [{ key: TARGET_MENU_ACTION_ITEMS.REBOOT, label: 'Reboot to finish setup' }] : []),
        ...(onInstall ? [{ key: TARGET_MENU_ACTION_ITEMS.INSTALL, label: 'Install prerequisites' }] : []),
    ];
    const handleAction = (action: Key) => {
        if (action === TARGET_MENU_ACTION_ITEMS.CHECK_STATUS) {
            onCheck?.();
        } else if (action === TARGET_MENU_ACTION_ITEMS.REBOOT) {
            onReboot?.();
        } else if (action === TARGET_MENU_ACTION_ITEMS.INSTALL) {
            onInstall?.();
        }
    };

    return (
        <MenuTrigger>
            <ActionButton aria-label={`More actions ${targetName}`} isQuiet>
                <MoreMenu />
            </ActionButton>
            <Menu
                items={items}
                onAction={handleAction}
                disabledKeys={[
                    ...(isChecking || onCheck === undefined ? [TARGET_MENU_ACTION_ITEMS.CHECK_STATUS] : []),
                    ...(isStarting ? [TARGET_MENU_ACTION_ITEMS.INSTALL] : []),
                ]}
            >
                {(item) => <Item key={item.key}>{item.label}</Item>}
            </Menu>
        </MenuTrigger>
    );
};

type TargetRowContentProps = {
    name: string;
    connectionModeText: string;
    isStarting: boolean;
    isChecking: boolean;
    onCheck?: () => void;
    onEdit: () => void;
    onDelete: () => void;
    onReboot?: () => void;
    onInstall?: () => void;
};

const targetRowCells = ({
    name,
    connectionModeText,
    isStarting,
    isChecking,
    onCheck,
    onEdit,
    onDelete,
    onReboot,
    onInstall,
}: TargetRowContentProps) => [
    <Text key='name' UNSAFE_className={classes.cellText}>
        {name}
    </Text>,
    <Text key='type' UNSAFE_className={classes.cellText}>
        {connectionModeText}
    </Text>,

    <div key='actions' onClick={(event) => event.stopPropagation()}>
        <Flex gap='size-100' alignItems='center'>
            <ActionButton aria-label={`Edit ${name}`} onPress={onEdit}>
                Edit
            </ActionButton>
            <ActionButton aria-label={`Remove ${name}`} onPress={onDelete}>
                Remove
            </ActionButton>
            <TargetMenuActions
                targetName={name}
                onCheck={onCheck}
                onReboot={onReboot}
                onInstall={onInstall}
                isChecking={isChecking}
                isStarting={isStarting}
            />
        </Flex>
    </div>,
];

type DirectUrlTargetRowProps = {
    trainer: SchemaRemoteTrainer;
    isExpanded: boolean;
    onExpandedChange: (isExpanded: boolean) => void;
    onExpand: () => void;
    onEdit: () => void;
    onDelete: () => void;
    onSetup?: (reboot: boolean) => void;
};

const DirectUrlTargetRow = ({
    trainer,
    isExpanded,
    onExpandedChange,
    onExpand,
    onEdit,
    onDelete,
    onSetup,
}: DirectUrlTargetRowProps) => {
    const health = useRemoteTrainersHealth([trainer.id]).get(trainer.id);
    const displayHealth = getDisplayHealth(trainer.id, health?.health, health?.hasError ?? false);
    const isChecking = health?.isChecking ?? false;
    const awaitingReboot = [
        'reboot_required',
        'reboot_blocked_active_containers',
        'nvidia_driver_unavailable',
    ].includes(displayHealth?.reason_code ?? '');

    return (
        <Table.ExpandableRow
            id={`training-target-row-${trainer.id}`}
            label={trainer.name}
            isExpanded={isExpanded}
            onExpandedChange={onExpandedChange}
            detail={<RemoteTrainerDetail remoteTrainer={trainer} health={displayHealth} isChecking={isChecking} />}
        >
            {targetRowCells({
                name: trainer.name,
                connectionModeText: instanceTypeLabel(trainer.connection_mode),
                isStarting: displayHealth?.status === 'starting',
                isChecking,
                onCheck: () => {
                    void health?.checkHealth();
                    onExpand();
                },
                onEdit,
                onDelete,
                onReboot:
                    trainer.connection_mode === 'ssh' && awaitingReboot && onSetup ? () => onSetup(true) : undefined,
                onInstall: trainer.connection_mode === 'ssh' && onSetup ? () => onSetup(false) : undefined,
            })}
        </Table.ExpandableRow>
    );
};

type TrainingTargetsTableProps = {
    rows: TrainingTargetRow[];
    onEdit: (row: TrainingTargetRow) => void;
    onDelete: (row: TrainingTargetRow) => void;
    onSetup?: (row: TrainingTargetRow, reboot: boolean) => void;
};

export const TrainingTargetsTable = ({ rows, onEdit, onDelete, onSetup }: TrainingTargetsTableProps) => {
    const [expandedId, setExpandedId] = useState<string | undefined>(
        rows[0] ? trainingTargetRowId(rows[0]) : undefined
    );

    const toggleExpanded = (id: string) => setExpandedId((current) => (current === id ? undefined : id));

    return (
        <Table columns={TRAINING_TARGET_COLUMNS} isEmphasized>
            {rows.map((row) => {
                const id = trainingTargetRowId(row);
                const isExpanded = expandedId === id;

                return (
                    <DirectUrlTargetRow
                        key={id}
                        trainer={row.trainer}
                        isExpanded={isExpanded}
                        onExpandedChange={() => toggleExpanded(id)}
                        onExpand={() => setExpandedId(id)}
                        onEdit={() => onEdit(row)}
                        onDelete={() => onDelete(row)}
                        onSetup={onSetup ? (reboot) => onSetup(row, reboot) : undefined}
                    />
                );
            })}
        </Table>
    );
};
