import { useState } from 'react';

import {
    ActionButton,
    Badge,
    Flex,
    Item,
    Key,
    Menu,
    MenuTrigger,
    StatusLight,
    Text,
    Tooltip,
    TooltipTrigger,
} from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { SchemaRemoteTrainer } from '../../../api/openapi-spec';
import { Table, TableColumn } from '../../../components/table/table';
import { connectionModeLabel } from '../remote-trainer-connection-utils';
import { deviceTypes, getDisplayHealth, healthLabel, healthVariant } from '../remote-trainer-health-utils';
import { RemoteTrainerDetail } from './remote-trainer-detail/remote-trainer-detail';
import { TrainingTargetRow, trainingTargetRowId } from './training-target-row';
import { useRemoteTrainersHealth } from './use-remote-trainers-health';

import classes from './training-targets-table.module.css';

const DEVICE_BADGE_CLASSES: Record<string, string> = {
    CUDA: classes.cudaBadge,
    XPU: classes.xpuBadge,
};

export const TRAINING_TARGET_COLUMNS: TableColumn[] = [
    { width: 'max-content' },
    { width: '1fr', header: 'Name' },
    { width: '1fr', header: 'Connection' },
    { width: '1fr', header: 'Status' },
    { width: '1fr', header: 'Compute' },
    { width: 'auto', align: 'end' },
];

const TARGET_MENU_ACTION_ITEMS = {
    CHECK_STATUS: 'check_status',
    EDIT: 'Edit',
    DELETE: 'Delete',
    INSTALL: 'install_prerequisites',
    REBOOT: 'reboot_after_install',
};

type TargetMenuActionsProps = {
    targetName: string;
    onCheck?: () => void;
    onEdit: () => void;
    onDelete: () => void;
    onInstall?: () => void;
    onReboot?: () => void;
    isChecking: boolean;
};

const TargetMenuActions = ({
    targetName,
    onCheck,
    onEdit,
    onDelete,
    onInstall,
    onReboot,
    isChecking,
}: TargetMenuActionsProps) => {
    const items = [
        { key: TARGET_MENU_ACTION_ITEMS.EDIT, label: 'Edit' },
        { key: TARGET_MENU_ACTION_ITEMS.DELETE, label: 'Delete' },
        { key: TARGET_MENU_ACTION_ITEMS.CHECK_STATUS, label: 'Check status' },
        ...(onInstall ? [{ key: TARGET_MENU_ACTION_ITEMS.INSTALL, label: 'Install prerequisites' }] : []),
        ...(onReboot ? [{ key: TARGET_MENU_ACTION_ITEMS.REBOOT, label: 'Reboot to finish setup' }] : []),
    ];
    const handleAction = (action: Key) => {
        if (action === TARGET_MENU_ACTION_ITEMS.CHECK_STATUS) {
            onCheck?.();
        } else if (action === TARGET_MENU_ACTION_ITEMS.EDIT) {
            onEdit();
        } else if (action === TARGET_MENU_ACTION_ITEMS.DELETE) {
            onDelete();
        } else if (action === TARGET_MENU_ACTION_ITEMS.INSTALL) {
            onInstall?.();
        } else if (action === TARGET_MENU_ACTION_ITEMS.REBOOT) {
            onReboot?.();
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
                disabledKeys={isChecking || onCheck === undefined ? [TARGET_MENU_ACTION_ITEMS.CHECK_STATUS] : undefined}
            >
                {(item) => <Item key={item.key}>{item.label}</Item>}
            </Menu>
        </MenuTrigger>
    );
};

type StatusVariant = 'positive' | 'notice' | 'negative' | 'neutral' | 'yellow';

type TargetRowContentProps = {
    name: string;
    connectionLabel: string;
    connectionModeText: string;
    statusVariant: StatusVariant;
    statusLabel: string;
    deviceTypes: string[];
    computeDetail: string;
    isChecking: boolean;
    onCheck?: () => void;
    onEdit: () => void;
    onDelete: () => void;
    onInstall?: () => void;
    onReboot?: () => void;
};

/**
 * The five cells rendered inside a row, one per column after the disclosure
 * column. Returns a flat array (not a fragment) so `Table.ExpandableRow`'s
 * `Children.toArray` sees five distinct children matching the five columns,
 * rather than a single wrapped element.
 */
const targetRowCells = ({
    name,
    connectionLabel,
    connectionModeText,
    statusVariant,
    statusLabel,
    deviceTypes: types,
    computeDetail,
    isChecking,
    onCheck,
    onEdit,
    onDelete,
    onInstall,
    onReboot,
}: TargetRowContentProps) => [
    <Text key='name'>{name}</Text>,

    <TooltipTrigger key='connection' delay={300}>
        <ActionButton isQuiet UNSAFE_className={classes.kindBadgeTrigger} aria-label={connectionLabel}>
            <Badge variant='neutral' UNSAFE_className={classes.kindBadge}>
                {connectionModeText}
            </Badge>
        </ActionButton>
        <Tooltip>{connectionLabel}</Tooltip>
    </TooltipTrigger>,

    <StatusLight key='status' variant={statusVariant} UNSAFE_className={classes.healthStatus}>
        {statusLabel}
    </StatusLight>,

    <Flex key='compute' gap='size-100' alignItems='center' wrap>
        {types.map((type) => (
            <Badge key={type} variant='neutral' UNSAFE_className={DEVICE_BADGE_CLASSES[type]}>
                {type}
            </Badge>
        ))}
        <Text UNSAFE_className={classes.cardMetaText}>{computeDetail}</Text>
    </Flex>,

    <div key='actions' onClick={(event) => event.stopPropagation()}>
        <TargetMenuActions
            targetName={name}
            onCheck={onCheck}
            onEdit={onEdit}
            onDelete={onDelete}
            onInstall={onInstall}
            onReboot={onReboot}
            isChecking={isChecking}
        />
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
    const types = deviceTypes(displayHealth);
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
                connectionLabel: trainer.url,
                connectionModeText: connectionModeLabel(trainer.connection_mode),
                statusVariant: healthVariant(displayHealth, isChecking),
                statusLabel: healthLabel(displayHealth, isChecking),
                deviceTypes: types,
                computeDetail:
                    displayHealth?.devices?.at(0)?.name ?? (isChecking ? 'Checking capability…' : 'Not reported'),
                isChecking,
                onCheck: () => {
                    void health?.checkHealth();
                    onExpand();
                },
                onEdit,
                onDelete,
                onInstall:
                    trainer.connection_mode === 'ssh' &&
                    displayHealth?.status !== 'starting' &&
                    !awaitingReboot &&
                    onSetup
                        ? () => onSetup(false)
                        : undefined,
                onReboot:
                    trainer.connection_mode === 'ssh' && awaitingReboot && onSetup ? () => onSetup(true) : undefined,
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
