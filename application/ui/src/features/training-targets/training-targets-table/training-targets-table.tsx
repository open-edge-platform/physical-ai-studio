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

import { SchemaPreflightResult, SchemaRemoteServer, SchemaRemoteTrainer } from '../../../api/openapi-spec';
import { Table, TableColumn } from '../../../components/table/table';
import {
    remoteServerComputeDetail,
    remoteServerStatusLabel,
    remoteServerStatusVariant,
} from '../remote-server-status-utils';
import { deviceTypes, getDisplayHealth, healthLabel, healthVariant } from '../remote-trainer-health-utils';
import { RemoteServerDetail } from './remote-server-detail/remote-server-detail';
import { RemoteTrainerDetail } from './remote-trainer-detail/remote-trainer-detail';
import { TrainingTargetRow, trainingTargetRowId } from './training-target-row';
import { useIsRemoteServerCheckRunning } from './use-is-remote-server-check-running';
import { useRemoteServerCheckMutation } from './use-remote-server-check-mutation';
import { useRemoteServersStatus } from './use-remote-servers-status';
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
};

type TargetMenuActionsProps = {
    targetName: string;
    onCheck?: () => void;
    onEdit: () => void;
    onDelete: () => void;
    isChecking: boolean;
};

const TargetMenuActions = ({ targetName, onCheck, onEdit, onDelete, isChecking }: TargetMenuActionsProps) => {
    const handleAction = (action: Key) => {
        if (action === TARGET_MENU_ACTION_ITEMS.CHECK_STATUS) {
            onCheck?.();
        } else if (action === TARGET_MENU_ACTION_ITEMS.EDIT) {
            onEdit();
        } else if (action === TARGET_MENU_ACTION_ITEMS.DELETE) {
            onDelete();
        }
    };

    return (
        <MenuTrigger>
            <ActionButton aria-label={`More actions ${targetName}`} isQuiet>
                <MoreMenu />
            </ActionButton>
            <Menu
                onAction={handleAction}
                disabledKeys={isChecking || onCheck === undefined ? [TARGET_MENU_ACTION_ITEMS.CHECK_STATUS] : undefined}
            >
                <Item key={TARGET_MENU_ACTION_ITEMS.EDIT}>Edit</Item>
                <Item key={TARGET_MENU_ACTION_ITEMS.DELETE}>Delete</Item>
                <Item key={TARGET_MENU_ACTION_ITEMS.CHECK_STATUS}>Check status</Item>
            </Menu>
        </MenuTrigger>
    );
};

type StatusVariant = 'positive' | 'notice' | 'negative' | 'neutral' | 'yellow';

type TargetRowContentProps = {
    name: string;
    connectionLabel: string;
    statusVariant: StatusVariant;
    statusLabel: string;
    deviceTypes: string[];
    computeDetail: string;
    kindBadge: 'SSH' | 'Direct URL';
    isChecking: boolean;
    onCheck?: () => void;
    onEdit: () => void;
    onDelete: () => void;
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
    statusVariant,
    statusLabel,
    deviceTypes: types,
    computeDetail,
    kindBadge,
    isChecking,
    onCheck,
    onEdit,
    onDelete,
}: TargetRowContentProps) => [
    <Text key='name'>{name}</Text>,

    <TooltipTrigger key='connection' delay={300}>
        <ActionButton isQuiet UNSAFE_className={classes.kindBadgeTrigger} aria-label={connectionLabel}>
            <Badge variant='neutral' UNSAFE_className={classes.kindBadge}>
                {kindBadge}
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
};

const DirectUrlTargetRow = ({
    trainer,
    isExpanded,
    onExpandedChange,
    onExpand,
    onEdit,
    onDelete,
}: DirectUrlTargetRowProps) => {
    const health = useRemoteTrainersHealth([trainer.id]).get(trainer.id);
    const displayHealth = getDisplayHealth(trainer.id, health?.health, health?.hasError ?? false);
    const isChecking = health?.isChecking ?? false;
    const types = deviceTypes(displayHealth);

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
                statusVariant: healthVariant(displayHealth, isChecking),
                statusLabel: healthLabel(displayHealth, isChecking),
                deviceTypes: types,
                computeDetail:
                    displayHealth?.devices?.at(0)?.name ?? (isChecking ? 'Checking capability…' : 'Not reported'),
                kindBadge: 'Direct URL',
                isChecking,
                onCheck: () => {
                    void health?.checkHealth();
                    onExpand();
                },
                onEdit,
                onDelete,
            })}
        </Table.ExpandableRow>
    );
};

type SshTargetRowProps = {
    server: SchemaRemoteServer;
    isExpanded: boolean;
    onExpandedChange: (isExpanded: boolean) => void;
    onExpand: () => void;
    onEdit: () => void;
    onDelete: () => void;
};

/**
 * Reconstructs the last persisted Tier 2 result from the server record itself
 * (`last_check_checks`/`last_check_at`), so a row that mounts fresh - e.g. after
 * navigating back to the training-targets page, or after verifying from the
 * post-save dialog's own mutation instance - still shows the server's real
 * verification state instead of "Not verified yet" just because *this*
 * component's local check mutation has never fired.
 */
const persistedTier2Result = (server: SchemaRemoteServer): SchemaPreflightResult | undefined => {
    if (server.last_check_checks === undefined || server.last_check_checks.length === 0) return undefined;

    return {
        remote_server_id: server.id,
        checks: server.last_check_checks,
        checked_at: server.last_check_at ?? new Date(0).toISOString(),
    };
};

const SshTargetRow = ({ server, isExpanded, onExpandedChange, onExpand, onEdit, onDelete }: SshTargetRowProps) => {
    const entry = useRemoteServersStatus([server.id]).get(server.id);
    const isChecking = entry?.isChecking ?? false;
    const checkMutation = useRemoteServerCheckMutation();
    const tier2Result = checkMutation.data ?? persistedTier2Result(server);
    const tier2CheckedAt = checkMutation.data?.checked_at ?? server.last_check_at ?? undefined;
    // Reflects any in-flight Tier 2 check for this server, including one
    // fired from the post-save dialog's own (now-unmounted) mutation
    // instance - not just this row's local `checkMutation.isPending`.
    const isRunningTier2 = useIsRemoteServerCheckRunning(server.id) || checkMutation.isPending;

    return (
        <Table.ExpandableRow
            id={`training-target-row-${server.id}`}
            label={server.name}
            isExpanded={isExpanded}
            onExpandedChange={onExpandedChange}
            detail={
                <RemoteServerDetail
                    remoteServer={server}
                    status={entry?.status}
                    isChecking={isChecking}
                    tier2Result={tier2Result}
                    tier2CheckedAt={tier2CheckedAt}
                    isRunningTier2={isRunningTier2}
                    onTestConnection={() => checkMutation.mutate({ params: { path: { remote_server_id: server.id } } })}
                />
            }
        >
            {targetRowCells({
                name: server.name,
                connectionLabel: `ssh ${server.ssh_host_alias}`,
                statusVariant: remoteServerStatusVariant(entry?.status, isChecking),
                statusLabel: remoteServerStatusLabel(entry?.status, isChecking),
                deviceTypes: [server.device_type.toUpperCase()],
                computeDetail: isChecking ? 'Checking…' : (remoteServerComputeDetail(entry?.status) ?? 'Not reported'),
                kindBadge: 'SSH',
                isChecking,
                onCheck: () => {
                    void entry?.checkStatus();
                    onExpand();
                },
                onEdit,
                onDelete,
            })}
        </Table.ExpandableRow>
    );
};

type TrainingTargetsTableProps = {
    rows: TrainingTargetRow[];
    onEdit: (row: TrainingTargetRow) => void;
    onDelete: (row: TrainingTargetRow) => void;
};

export const TrainingTargetsTable = ({ rows, onEdit, onDelete }: TrainingTargetsTableProps) => {
    const [expandedId, setExpandedId] = useState<string | undefined>(
        rows[0] ? trainingTargetRowId(rows[0]) : undefined
    );

    const toggleExpanded = (id: string) => setExpandedId((current) => (current === id ? undefined : id));

    return (
        <Table columns={TRAINING_TARGET_COLUMNS} isEmphasized>
            {rows.map((row) => {
                const id = trainingTargetRowId(row);
                const isExpanded = expandedId === id;

                if (row.kind === 'direct-url') {
                    return (
                        <DirectUrlTargetRow
                            key={id}
                            trainer={row.trainer}
                            isExpanded={isExpanded}
                            onExpandedChange={() => toggleExpanded(id)}
                            onExpand={() => setExpandedId(id)}
                            onEdit={() => onEdit(row)}
                            onDelete={() => onDelete(row)}
                        />
                    );
                }

                return (
                    <SshTargetRow
                        key={id}
                        server={row.server}
                        isExpanded={isExpanded}
                        onExpandedChange={() => toggleExpanded(id)}
                        onExpand={() => setExpandedId(id)}
                        onEdit={() => onEdit(row)}
                        onDelete={() => onDelete(row)}
                    />
                );
            })}
        </Table>
    );
};
