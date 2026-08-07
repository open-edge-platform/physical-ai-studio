import { ReactNode, useState } from 'react';

import { ActionButton, Badge, Flex, Grid, Icon, Item, Key, Menu, MenuTrigger, StatusLight, Text } from '@geti-ui/ui';
import { ChevronRightSmallLight, MoreMenu } from '@geti-ui/ui/icons';

import { SchemaRemoteServer, SchemaRemoteTrainer } from '../../../api/openapi-spec';
import { remoteServerStatusLabel, remoteServerStatusVariant } from '../remote-server-status-utils';
import { deviceTypes, getDisplayHealth, healthLabel, healthVariant } from '../remote-trainer-health-utils';
import { RemoteServerDetail } from './remote-server-detail/remote-server-detail';
import { RemoteTrainerDetail } from './remote-trainer-detail/remote-trainer-detail';
import { TrainingTargetRow, trainingTargetRowId } from './training-target-row';
import { useRemoteServerCheckMutation } from './use-remote-server-check-mutation';
import { useRemoteServersStatus } from './use-remote-servers-status';
import { useRemoteTrainersHealth } from './use-remote-trainers-health';

import classes from './training-targets-table.module.css';

export const TRAINING_TARGETS_GRID_COLUMNS = 'max-content 1fr 1fr 1fr 1fr auto';

const DEVICE_BADGE_CLASSES: Record<string, string> = {
    CUDA: classes.cudaBadge,
    XPU: classes.xpuBadge,
};

export const TrainingTargetsTableHeader = () => (
    <div className={classes.tableHeader}>
        <div />
        <Text>Name</Text>
        <Text>Connection</Text>
        <Text>Status</Text>
        <Text>Compute</Text>
        <div />
    </div>
);

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

type TargetRowShellProps = {
    id: string;
    name: string;
    connectionLabel: string;
    statusVariant: StatusVariant;
    statusLabel: string;
    deviceTypes: string[];
    computeDetail: string;
    kindBadge: 'SSH' | 'Direct URL';
    isChecking: boolean;
    isExpanded: boolean;
    onToggleExpanded: () => void;
    onCheck?: () => void;
    onEdit: () => void;
    onDelete: () => void;
    children: ReactNode;
};

const TargetRowShell = ({
    id,
    name,
    connectionLabel,
    statusVariant,
    statusLabel,
    deviceTypes: types,
    computeDetail,
    kindBadge,
    isChecking,
    isExpanded,
    onToggleExpanded,
    onCheck,
    onEdit,
    onDelete,
    children,
}: TargetRowShellProps) => {
    const contentId = `training-target-detail-${id}`;

    return (
        <div
            data-testid={`training-target-row-${id}`}
            onClick={onToggleExpanded}
            className={`${classes.trainerRow} ${isExpanded ? classes.rowExpanded : ''}`}
        >
            <ActionButton
                isQuiet
                aria-expanded={isExpanded}
                aria-controls={contentId}
                aria-label={`Show details for ${name}`}
                onPress={onToggleExpanded}
                UNSAFE_className={classes.disclosureButton}
            >
                <Icon>
                    <ChevronRightSmallLight />
                </Icon>
            </ActionButton>

            <Text>{name}</Text>

            <Flex gap='size-100' alignItems='center'>
                <Badge variant='neutral' UNSAFE_className={classes.kindBadge}>
                    {kindBadge}
                </Badge>
                <Text UNSAFE_className={classes.trainerUrl}>{connectionLabel}</Text>
            </Flex>

            <StatusLight variant={statusVariant} UNSAFE_className={classes.healthStatus}>
                {statusLabel}
            </StatusLight>

            <Flex gap='size-100' alignItems='center' wrap>
                {types.map((type) => (
                    <Badge key={type} variant='neutral' UNSAFE_className={DEVICE_BADGE_CLASSES[type]}>
                        {type}
                    </Badge>
                ))}
                <Text UNSAFE_className={classes.cardMetaText}>{computeDetail}</Text>
            </Flex>

            <Flex gap='size-100' wrap UNSAFE_className={classes.actionsCell}>
                <TargetMenuActions
                    targetName={name}
                    onCheck={onCheck}
                    onEdit={onEdit}
                    onDelete={onDelete}
                    isChecking={isChecking}
                />
            </Flex>

            {isExpanded && (
                <Grid id={contentId} gridColumn={'1/-1'} marginTop={'size-150'}>
                    {children}
                </Grid>
            )}
        </div>
    );
};

type DirectUrlTargetRowProps = {
    trainer: SchemaRemoteTrainer;
    isExpanded: boolean;
    onToggleExpanded: () => void;
    onExpand: () => void;
    onEdit: () => void;
    onDelete: () => void;
};

const DirectUrlTargetRow = ({
    trainer,
    isExpanded,
    onToggleExpanded,
    onExpand,
    onEdit,
    onDelete,
}: DirectUrlTargetRowProps) => {
    const health = useRemoteTrainersHealth([trainer.id]).get(trainer.id);
    const displayHealth = getDisplayHealth(trainer.id, health?.health, health?.hasError ?? false);
    const isChecking = health?.isChecking ?? false;
    const types = deviceTypes(displayHealth);

    return (
        <TargetRowShell
            id={trainer.id}
            name={trainer.name}
            connectionLabel={trainer.url}
            statusVariant={healthVariant(displayHealth, isChecking)}
            statusLabel={healthLabel(displayHealth, isChecking)}
            deviceTypes={types}
            computeDetail={
                displayHealth?.devices?.at(0)?.name ?? (isChecking ? 'Checking capability…' : 'Not reported')
            }
            kindBadge='Direct URL'
            isChecking={isChecking}
            isExpanded={isExpanded}
            onToggleExpanded={onToggleExpanded}
            onCheck={() => {
                void health?.checkHealth();
                onExpand();
            }}
            onEdit={onEdit}
            onDelete={onDelete}
        >
            <RemoteTrainerDetail remoteTrainer={trainer} health={displayHealth} isChecking={isChecking} />
        </TargetRowShell>
    );
};

type SshTargetRowProps = {
    server: SchemaRemoteServer;
    isExpanded: boolean;
    onToggleExpanded: () => void;
    onExpand: () => void;
    onEdit: () => void;
    onDelete: () => void;
};

const SshTargetRow = ({ server, isExpanded, onToggleExpanded, onExpand, onEdit, onDelete }: SshTargetRowProps) => {
    const entry = useRemoteServersStatus([server.id]).get(server.id);
    const isChecking = entry?.isChecking ?? false;
    const checkMutation = useRemoteServerCheckMutation();

    return (
        <TargetRowShell
            id={server.id}
            name={server.name}
            connectionLabel={`ssh ${server.ssh_host_alias}`}
            statusVariant={remoteServerStatusVariant(entry?.status, isChecking)}
            statusLabel={remoteServerStatusLabel(entry?.status, isChecking)}
            deviceTypes={[server.device_type.toUpperCase()]}
            computeDetail={isChecking ? 'Checking…' : `ssh host alias: ${server.ssh_host_alias}`}
            kindBadge='SSH'
            isChecking={isChecking}
            isExpanded={isExpanded}
            onToggleExpanded={onToggleExpanded}
            onCheck={() => {
                void entry?.checkStatus();
                onExpand();
            }}
            onEdit={onEdit}
            onDelete={onDelete}
        >
            <RemoteServerDetail
                remoteServer={server}
                status={entry?.status}
                isChecking={isChecking}
                tier2Result={checkMutation.data}
                tier2CheckedAt={checkMutation.data?.checked_at}
                isRunningTier2={checkMutation.isPending}
                onTestConnection={() => checkMutation.mutate({ params: { path: { remote_server_id: server.id } } })}
            />
        </TargetRowShell>
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
        <Grid columns={TRAINING_TARGETS_GRID_COLUMNS} columnGap='size-100' width='100%'>
            <TrainingTargetsTableHeader />
            {rows.map((row) => {
                const id = trainingTargetRowId(row);
                const isExpanded = expandedId === id;

                if (row.kind === 'direct-url') {
                    return (
                        <DirectUrlTargetRow
                            key={id}
                            trainer={row.trainer}
                            isExpanded={isExpanded}
                            onToggleExpanded={() => toggleExpanded(id)}
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
                        onToggleExpanded={() => toggleExpanded(id)}
                        onExpand={() => setExpandedId(id)}
                        onEdit={() => onEdit(row)}
                        onDelete={() => onDelete(row)}
                    />
                );
            })}
        </Grid>
    );
};
