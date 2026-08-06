import { useState } from 'react';

import {
    ActionButton,
    DialogContainer,
    Flex,
    Grid,
    Icon,
    Item,
    Key,
    Menu,
    MenuTrigger,
    StatusLight,
    Text,
    View,
} from '@geti-ui/ui';
import { ChevronDownSmallLight, ChevronRightSmallLight, MoreMenu } from '@geti-ui/ui/icons';

import { SchemaRemoteTrainer, SchemaRemoteTrainerHealth } from '../../../api/openapi-spec';
import {
    deviceTagClass,
    deviceTypes,
    getDisplayHealth,
    healthLabel,
    healthVariant,
} from '../remote-trainer-health-utils';
import { DeleteRemoteTrainerDialog } from './delete-remote-trainer-dialog';
import { RemoteTrainerDetail } from './remote-trainer-detail';
import { RemoteTrainerForm } from './remote-trainer-form';
import { useRemoteTrainersHealth } from './use-remote-trainers-health';

import classes from './remote-trainers-table.module.css';

export const REMOTE_TRAINERS_GRID_COLUMNS = 'max-content 1fr 1fr 1fr auto';

export const RemoteTrainersTableHeader = () => (
    <Grid
        columns={REMOTE_TRAINERS_GRID_COLUMNS}
        alignItems='center'
        width='100%'
        gap={'size-100'}
        UNSAFE_className={classes.tableHeader}
    >
        <div />
        <Text>Name</Text>
        <Text>Status</Text>
        <Text>Compute</Text>
        <div />
    </Grid>
);

const REMOTE_TRAINERS_MENU_ACTION_ITEMS = {
    CHECK_STATE: 'check_state',
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
        if (action === REMOTE_TRAINERS_MENU_ACTION_ITEMS.CHECK_STATE) {
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
                <MoreMenu style={{ fill: '#fff' }} />
            </ActionButton>
            <Menu
                onAction={handleAction}
                disabledKeys={isChecking ? [REMOTE_TRAINERS_MENU_ACTION_ITEMS.CHECK_STATE] : undefined}
            >
                <Item key={REMOTE_TRAINERS_MENU_ACTION_ITEMS.CHECK_STATE}>Check state</Item>
                <Item key={REMOTE_TRAINERS_MENU_ACTION_ITEMS.EDIT}>Edit</Item>
                <Item key={REMOTE_TRAINERS_MENU_ACTION_ITEMS.DELETE}>Delete</Item>
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
    const contentId = `remote-trainer-detail-${remoteTrainer.id}`;

    return (
        <div
            data-testid={`remote-trainer-row-${remoteTrainer.id}`}
            onClick={onToggleExpanded}
            className={classes.trainerRow}
        >
            <Grid
                columns={REMOTE_TRAINERS_GRID_COLUMNS}
                alignItems='center'
                width='100%'
                UNSAFE_className={`${classes.row} ${isExpanded ? classes.rowExpanded : ''}`}
            >
                <ActionButton
                    isQuiet
                    aria-expanded={isExpanded}
                    aria-controls={contentId}
                    aria-label={`Show details for ${remoteTrainer.name}`}
                    onPress={onToggleExpanded}
                    UNSAFE_className={classes.disclosureButton}
                >
                    <Icon>{isExpanded ? <ChevronDownSmallLight /> : <ChevronRightSmallLight />}</Icon>
                </ActionButton>

                <Flex gap='size-100' alignItems='center' wrap>
                    <Text UNSAFE_className={classes.trainerName}>{remoteTrainer.name}</Text>
                </Flex>

                <StatusLight variant={healthVariant(health, isChecking)}>{healthLabel(health, isChecking)}</StatusLight>

                <Flex gap='size-100' alignItems='center' wrap>
                    {types.map((type) => (
                        <Text key={type} UNSAFE_className={deviceTagClass(type)}>
                            {type}
                        </Text>
                    ))}
                    <Text UNSAFE_className={classes.cardMetaText}>
                        {health?.devices?.at(0)?.name ?? (isChecking ? 'Checking capability…' : 'Not reported')}
                    </Text>
                </Flex>

                <Flex gap='size-100' wrap UNSAFE_className={classes.actionsCell}>
                    <RemoteTrainersMenuActions
                        remoteTrainerName={remoteTrainer.name}
                        onCheck={onCheck}
                        onEdit={onEdit}
                        onDelete={onDelete}
                        isChecking={isChecking}
                    />
                </Flex>
            </Grid>

            {isExpanded && (
                <View id={contentId}>
                    <RemoteTrainerDetail remoteTrainer={remoteTrainer} health={health} isChecking={isChecking} />
                </View>
            )}
        </div>
    );
};

export type RemoteTrainerAction =
    | { type: 'create' }
    | { type: 'edit'; remoteTrainer: SchemaRemoteTrainer }
    | { type: 'delete'; remoteTrainer: SchemaRemoteTrainer }
    | undefined;

type RemoteTrainersTableProps = {
    remoteTrainers: SchemaRemoteTrainer[];
    setAction: (action: RemoteTrainerAction) => void;
    action: RemoteTrainerAction;
};

export const RemoteTrainersTable = ({ remoteTrainers, setAction, action }: RemoteTrainersTableProps) => {
    const [expandedRemoteTrainerId, setExpandedRemoteTrainerId] = useState<string | undefined>(remoteTrainers[0]?.id);

    const health = useRemoteTrainersHealth(remoteTrainers.map((remoteTrainer) => remoteTrainer.id));

    return (
        <>
            <RemoteTrainersTableHeader />
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
                        onCheck={() => void entry?.checkHealth()}
                        onEdit={() => setAction({ type: 'edit', remoteTrainer })}
                        onDelete={() => setAction({ type: 'delete', remoteTrainer })}
                    />
                );
            })}
            <DialogContainer onDismiss={() => setAction(undefined)}>
                {(action?.type === 'create' || action?.type === 'edit') && (
                    <RemoteTrainerForm
                        remoteTrainer={action.type === 'edit' ? action.remoteTrainer : undefined}
                        close={() => setAction(undefined)}
                    />
                )}
                {action?.type === 'delete' && (
                    <DeleteRemoteTrainerDialog
                        remoteTrainer={action.remoteTrainer}
                        onCancel={() => setAction(undefined)}
                        onDeleted={() => {
                            setAction(undefined);
                            setExpandedRemoteTrainerId(undefined);
                        }}
                    />
                )}
            </DialogContainer>
        </>
    );
};
