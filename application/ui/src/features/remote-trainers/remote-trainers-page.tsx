import { useState } from 'react';

import {
    ActionButton,
    AlertDialog,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    Divider,
    Flex,
    Form,
    Heading,
    Icon,
    StatusLight,
    Text,
    TextField,
    View,
} from '@geti-ui/ui';
import { Add, Delete, Edit, Refresh } from '@geti-ui/ui/icons';

import { $api } from '../../api/client';
import { getApiErrorMessage } from '../../api/errors';
import { SchemaRemoteTrainer, SchemaRemoteTrainerHealth } from '../../api/openapi-spec';
import { useRemoteTrainerHealth } from './use-remote-trainer-health';

import classes from './remote-trainers-page.module.css';

type RemoteTrainerFormProps = {
    remoteTrainer?: SchemaRemoteTrainer;
    close: () => void;
};

const RemoteTrainerForm = ({ remoteTrainer, close }: RemoteTrainerFormProps) => {
    const [name, setName] = useState(remoteTrainer?.name ?? '');
    const [url, setUrl] = useState(remoteTrainer?.url ?? '');
    const [error, setError] = useState<string>();
    const isEditing = remoteTrainer !== undefined;

    const createMutation = $api.useMutation('post', '/api/remote-trainers', {
        meta: {
            invalidates: [['get', '/api/remote-trainers']],
        },
    });
    const updateMutation = $api.useMutation('patch', '/api/remote-trainers/{remote_trainer_id}', {
        meta: {
            invalidates: [['get', '/api/remote-trainers']],
        },
    });
    const isPending = createMutation.isPending || updateMutation.isPending;

    const save = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setError(undefined);
        const normalizedName = name.trim();

        try {
            if (isEditing) {
                await updateMutation.mutateAsync({
                    params: { path: { remote_trainer_id: remoteTrainer.id } },
                    body: { name: normalizedName, url },
                });
            } else {
                await createMutation.mutateAsync({ body: { name: normalizedName, url } });
            }
            close();
        } catch (mutationError) {
            setError(getApiErrorMessage(mutationError) ?? 'The remote trainer could not be saved. Try again.');
        }
    };

    return (
        <Form onSubmit={save} validationBehavior='native' width='size-6000'>
            <Dialog>
                <Heading>{isEditing ? 'Edit remote trainer' : 'Add remote trainer'}</Heading>
                <Divider />
                <Content>
                    <Flex direction='column' gap='size-200'>
                        <div className={classes.formIntro}>
                            <Text UNSAFE_className={classes.formIntroLabel}>Connection type</Text>
                            <Text UNSAFE_className={classes.connectionTypeOption}>Direct trainer URL</Text>
                            <Text UNSAFE_className={classes.formIntroDescription}>
                                Direct endpoints run an already-managed trainer. Studio does not provision the host.
                            </Text>
                        </div>
                        <TextField
                            // eslint-disable-next-line jsx-a11y/no-autofocus
                            autoFocus
                            isRequired
                            label='Name'
                            value={name}
                            onChange={setName}
                            width='100%'
                        />
                        <TextField
                            isRequired
                            label='Trainer URL'
                            type='url'
                            value={url}
                            onChange={setUrl}
                            description='Use the endpoint URL that accepts Physical AI Studio training jobs.'
                            width='100%'
                        />
                        {error && <Text UNSAFE_className={classes.errorMessage}>{error}</Text>}
                    </Flex>
                </Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={close} isDisabled={isPending}>
                        Cancel
                    </Button>
                    <Button variant='accent' type='submit' isDisabled={!name.trim() || !url} isPending={isPending}>
                        {isEditing ? 'Save changes' : 'Add trainer'}
                    </Button>
                </ButtonGroup>
            </Dialog>
        </Form>
    );
};

type DeleteRemoteTrainerDialogProps = {
    remoteTrainer: SchemaRemoteTrainer;
    onDeleted: () => void;
};

const DeleteRemoteTrainerDialog = ({ remoteTrainer, onDeleted }: DeleteRemoteTrainerDialogProps) => {
    const [error, setError] = useState<string>();
    const deleteMutation = $api.useMutation('delete', '/api/remote-trainers/{remote_trainer_id}', {
        meta: {
            invalidates: [['get', '/api/remote-trainers']],
        },
    });

    const remove = async () => {
        setError(undefined);
        try {
            await deleteMutation.mutateAsync({ params: { path: { remote_trainer_id: remoteTrainer.id } } });
            onDeleted();
        } catch (mutationError) {
            setError(getApiErrorMessage(mutationError) ?? 'The remote trainer could not be deleted. Try again.');
        }
    };

    return (
        <AlertDialog
            title='Delete remote trainer'
            variant='warning'
            primaryActionLabel='Delete'
            onPrimaryAction={remove}
            isPrimaryActionDisabled={deleteMutation.isPending}
        >
            <Flex direction='column' gap='size-150'>
                <Text>
                    Delete {remoteTrainer.name}? Submitted remote jobs retain their pinned endpoint URL and are not
                    changed.
                </Text>
                {error && <Text UNSAFE_className={classes.errorMessage}>{error}</Text>}
            </Flex>
        </AlertDialog>
    );
};

type RemoteTrainerDetailProps = {
    remoteTrainer: SchemaRemoteTrainer;
    health?: SchemaRemoteTrainerHealth;
    isChecking: boolean;
    onCheck: () => void;
    onEdit: () => void;
    onDelete: () => void;
};

const healthLabel = (health?: SchemaRemoteTrainerHealth, isChecking = false) => {
    if (isChecking) return 'Checking…';
    if (health === undefined) return 'Not checked';
    if (health.reason_code === 'check_failed') return 'Check failed';
    return health.status === 'healthy' ? 'Healthy' : health.status === 'degraded' ? 'Degraded' : 'Unreachable';
};

const healthVariant = (health?: SchemaRemoteTrainerHealth, isChecking = false) => {
    if (isChecking || health === undefined) return 'neutral' as const;
    return health.status === 'healthy'
        ? ('positive' as const)
        : health.status === 'degraded'
          ? ('yellow' as const)
          : ('negative' as const);
};

const healthDescription = (health?: SchemaRemoteTrainerHealth) => {
    if (health === undefined) return 'Connection status has not been checked.';
    if (health.status === 'healthy') return 'The trainer health endpoint and device report are available.';
    switch (health.reason_code) {
        case 'timeout':
            return 'The trainer did not respond within five seconds.';
        case 'connection_failed':
            return 'Studio could not connect to the configured trainer URL.';
        case 'http_error':
            return 'The trainer returned an error response.';
        case 'unhealthy':
            return 'The trainer health endpoint did not report a healthy status.';
        case 'check_failed':
            return 'Studio could not complete the health check. Try again.';
        default:
            return 'The trainer returned an invalid device report.';
    }
};

type CheckState = 'positive' | 'yellow' | 'negative' | 'neutral';

type HealthCheckRowProps = {
    label: string;
    detail: string;
    state: CheckState;
    status: string;
};

const HealthCheckRow = ({ label, detail, state, status }: HealthCheckRowProps) => (
    <div className={classes.checkRow}>
        <span className={`${classes.checkIcon} ${classes[state]}`} aria-hidden='true'>
            {state === 'positive' ? '✓' : state === 'negative' ? '×' : state === 'yellow' ? '!' : '–'}
        </span>
        <span className={classes.checkContent}>
            <Text UNSAFE_className={classes.checkLabel}>{label}</Text>
            <Text UNSAFE_className={classes.checkDetail}>{detail}</Text>
        </span>
        <StatusLight variant={state}>{status}</StatusLight>
    </div>
);

const deviceTypes = (health?: SchemaRemoteTrainerHealth) => [
    ...new Set((health?.devices ?? []).map((device) => device.type.toUpperCase())),
];

const deviceTagClass = (type: string) =>
    `${classes.deviceTag} ${type === 'CUDA' ? classes.cudaTag : type === 'XPU' ? classes.xpuTag : ''}`;

const formatBytes = (bytes: number): string => {
    if (bytes <= 0) return '0 GB';
    const gib = bytes / 1024 ** 3;
    return gib >= 1024 ? `${(gib / 1024).toFixed(1)} TB` : `${gib.toFixed(1)} GB`;
};

const formatStorage = (storage: SchemaRemoteTrainerHealth['storage']) =>
    storage ? `${formatBytes(storage.free_bytes)} free of ${formatBytes(storage.total_bytes)}` : undefined;

const getCapabilityState = (health: SchemaRemoteTrainerHealth | undefined, isChecking: boolean): CheckState => {
    if (isChecking || health === undefined || health.status === 'unreachable') return 'neutral';
    return (health.devices?.length ?? 0) > 0 ? 'positive' : 'yellow';
};

const getStorageState = (health: SchemaRemoteTrainerHealth | undefined, isChecking: boolean): CheckState => {
    if (isChecking || health === undefined || health.status === 'unreachable') return 'neutral';
    return health.storage ? 'positive' : 'yellow';
};

const getDisplayHealth = (remoteTrainerId: string, health: SchemaRemoteTrainerHealth | undefined, hasError: boolean) =>
    health ??
    (hasError
        ? {
              remote_trainer_id: remoteTrainerId,
              status: 'unreachable' as const,
              checked_at: new Date().toISOString(),
              latency_ms: null,
              devices: [],
              reason_code: 'check_failed' as const,
          }
        : undefined);

const RemoteTrainerDetail = ({
    remoteTrainer,
    health,
    isChecking,
    onCheck,
    onEdit,
    onDelete,
}: RemoteTrainerDetailProps) => {
    const devices = health?.devices ?? [];
    const types = deviceTypes(health);
    const state = healthVariant(health, isChecking);
    const deviceReportIsInvalid = health?.reason_code === 'invalid_devices_response';
    const trainerHealthState = deviceReportIsInvalid ? 'positive' : state;
    const capabilityState = getCapabilityState(health, isChecking);
    const storageState = getStorageState(health, isChecking);
    const lastChecked = health ? new Date(health.checked_at).toLocaleString() : 'Not checked';

    return (
        <View UNSAFE_className={classes.detailPane}>
            <Flex
                justifyContent='space-between'
                gap='size-200'
                alignItems='start'
                wrap
                UNSAFE_className={classes.detailHeader}
            >
                <div>
                    <Flex gap='size-100' alignItems='center' wrap>
                        <Heading level={2}>{remoteTrainer.name}</Heading>
                        <Text UNSAFE_className={classes.connectionTag}>DIRECT TRAINER URL</Text>
                        {types.map((type) => (
                            <Text key={type} UNSAFE_className={deviceTagClass(type)}>
                                {type}
                            </Text>
                        ))}
                    </Flex>
                    <Text UNSAFE_className={classes.url}>last checked {lastChecked}</Text>
                </div>
                <Flex gap='size-100' wrap UNSAFE_className={classes.detailActions}>
                    <ActionButton
                        aria-label={`Check connection to ${remoteTrainer.name}`}
                        onPress={onCheck}
                        isDisabled={isChecking}
                    >
                        <Icon>
                            <Refresh />
                        </Icon>
                        <Text>Test connection</Text>
                    </ActionButton>
                    <ActionButton aria-label={`Edit ${remoteTrainer.name}`} onPress={onEdit}>
                        <Icon>
                            <Edit />
                        </Icon>
                        <Text>Edit</Text>
                    </ActionButton>
                    <div className={classes.destructiveAction}>
                        <ActionButton
                            aria-label={`Delete ${remoteTrainer.name}`}
                            onPress={onDelete}
                            UNSAFE_className={classes.deleteAction}
                        >
                            <Icon>
                                <Delete />
                            </Icon>
                            <Text>Delete</Text>
                        </ActionButton>
                    </div>
                </Flex>
            </Flex>

            <View UNSAFE_className={classes.detailSection}>
                <Heading level={3} UNSAFE_className={classes.sectionHeading}>
                    Health &amp; capability
                </Heading>
                <div className={classes.checkList}>
                    <HealthCheckRow
                        label='Trainer health endpoint'
                        detail={
                            isChecking
                                ? 'connection check in progress'
                                : health?.status === 'healthy' || deviceReportIsInvalid
                                  ? health?.latency_ms !== null && health?.latency_ms !== undefined
                                      ? `responded in ${health.latency_ms} ms and is ready for training requests`
                                      : 'ready for training requests'
                                  : health?.status === 'degraded'
                                    ? 'responded with a degraded status'
                                    : healthDescription(health)
                        }
                        state={trainerHealthState}
                        status={deviceReportIsInvalid ? 'Healthy' : healthLabel(health, isChecking)}
                    />
                    <HealthCheckRow
                        label='Compute capability'
                        detail={
                            devices.length > 0
                                ? devices.map((device) => `${device.type.toUpperCase()} · ${device.name}`).join(', ')
                                : health === undefined || isChecking
                                  ? 'awaiting device report'
                                  : 'no compute device reported'
                        }
                        state={capabilityState}
                        status={devices.length > 0 ? 'Available' : 'Unknown'}
                    />
                    <HealthCheckRow
                        label='Storage capacity'
                        detail={
                            formatStorage(health?.storage) ??
                            (health === undefined || isChecking ? 'awaiting storage report' : 'no storage reported')
                        }
                        state={storageState}
                        status={health?.storage ? 'Available' : 'Unknown'}
                    />
                </div>
            </View>

            <View UNSAFE_className={classes.detailSection}>
                <Heading level={3} UNSAFE_className={classes.sectionHeading}>
                    Connection
                </Heading>
                <dl className={classes.definitionList}>
                    <dt>Connection type</dt>
                    <dd>Direct trainer URL</dd>
                    <dt>Trainer URL</dt>
                    <dd>{remoteTrainer.url}</dd>
                    <dt>Device type</dt>
                    <dd>{types.join(', ') || 'Not reported'}</dd>
                    <dt>Available storage</dt>
                    <dd>{formatStorage(health?.storage) ?? 'Not reported'}</dd>
                    <dt>Health status</dt>
                    <dd>{healthLabel(health, isChecking)}</dd>
                    <dt>Added</dt>
                    <dd>
                        {remoteTrainer.created_at ? new Date(remoteTrainer.created_at).toLocaleString() : 'Unknown'}
                    </dd>
                    <dt>Last checked</dt>
                    <dd>{lastChecked}</dd>
                </dl>
            </View>
        </View>
    );
};

type RemoteTrainerAction =
    | { type: 'create' }
    | { type: 'edit'; remoteTrainer: SchemaRemoteTrainer }
    | { type: 'delete'; remoteTrainer: SchemaRemoteTrainer }
    | undefined;

const SelectTrainerButton = ({
    remoteTrainer,
    isSelected,
    onSelect,
}: {
    remoteTrainer: SchemaRemoteTrainer;
    isSelected: boolean;
    onSelect: () => void;
}) => {
    const { health, hasError, isChecking } = useRemoteTrainerHealth(remoteTrainer.id);
    const displayHealth = getDisplayHealth(remoteTrainer.id, health, hasError);

    return (
        <ActionButton
            aria-pressed={isSelected}
            onPress={onSelect}
            isQuiet
            UNSAFE_className={`${classes.trainerCard} ${isSelected ? classes.trainerCardSelected : ''}`}
        >
            <Flex direction='column' alignItems='start' gap='size-50'>
                <Flex width='100%' justifyContent='space-between' alignItems='center'>
                    <Text UNSAFE_className={classes.trainerName}>{remoteTrainer.name}</Text>
                    <StatusLight variant={healthVariant(displayHealth, isChecking)}>
                        {healthLabel(displayHealth, isChecking)}
                    </StatusLight>
                </Flex>
                <Flex gap='size-100' alignItems='center' wrap UNSAFE_className={classes.cardMeta}>
                    <Text UNSAFE_className={classes.connectionTag}>TRAINER URL</Text>
                    {deviceTypes(displayHealth).map((type) => (
                        <Text key={type} UNSAFE_className={deviceTagClass(type)}>
                            {type}
                        </Text>
                    ))}
                    <Text UNSAFE_className={classes.cardMetaText}>
                        {displayHealth?.devices?.[0]?.name ??
                            (isChecking ? 'Checking capability…' : 'Capability not reported')}
                    </Text>
                </Flex>
            </Flex>
        </ActionButton>
    );
};

const SelectedRemoteTrainerDetail = ({
    remoteTrainer,
    onEdit,
    onDelete,
}: {
    remoteTrainer: SchemaRemoteTrainer;
    onEdit: () => void;
    onDelete: () => void;
}) => {
    const { health, hasError, isChecking, checkHealth } = useRemoteTrainerHealth(remoteTrainer.id);
    const displayHealth = getDisplayHealth(remoteTrainer.id, health, hasError);

    return (
        <RemoteTrainerDetail
            remoteTrainer={remoteTrainer}
            health={displayHealth}
            isChecking={isChecking}
            onCheck={() => void checkHealth()}
            onEdit={onEdit}
            onDelete={onDelete}
        />
    );
};

export const RemoteTrainersPage = () => {
    const { data: remoteTrainers } = $api.useSuspenseQuery('get', '/api/remote-trainers');
    const [selectedRemoteTrainerId, setSelectedRemoteTrainerId] = useState<string | undefined>(remoteTrainers[0]?.id);
    const [action, setAction] = useState<RemoteTrainerAction>();
    const selectedRemoteTrainer =
        remoteTrainers.find((remoteTrainer) => remoteTrainer.id === selectedRemoteTrainerId) ?? remoteTrainers[0];

    return (
        <View padding='size-400' height='100%' maxWidth='240ch' marginX='auto'>
            <div className={classes.pageHeader}>
                <Heading level={1}>Remote Trainers</Heading>
                <Text>Configure and monitor trainer endpoints for remote training jobs.</Text>
            </div>

            <div className={classes.layout}>
                <View UNSAFE_className={classes.listPane}>
                    <Button
                        variant='accent'
                        width='100%'
                        UNSAFE_className={classes.addButton}
                        onPress={() => setAction({ type: 'create' })}
                    >
                        <Add />
                        New remote trainer
                    </Button>
                    {remoteTrainers.length === 0 ? (
                        <Text UNSAFE_className={classes.emptyList}>No remote trainers are configured.</Text>
                    ) : (
                        remoteTrainers.map((remoteTrainer) => (
                            <SelectTrainerButton
                                key={remoteTrainer.id}
                                remoteTrainer={remoteTrainer}
                                isSelected={remoteTrainer.id === selectedRemoteTrainer.id}
                                onSelect={() => setSelectedRemoteTrainerId(remoteTrainer.id)}
                            />
                        ))
                    )}
                </View>
                {selectedRemoteTrainer ? (
                    <SelectedRemoteTrainerDetail
                        remoteTrainer={selectedRemoteTrainer}
                        onEdit={() => setAction({ type: 'edit', remoteTrainer: selectedRemoteTrainer })}
                        onDelete={() => setAction({ type: 'delete', remoteTrainer: selectedRemoteTrainer })}
                    />
                ) : (
                    <View UNSAFE_className={classes.emptyDetail}>
                        <Heading level={2}>Select a remote trainer</Heading>
                        <Text>Choose a configured endpoint to view or edit its connection details.</Text>
                    </View>
                )}
            </div>
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
                        onDeleted={() => {
                            setAction(undefined);
                            setSelectedRemoteTrainerId(undefined);
                        }}
                    />
                )}
            </DialogContainer>
        </View>
    );
};
