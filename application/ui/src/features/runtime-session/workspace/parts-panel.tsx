import { ReactNode, useState } from 'react';

import { ActionButton, Flex, Heading, Item, Menu, MenuTrigger, ProgressCircle, StatusLight, Text } from '@geti-ui/ui';
import { ChevronDownLight, ChevronRightSmallLight, MoreMenu } from '@geti-ui/ui/icons';

import { SchemaSessionEnvironment, SchemaStatusMessage } from '../../../api/openapi-spec';
import { failureOf, SessionPartName, Statuses, useRuntimeSession } from '../runtime-session-provider';
import { PartAction, usePartControls } from './part-controls';

const DEVICE_STATUS = { starting: 'notice', running: 'positive', failed: 'negative', stopped: 'neutral' } as const;

/** Each of the environment's devices, as it last reported. */
const DeviceStatuses = ({
    devices,
    environment,
}: {
    devices: SchemaStatusMessage[];
    environment: SchemaSessionEnvironment;
}) => {
    const label = (device: string) =>
        ({ follower: 'Follower', leader: 'Leader' })[device] ??
        environment.cameras.find((camera) => camera.key === device)?.name ??
        device;

    return (
        <Flex direction='column' marginStart='size-300'>
            {devices.map(({ component, state, error }) => (
                <Flex key={component} direction='column'>
                    <StatusLight variant={DEVICE_STATUS[state]}>
                        {label(component.slice('environment/'.length))}
                    </StatusLight>
                    {error && (
                        <Text marginStart='size-300' UNSAFE_style={{ color: 'var(--spectrum-negative-visual-color)' }}>
                            {error}
                        </Text>
                    )}
                </Flex>
            ))}
        </Flex>
    );
};

const PartRow = ({
    label,
    name,
    detail,
    error,
    light,
    isLoaded,
    isPending,
    actions,
    onAction,
    expander,
}: {
    label: string;
    name: string | undefined;
    detail?: string;
    error?: string | null;
    /** The light, when more than this part's own state decides it. */
    light?: 'notice' | 'negative';
    isLoaded: boolean;
    isPending: boolean;
    actions: PartAction[];
    onAction: (action: PartAction) => void;
    expander?: ReactNode;
}) => (
    <Flex alignItems='center' justifyContent='space-between' gap='size-100'>
        <Flex direction='column' minWidth={0}>
            <StatusLight
                variant={isPending ? 'notice' : error ? 'negative' : (light ?? (isLoaded ? 'positive' : 'neutral'))}
            >
                {label}
            </StatusLight>
            <Text marginStart='size-300' UNSAFE_style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {isPending ? 'Loading…' : (name ?? 'Not loaded')}
                {detail && !isPending ? ` · ${detail}` : ''}
            </Text>
            {error && !isPending && (
                <Text marginStart='size-300' UNSAFE_style={{ color: 'var(--spectrum-negative-visual-color)' }}>
                    {error}
                </Text>
            )}
        </Flex>
        <Flex alignItems='center' flexShrink={0}>
            {expander}
            {isPending ? (
                <ProgressCircle size='S' isIndeterminate aria-label={`Loading ${label}`} />
            ) : (
                <MenuTrigger>
                    <ActionButton isQuiet aria-label={`${label} actions`}>
                        <MoreMenu />
                    </ActionButton>
                    <Menu onAction={(key) => onAction(key as PartAction)}>
                        {actions.map((action) => (
                            <Item key={action}>
                                {action === 'reload'
                                    ? 'Reload'
                                    : action === 'swap'
                                      ? isLoaded
                                          ? 'Swap…'
                                          : 'Load…'
                                      : 'Unload'}
                            </Item>
                        ))}
                    </Menu>
                </MenuTrigger>
            )}
        </Flex>
    </Flex>
);

/** The environment as one light, its devices' worst state; expand it for each device and why it failed. */
const EnvironmentRow = ({
    environment,
    statuses,
    actions,
    isPending,
    onAction,
}: {
    environment: SchemaSessionEnvironment | null | undefined;
    statuses: Statuses;
    actions: PartAction[];
    isPending: boolean;
    onAction: (action: PartAction) => void;
}) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const devices = Object.values(statuses).filter(({ component }) => component.startsWith('environment/'));
    const failed = devices.filter(({ state }) => state === 'failed').length;
    const isStarting = devices.some(({ state }) => state === 'starting');
    const canExpand = environment != null && devices.length > 0;

    return (
        <>
            <PartRow
                label='Environment'
                name={environment?.name}
                detail={failed > 0 ? `${failed} device${failed === 1 ? '' : 's'} failed` : undefined}
                error={failureOf(statuses, 'environment')}
                light={failed > 0 ? 'negative' : isStarting ? 'notice' : undefined}
                isLoaded={environment != null}
                isPending={isPending}
                actions={actions}
                onAction={onAction}
                expander={
                    canExpand && (
                        <ActionButton
                            isQuiet
                            aria-label={isExpanded ? 'Hide devices' : 'Show devices'}
                            aria-expanded={isExpanded}
                            onPress={() => setIsExpanded(!isExpanded)}
                        >
                            {isExpanded ? <ChevronDownLight /> : <ChevronRightSmallLight />}
                        </ActionButton>
                    )
                }
            />
            {canExpand && isExpanded && <DeviceStatuses devices={devices} environment={environment} />}
        </>
    );
};

/** What the session has loaded, each part with its status and reload, swap and unload. */
export const PartsPanel = () => {
    const { state, statuses } = useRuntimeSession();
    const { pendingPart, act } = usePartControls();

    const environment = state?.environment;
    const dataset = state?.dataset;
    const model = state?.model;
    const environmentFailure = failureOf(statuses, 'environment');

    const loaded = (isLoaded: boolean): PartAction[] => (isLoaded ? ['reload', 'swap', 'unload'] : ['swap']);

    /** A dataset or model is kept without a working environment, but loading one needs it. */
    const partActions = (isLoaded: boolean): PartAction[] => {
        if (environment == null || environmentFailure !== null) {
            return isLoaded ? ['unload'] : [];
        }
        return loaded(isLoaded);
    };

    const row = (part: SessionPartName) => ({
        isPending: pendingPart === part,
        onAction: (action: PartAction) => act(part, action),
    });

    return (
        <Flex direction='column' gap='size-150'>
            <Heading level={4} margin={0}>
                State
            </Heading>
            <EnvironmentRow
                environment={environment}
                statuses={statuses}
                actions={loaded(environment != null)}
                {...row('environment')}
            />
            <PartRow
                label='Dataset'
                name={dataset?.name}
                detail={dataset?.episodes_recorded ? `${dataset.episodes_recorded} recorded this session` : undefined}
                error={failureOf(statuses, 'dataset')}
                isLoaded={dataset != null}
                actions={partActions(dataset != null)}
                {...row('dataset')}
            />
            <PartRow
                label='Model'
                name={model?.name}
                detail={model ? `${model.backend} · ${model.device}` : undefined}
                error={failureOf(statuses, 'model')}
                isLoaded={model != null}
                actions={partActions(model != null)}
                {...row('model')}
            />
        </Flex>
    );
};
