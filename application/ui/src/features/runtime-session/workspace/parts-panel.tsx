import { ActionButton, Flex, Heading, Item, Menu, MenuTrigger, ProgressCircle, StatusLight, Text } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { SchemaSessionEnvironment } from '../../../api/openapi-spec';
import { failureOf, SessionPartName, Statuses, useRuntimeSession } from '../runtime-session-provider';
import { PartAction, usePartControls } from './part-controls';

const DEVICE_STATUS = { starting: 'notice', running: 'positive', failed: 'negative', stopped: 'neutral' } as const;

/** Each of the environment's devices, as it last reported. */
const DeviceStatuses = ({ statuses, environment }: { statuses: Statuses; environment: SchemaSessionEnvironment }) => {
    const label = (device: string) =>
        ({ follower: 'Follower', leader: 'Leader' })[device] ??
        environment.cameras.find((camera) => camera.key === device)?.name ??
        device;
    const devices = Object.values(statuses).filter(({ component }) => component.startsWith('environment/'));

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
    isLoaded,
    isPending,
    actions,
    onAction,
}: {
    label: string;
    name: string | undefined;
    detail?: string;
    error?: string | null;
    isLoaded: boolean;
    isPending: boolean;
    actions: PartAction[];
    onAction: (action: PartAction) => void;
}) => (
    <Flex alignItems='center' justifyContent='space-between' gap='size-100'>
        <Flex direction='column' minWidth={0}>
            <StatusLight variant={isPending ? 'notice' : error ? 'negative' : isLoaded ? 'positive' : 'neutral'}>
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
);

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
            <PartRow
                label='Environment'
                name={environment?.name}
                error={environmentFailure}
                isLoaded={environment != null}
                actions={loaded(environment != null)}
                {...row('environment')}
            />
            {environment != null && <DeviceStatuses statuses={statuses} environment={environment} />}
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
