import { ReactNode, Suspense, useEffect, useRef, useState } from 'react';

import { Button, Flex, Heading, Loading, ProgressCircle, Text, View } from '@geti-ui/ui';
import {
    DockviewApi,
    DockviewReact,
    DockviewReadyEvent,
    IDockviewPanelProps,
    IDockviewReactProps,
} from 'dockview-react';

import { $api } from '../../api/client';
import { SchemaSessionEnvironment } from '../../api/openapi-spec';
import { physicalAiTheme } from '../dockview';
import { useProjectId } from '../projects/use-project';
import { InlineAlert } from '../robots/setup-wizard/shared/inline-alert';
import { useRobotSimulationLaunch } from '../simulation/robot-simulation';
import { CameraCell } from './cells/camera-cell';
import { RobotCell } from './cells/robot-cell';
import { failureOf, useRuntimeSession } from './runtime-session-provider';
import { usePartControls } from './workspace/part-controls';

const components = {
    // The workspace's side bar has the Teleoperate switch.
    follower: (props: IDockviewPanelProps<{ robot_id: string }>) => (
        <RobotCell follower_id={props.params.robot_id} showTeleop={false} />
    ),
    camera: (props: IDockviewPanelProps<{ camera_id: string }>) => <CameraCell camera_id={props.params.camera_id} />,
} satisfies IDockviewReactProps['components'];

/** One panel per camera and one for the follower, keeping panels that are still part of the environment. */
const buildPanels = (api: DockviewApi, environment: SchemaSessionEnvironment) => {
    const panels = new Set<string>();

    environment.cameras.forEach((camera) => {
        panels.add(camera.camera_id);
        if (!api.panels.some((panel) => panel.id === camera.camera_id)) {
            api.addPanel({
                id: camera.camera_id,
                title: camera.name,
                component: 'camera',
                params: { camera_id: camera.camera_id },
                position: { direction: 'left', referencePanel: '' },
            });
        }
    });

    panels.add(environment.robot_id);
    if (!api.panels.some((panel) => panel.id === environment.robot_id)) {
        api.addPanel({
            id: environment.robot_id,
            title: 'Follower',
            component: 'follower',
            params: { robot_id: environment.robot_id },
            position: { direction: 'right', referencePanel: '' },
        });
    }

    api.panels.filter((panel) => !panels.has(panel.id)).forEach((panel) => api.removePanel(panel));
};

const SessionDock = ({ environment }: { environment: SchemaSessionEnvironment }) => {
    const api = useRef<DockviewApi>(null);

    const onReady = (event: DockviewReadyEvent) => {
        api.current = event.api;
        buildPanels(event.api, environment);
    };

    useEffect(() => {
        if (api.current) {
            buildPanels(api.current, environment);
        }
    }, [environment]);

    return (
        <View flex backgroundColor={'gray-200'} height={'100%'} minHeight={0} position={'relative'}>
            <DockviewReact onReady={onReady} components={components} theme={physicalAiTheme} />
        </View>
    );
};

/** Until the environment is up: what it is doing, or a way to load one when the session has none. */
const Initializing = ({ cancelHref }: { cancelHref: string }) => {
    const { state, error, statuses, loading, retry } = useRuntimeSession();
    const { pendingPart, act } = usePartControls();
    const failure = failureOf(statuses, 'environment');
    const isIdle = state !== null && state.environment == null && loading === null && pendingPart === null;

    return (
        <Flex width='100%' height={'100%'} alignItems={'center'} justifyContent={'center'} direction={'column'}>
            <Heading level={2}>
                <Text>{isIdle ? 'No environment loaded' : 'Initializing'}</Text>
                {!isIdle && error === null && failure === null && (
                    <ProgressCircle marginStart='size-200' size='S' isIndeterminate alignSelf={'center'} />
                )}
            </Heading>
            <Flex direction='column' margin='size-200' gap='size-100'>
                {failure !== null && <InlineAlert variant='error'>{failure}</InlineAlert>}
                {error !== null && <InlineAlert variant='error'>{error.message}</InlineAlert>}
            </Flex>
            <Flex gap='size-100'>
                <Button variant={'secondary'} href={cancelHref}>
                    Cancel
                </Button>
                {isIdle && error !== null && (
                    <Button variant='accent' onPress={retry}>
                        Try again
                    </Button>
                )}
                {isIdle && (
                    <Button
                        variant={error !== null ? 'secondary' : 'accent'}
                        onPress={() => act('environment', 'swap')}
                    >
                        Load environment…
                    </Button>
                )}
            </Flex>
        </Flex>
    );
};

/**
 * The dock once shown, and before that only for a simulated follower: its cell offers to start the
 * simulation and loads the environment again once it runs, so it shows while the environment
 * connects or has failed, not only once it is up.
 */
const UntilSimulatedFollower = ({
    follower_id,
    isShown,
    fallback,
    children,
}: {
    follower_id: string;
    isShown: boolean;
    fallback: ReactNode;
    children: ReactNode;
}) => {
    const { project_id } = useProjectId();
    // Not suspending: a swap to another follower keeps the dock while its robot loads.
    const { data: robot } = $api.useQuery(
        'get',
        '/api/projects/{project_id}/robots/{robot_id}',
        { params: { path: { project_id, robot_id: follower_id } } },
        { enabled: !isShown }
    );
    const isSimulated = useRobotSimulationLaunch(robot?.type ?? '') !== null;

    return isShown || isSimulated ? children : fallback;
};

/**
 * A session's robot and cameras in a dock, under an optional toolbar.
 *
 * The middle of the session workspace. Until the environment is loaded it shows what it is
 * waiting for instead; what a mode needs besides is the side bar's to say. Once it has shown
 * the dock, it keeps showing it while the environment fails or reloads.
 */
export const SessionViewer = ({ toolbar, cancelHref }: { toolbar?: ReactNode; cancelHref: string }) => {
    const { state, statuses } = useRuntimeSession();
    const environment = state?.environment;

    const isReady = environment != null && failureOf(statuses, 'environment') === null;
    const [hasStarted, setHasStarted] = useState(false);
    // Adjusted while rendering: started once ready, and over once the environment is unloaded.
    if (hasStarted !== (isReady || (hasStarted && environment != null))) {
        setHasStarted(!hasStarted);
    }

    if (environment == null) {
        return <Initializing cancelHref={cancelHref} />;
    }

    // One tree whether or not it waits for the follower, so the cells are not mounted again once it is up.
    return (
        <Suspense fallback={<Initializing cancelHref={cancelHref} />}>
            <UntilSimulatedFollower
                follower_id={environment.robot_id}
                isShown={isReady || hasStarted}
                fallback={<Initializing cancelHref={cancelHref} />}
            >
                <Flex direction={'column'} height={'100%'} gap='size-100'>
                    {toolbar}
                    {/* The cells suspend while they load; this keeps that inside the viewer. */}
                    <Suspense fallback={<Loading mode='inline' />}>
                        <SessionDock environment={environment} />
                    </Suspense>
                </Flex>
            </UntilSimulatedFollower>
        </Suspense>
    );
};
