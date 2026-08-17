import { Suspense, useEffect, useMemo, useRef } from 'react';

import { Content, Flex, Heading, IllustratedMessage, Loading, Text, View } from '@geti-ui/ui';
import { DockviewApi, IDockviewPanelProps } from 'dockview';
import { DockviewReact, DockviewReadyEvent, IDockviewReactProps } from 'dockview-react';

import { $api } from '../../../api/client';
import { physicalAiTheme } from '../../dockview';
import { useProjectId } from '../../projects/use-project';
import { ReactComponent as RobotIllustration } from './../../../assets/illustrations/INTEL_08_NO-TESTS.svg';
import { CameraCell } from './cells/camera-cell';
import { RobotCell } from './cells/robot-cell';
import { EnvironmentFormState, useEnvironmentForm } from './provider';

const EmptyPreview = () => {
    return (
        <IllustratedMessage>
            <RobotIllustration />

            <Flex direction='column' gap='size-200'>
                <Content>
                    <Text>
                        Choose the robots and cameras you&apos; like to add using the form on the left. After connecting
                        the robots and cameras, the preview will appear here.
                    </Text>
                </Content>
                <Heading>Setup your new environment</Heading>
            </Flex>
        </IllustratedMessage>
    );
};

const components = {
    follower: (
        props: IDockviewPanelProps<{
            title: string;
            follower_id: string;
            leader_id: string | undefined;
            camera_ids: string[];
        }>
    ) => {
        return (
            <RobotCell
                follower_id={props.params.follower_id}
                leader_id={props.params.leader_id}
                camera_ids={props.params.camera_ids}
            />
        );
    },
    camera: (props: IDockviewPanelProps<{ camera_id: string }>) => {
        return <CameraCell camera_id={props.params.camera_id} />;
    },
    default: (props: IDockviewPanelProps<{ title: string }>) => {
        return <div style={{ padding: '20px', color: 'white' }}>{props.params.title}</div>;
    },
} satisfies IDockviewReactProps['components'];

const buildDockviewPanels = (
    api: DockviewReadyEvent['api'],
    environment: EnvironmentFormState,
    cameraNameMap: Record<string, string>,
    cameraIds: string[]
) => {
    if (environment === null) {
        return api;
    }

    const panels = new Set<string>();

    environment.cameras.forEach(({ camera_id }) => {
        panels.add(camera_id);
        if (!api.panels.some((panel) => panel.id === camera_id)) {
            api.addPanel({
                id: camera_id,
                title: cameraNameMap[camera_id] ?? camera_id,
                component: 'camera',
                params: {
                    title: cameraNameMap[camera_id] ?? camera_id,
                    camera_id,
                },
                position: {
                    direction: 'right',
                    referencePanel: '',
                },
            });
        }
    });

    environment.robots.forEach((robot) => {
        const teleoperator_id = robot.teleoperator.type === 'robot' ? robot.teleoperator.robot_id : undefined;
        panels.add(robot.robot_id);
        // Existing follower panels keep the camera_ids they were created with.
        // Adding a camera here must not restart the live preview; session cameras
        // are not what this preview displays. A later client that needs the new
        // cameras (record) restarts from the owner.
        if (!api.panels.some((panel) => panel.id === robot.robot_id)) {
            api.addPanel({
                id: robot.robot_id,
                params: {
                    title: 'Follower',
                    follower_id: robot.robot_id,
                    leader_id: teleoperator_id,
                    camera_ids: cameraIds,
                },
                title: 'Follower',
                component: 'follower',

                position: {
                    direction: 'below',
                    referencePanel: '',
                },
            });
        }
    });

    // Remove any panels that are no longer part of the environment
    api.panels
        .filter((panel) => panels.has(panel.id) === false)
        .forEach((panel) => {
            api.removePanel(panel);
        });

    return api;
};

const ActualPreview = () => {
    const environment = useEnvironmentForm();
    const { project_id } = useProjectId();
    const api = useRef<DockviewApi>(null);

    const camerasQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras', {
        params: { path: { project_id } },
    });

    const cameraNameMap: Record<string, string> = useMemo(() => {
        const map: Record<string, string> = {};
        for (const camera of camerasQuery.data) {
            if (camera.id == undefined) {
                continue;
            }
            map[camera.id] = camera.name;
        }
        return map;
    }, [camerasQuery.data]);

    const cameraIds = useMemo(() => environment.cameras.map(({ camera_id }) => camera_id), [environment.cameras]);

    const onReady = (event: DockviewReadyEvent): void => {
        api.current = buildDockviewPanels(event.api, environment, cameraNameMap, cameraIds);
    };

    useEffect(() => {
        if (!api.current) {
            return;
        }

        buildDockviewPanels(api.current, environment, cameraNameMap, cameraIds);
    }, [environment, cameraNameMap, cameraIds]);

    return <DockviewReact onReady={onReady} components={components} theme={physicalAiTheme} />;
};

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

export const Preview = () => {
    const environment = useEnvironmentForm();

    const hasRobots = environment.robots.length > 0;
    const hasCameras = environment.cameras.length > 0;

    if (hasRobots || hasCameras) {
        return (
            <View height='100%'>
                <Suspense fallback={<CenteredLoading />}>
                    <ActualPreview />
                </Suspense>
            </View>
        );
    }

    return (
        <View padding={'size-400'} height='100%'>
            <View
                backgroundColor={'gray-200'}
                height={'100%'}
                maxHeight='100vh'
                padding={'size-200'}
                UNSAFE_style={{
                    borderRadius: 'var(--spectrum-alias-border-radius-regular)',
                    borderColor: 'var(--spectrum-global-color-gray-700)',
                    borderWidth: '1px',
                    borderStyle: 'dashed',
                }}
                position={'relative'}
            >
                <EmptyPreview />
            </View>
        </View>
    );
};
