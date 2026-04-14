import { Suspense, useEffect, useRef } from 'react';

import { Flex, Loading, View } from '@geti-ui/ui';
import {
    DockviewApi,
    DockviewReact,
    DockviewReadyEvent,
    IDockviewPanelProps,
    IDockviewReactProps,
} from 'dockview-react';

import { SchemaEnvironmentWithRelations } from '../../../api/openapi-spec';
import { useRobotControl } from '../robot-control-provider';
import { CameraCell } from './camera-cell.component';
import { RobotCell } from './robot-cell.component';

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

const components = {
    follower: (props: IDockviewPanelProps<{ title: string; robot_id: string }>) => {
        return <RobotCell robot_id={props.params.robot_id} />;
    },
    camera: (props: IDockviewPanelProps<{ camera_id: string; camera_name: string }>) => {
        return <CameraCell camera_id={props.params.camera_id} camera_name={props.params.camera_name} />;
    },
    default: (props: IDockviewPanelProps<{ title: string }>) => {
        return <div style={{ padding: '20px', color: 'white' }}>{props.params.title}</div>;
    },
} satisfies IDockviewReactProps['components'];

const buildDockviewPanels = (api: DockviewReadyEvent['api'], environment: SchemaEnvironmentWithRelations) => {
    if (environment === null) {
        return api;
    }

    const panels = new Set<string>();

    environment.cameras?.forEach((camera) => {
        if (camera.id == undefined) {
            return;
        }
        panels.add(camera.id);
        if (!api.panels.some((panel) => panel.id === camera.id)) {
            api.addPanel({
                id: camera.id,
                title: camera.name,
                component: 'camera',
                params: {
                    title: camera.name,
                    camera_id: camera.id,
                    camera_name: camera.name,
                },
                position: {
                    direction: 'left',
                    referencePanel: '',
                },
            });
        }
    });

    environment.robots?.forEach((robot) => {
        panels.add(robot.robot.id);
        if (!api.panels.some((panel) => panel.id === robot.robot.id)) {
            api.addPanel({
                id: robot.robot.id,
                params: { title: 'Follower', robot_id: robot.robot.id },
                title: 'Follower',
                component: 'follower',

                position: {
                    direction: 'right',
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

export const RobotControlView = () => {
    const { environment, state } = useRobotControl();
    const api = useRef<DockviewApi>(null);

    const onReady = (event: DockviewReadyEvent): void => {
        api.current = buildDockviewPanels(event.api, environment);
    };

    useEffect(() => {
        if (!api.current || !state.environment_loaded) {
            return;
        }

        buildDockviewPanels(api.current, environment);
    }, [environment, state.environment_loaded]);

    return (
        <View flex>
            <View backgroundColor={'gray-200'} height={'100%'} maxHeight='100vh' position={'relative'}>
                <Suspense fallback={<CenteredLoading />}>
                    <DockviewReact onReady={onReady} components={components} />
                </Suspense>
            </View>
        </View>
    );
};
