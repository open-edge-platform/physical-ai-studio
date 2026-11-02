import { createContext, RefObject, Suspense, useContext, useEffect, useRef } from 'react';

import { Flex, Loading, View } from '@geti-ui/ui';
import {
    DockviewApi,
    DockviewReact,
    DockviewReadyEvent,
    IDockviewPanelProps,
    IDockviewReactProps,
} from 'dockview-react';

import { SchemaEnvironmentWithRelations } from '../../../api/openapi-spec';
import { physicalAiTheme } from '../../dockview';
import { CameraCell } from './camera-cell.component';
import { RobotCell } from './robot-cell.component';

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

interface RobotViewContextValue {
    environment: SchemaEnvironmentWithRelations;
    joints: RefObject<Record<string, number> | undefined>;
}

const RobotViewContext = createContext<RobotViewContextValue | null>(null);

const FollowerPanel = (props: IDockviewPanelProps<{ title: string; robot_id: string }>) => {
    const view = useContext(RobotViewContext);
    if (view === null) {
        return null;
    }
    return <RobotCell robot_id={props.params.robot_id} environment={view.environment} joints={view.joints} />;
};

const components = {
    follower: FollowerPanel,
    camera: (props: IDockviewPanelProps<{ camera_id: string }>) => {
        return <CameraCell camera_id={props.params.camera_id} />;
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

    api.panels
        .filter((panel) => panels.has(panel.id) === false)
        .forEach((panel) => {
            api.removePanel(panel);
        });

    return api;
};

export const RobotControlView = ({
    environment,
    isReady,
    joints,
}: {
    environment: SchemaEnvironmentWithRelations;
    isReady: boolean;
    joints: RefObject<Record<string, number> | undefined>;
}) => {
    const api = useRef<DockviewApi>(null);

    const onReady = (event: DockviewReadyEvent): void => {
        api.current = buildDockviewPanels(event.api, environment);
    };

    useEffect(() => {
        if (!api.current || !isReady) {
            return;
        }

        buildDockviewPanels(api.current, environment);
    }, [environment, isReady]);

    return (
        <RobotViewContext.Provider value={{ environment, joints }}>
            <View flex>
                <View backgroundColor={'gray-200'} height={'100%'} maxHeight='100vh' position={'relative'}>
                    <Suspense fallback={<CenteredLoading />}>
                        <DockviewReact onReady={onReady} components={components} theme={physicalAiTheme} />
                    </Suspense>
                </View>
            </View>
        </RobotViewContext.Provider>
    );
};
