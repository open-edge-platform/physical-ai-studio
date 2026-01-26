import { Suspense, useRef } from 'react';

import { Content, Flex, Heading, IllustratedMessage, Loading, Text, View } from '@geti/ui';
import { DockviewApi, IDockviewPanelProps } from 'dockview';
import { DockviewReact, DockviewReadyEvent, IDockviewReactProps } from 'dockview-react';

import { ReactComponent as RobotIllustration } from './../../../assets/illustrations/INTEL_08_NO-TESTS.svg';
import { CameraCell } from './cells/camera-cell';
import { RobotCell } from './cells/robot-cell';
import { useEnvironmentForm } from './provider';

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
    leader: (props: IDockviewPanelProps<{ title: string; robot_id: string }>) => {
        return <RobotCell robot_id={props.params.robot_id} />;
    },
    follower: (props: IDockviewPanelProps<{ title: string; robot_id: string }>) => {
        return <RobotCell robot_id={props.params.robot_id} />;
    },
    camera: (props: IDockviewPanelProps<{ camera_id: string }>) => {
        return <CameraCell camera_id={props.params.camera_id} />;
    },
    default: (props: IDockviewPanelProps<{ title: string }>) => {
        return <div style={{ padding: '20px', color: 'white' }}>{props.params.title}</div>;
    },
} satisfies IDockviewReactProps['components'];

const ActualPreview = () => {
    const environment = useEnvironmentForm();
    const api = useRef<DockviewApi>(null);

    const onReady = (event: DockviewReadyEvent): void => {
        environment.camera_ids.forEach((camera_id, idx) => {
            event.api.addPanel({
                id: camera_id,
                component: 'camera',
                params: {
                    title: `Camera ${idx}`,
                    camera_id,
                },
                position: {
                    direction: 'right',
                    referencePanel: '',
                },
            });
        });

        environment.robots.forEach((robot) => {
            event.api.addPanel({
                id: robot.robot_id,
                params: { title: 'Follower', robot_id: robot.robot_id },
                component: 'follower',

                position: {
                    direction: 'below',
                    referencePanel: '',
                },
            });

            if (robot.teleoperator.type === 'robot') {
                event.api.addPanel({
                    id: robot.teleoperator.robot_id,
                    params: { title: 'Leader', robot_id: robot.teleoperator.robot_id },
                    component: 'leader',

                    position: {
                        direction: 'right',
                        referencePanel: robot.robot_id,
                    },
                });
            }
        });

        api.current = event.api;
    };

    return <DockviewReact onReady={onReady} components={components} />;
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
    const hasCameras = environment.camera_ids.length > 0;

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
