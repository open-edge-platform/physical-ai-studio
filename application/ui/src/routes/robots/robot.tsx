import { Button, ButtonGroup, Flex, Item, TabList, TabPanels, Tabs } from '@geti/ui';
import { Outlet, useLocation, useParams } from 'react-router-dom';

import { $api } from '../../api/client';
import { paths } from '../../router';
import { getPathSegment } from '../../utils';

const useRobot = () => {
    const params = useParams<{ project_id: string; robot_id: string }>() as {
        project_id: string;
        robot_id: string;
    };

    const robots = $api.useSuspenseQuery('get', '/api/hardware/robots');

    return robots.data.find((robot) => robot.serial_id === params.robot_id);
};

export const Robot = () => {
    const { pathname } = useLocation();
    const params = useParams<{ project_id: string; robot_id: string }>() as {
        project_id: string;
        robot_id: string;
    };
    const robot = useRobot();
    const identifyRobotMutation = $api.useMutation('put', '/api/hardware/identify');
    const moveJointMutation = $api.useMutation('put', '/api/hardware/robots/{robot_id}/joints/{joint}');

    return (
        <Tabs aria-label='Robot configuration navigation' selectedKey={getPathSegment(pathname, 5)} height='100%'>
            <Flex>
                <TabList
                    width='100%'
                    UNSAFE_style={{
                        '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)',
                    }}
                >
                    <Item
                        key={paths.project.robotConfiguration.controller(params)}
                        href={paths.project.robotConfiguration.controller(params)}
                    >
                        Robot controller
                    </Item>
                    <Item
                        key={paths.project.robotConfiguration.calibration(params)}
                        href={paths.project.robotConfiguration.calibration(params)}
                    >
                        Calibration
                    </Item>
                    <Item
                        key={paths.project.robotConfiguration.setupMotors(params)}
                        href={paths.project.robotConfiguration.setupMotors(params)}
                    >
                        Setup motors
                    </Item>
                </TabList>
                <div
                    style={{
                        display: 'flex',
                        flex: '0 0 auto',
                        borderBottom:
                            'var(--spectrum-alias-border-size-thick) solid var(--spectrum-global-color-gray-300)',
                    }}
                >
                    <ButtonGroup>
                        <Button
                            isHidden
                            variant='secondary'
                            onPress={() => {
                                if (robot === undefined) {
                                    return;
                                }

                                moveJointMutation.mutate({
                                    params: {
                                        path: {
                                            joint: 'gripper',
                                        },
                                        query: {
                                            joint_value: 2800,
                                        },
                                    },
                                    body: {
                                        device_name: robot.device_name,
                                        port: robot.port,
                                        serial_id: robot.serial_id,
                                    },
                                });
                            }}
                            isPending={moveJointMutation.isPending}
                        >
                            Test
                        </Button>
                        <Button
                            variant='secondary'
                            onPress={() => {
                                if (robot === undefined) {
                                    return;
                                }

                                identifyRobotMutation.mutate({
                                    body: {
                                        device_name: robot.device_name,
                                        port: robot.port,
                                        serial_id: robot.serial_id,
                                    },
                                });
                            }}
                            isPending={identifyRobotMutation.isPending}
                        >
                            Identify
                        </Button>
                        <Button variant='secondary'>Connect</Button>
                        <Button variant='secondary' isHidden>
                            Edit
                        </Button>
                    </ButtonGroup>
                </div>
            </Flex>
            <TabPanels>
                <Item key={paths.project.robotConfiguration.controller(params)}>
                    <Outlet />
                </Item>
                <Item key={paths.project.robotConfiguration.calibration(params)}>
                    <Outlet />
                </Item>
                <Item key={paths.project.robotConfiguration.setupMotors(params)}>
                    <Outlet />
                </Item>
            </TabPanels>
        </Tabs>
    );
};
