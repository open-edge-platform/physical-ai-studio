import { Button, ButtonGroup, Flex, Item, TabList, TabPanels, Tabs } from '@geti/ui';
import { Outlet, useLocation, useParams } from 'react-router-dom';

import { paths } from '../../router';
import { getPathSegment } from '../../utils';

export const Robot = () => {
    const { pathname } = useLocation();
    const params = useParams<{ project_id: string; robot_id: string }>() as {
        project_id: string;
        robot_id: string;
    };

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
                        <Button variant='secondary'>Identify</Button>
                        <Button variant='secondary'>Connect</Button>
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
