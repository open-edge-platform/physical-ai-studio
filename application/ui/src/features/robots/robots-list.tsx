import { Grid, StatusLight } from '@adobe/react-spectrum';
import { ActionButton, Button, Flex, Heading, Icon, Item, Menu, MenuTrigger, View } from '@geti/ui';
import { Add, MoreMenu } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { NavLink } from 'react-router-dom';

import { $api } from '../../api/client';
import { paths } from '../../router';
import { useProjectId } from '../projects/use-project';
import RobotArm from './../../assets/robot-arm.png';

import classes from './robots-list.module.scss';

const MenuActions = ({ robot_id }: { robot_id: string }) => {
    const { project_id } = useProjectId();
    const deleteRobotMutation = $api.useMutation('delete', '/api/projects/{project_id}/robots/{robot_id}');

    return (
        <MenuTrigger>
            <ActionButton isQuiet UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }}>
                <MoreMenu />
            </ActionButton>
            <Menu
                selectionMode='single'
                onAction={(action) => {
                    if (action === 'delete') {
                        deleteRobotMutation.mutate({ params: { path: { project_id, robot_id } } });
                    }
                }}
            >
                <Item href={paths.project.robots.edit({ project_id, robot_id })}>Edit</Item>
                <Item key='delete'>Delete</Item>
            </Menu>
        </MenuTrigger>
    );
};

export const ConnectionStatus = ({ status }: { status: 'connected' | 'disconnected' }) => {
    return (
        <StatusLight
            variant={status === 'connected' ? 'positive' : 'negative'}
            UNSAFE_className={classes.connectionStatus}
        >
            {status === 'connected' ? 'Online' : 'Offline'}
        </StatusLight>
    );
};

const RobotListItem = ({
    id,
    name,
    status,
    isActive,
    type,
    port,
}: {
    id: string;
    name: string;
    type: string;
    status: 'connected' | 'disconnected';
    port: string | undefined;
    isActive: boolean;
}) => {
    return (
        <View
            padding='size-200'
            UNSAFE_className={clsx({
                [classes.robotListItem]: true,
                [classes.robotListItemActive]: isActive,
            })}
        >
            <Flex justifyContent={'space-between'} direction='column' gap='size-100'>
                <Grid areas={['icon name status', 'icon type status']} columns={['auto', '1fr']} columnGap={'size-100'}>
                    <View gridArea={'icon'} padding='size-100'>
                        <img src={RobotArm} style={{ maxWidth: '32px' }} alt='Robot arm icon' />
                    </View>
                    <Heading level={2} gridArea='name' UNSAFE_style={isActive ? { color: 'var(--energy-blue)' } : {}}>
                        {name}
                    </Heading>
                    <View gridArea='type' UNSAFE_style={{ fontSize: '14px' }}>
                        {type}
                    </View>
                    <View gridArea='status'>
                        <ConnectionStatus status={status} />
                    </View>
                </Grid>
                <Flex direction={'row'} justifyContent={'space-between'}>
                    <View>
                        <ul
                            style={{
                                display: 'flex',
                                flexDirection: 'column',
                                gap: 'var(--spectrum-global-dimension-size-10)',
                                listStyleType: 'disc',
                                fontSize: '10px',
                            }}
                        >
                            <li style={{ marginLeft: 'var(--spectrum-global-dimension-size-200)' }}>
                                Port:{' '}
                                <pre style={{ margin: 0, display: 'inline' }}>
                                    {port === undefined ? 'Unknown' : port}
                                </pre>
                            </li>
                            <li style={{ marginLeft: 'var(--spectrum-global-dimension-size-200)' }}>
                                ID: <pre style={{ margin: 0, display: 'inline' }}>{id}</pre>
                            </li>
                            <li style={{ marginLeft: 'var(--spectrum-global-dimension-size-200)' }}>
                                Calibrated: 09/02/2025 7:34 PM
                            </li>
                        </ul>
                    </View>
                    <View alignSelf={'end'}>
                        <MenuActions robot_id={id} />
                    </View>
                </Flex>
            </Flex>
        </View>
    );
};

export const RobotsList = () => {
    const { project_id } = useProjectId();
    const { data: robots } = $api.useSuspenseQuery('get', '/api/hardware/robots');
    const { data: projectRobots } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots', {
        params: { path: { project_id } },
    });

    return (
        <Flex direction='column' gap='size-100'>
            <Button
                variant='secondary'
                href={paths.project.robots.new({ project_id })}
                UNSAFE_className={classes.addNewRobotButton}
            >
                <Icon marginEnd='size-50'>
                    <Add />
                </Icon>
                Add new robot
            </Button>

            {projectRobots.map((robot) => {
                const hardwareRobot = robots.find((hardware) => {
                    return hardware.serial_id === robot.serial_id;
                });

                const to = paths.project.robots.show({ project_id, robot_id: robot.id });

                return (
                    <NavLink key={robot.id} to={to}>
                        {({ isActive }) => {
                            return (
                                <RobotListItem
                                    id={robot.id}
                                    name={robot.name}
                                    port={hardwareRobot?.port}
                                    type={robot.type}
                                    status={hardwareRobot !== undefined ? 'connected' : 'disconnected'}
                                    isActive={isActive}
                                />
                            );
                        }}
                    </NavLink>
                );
            })}
        </Flex>
    );
};
