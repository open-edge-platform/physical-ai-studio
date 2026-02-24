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

export const ConnectionStatus = ({ status }: { status: 'online' | 'offline' | 'unknown' }) => {
    const Capitalize = (str: string) => {
        return str.charAt(0).toUpperCase() + str.slice(1);
    };

    return (
        <StatusLight
            variant={status === 'online' ? 'positive' : status == 'unknown' ? 'notice' : 'negative'}
            UNSAFE_className={classes.connectionStatus}
        >
            {status === 'unknown' ? <View>Loading...</View> : <View>{Capitalize(status)}</View>}
        </StatusLight>
    );
};

const RobotListItem = ({
    id,
    name,
    status,
    isActive,
    type,
    connectionString,
    serialNumber,
}: {
    id: string;
    name: string;
    type: string;
    status: 'online' | 'offline' | 'unknown';
    connectionString: string | undefined;
    serialNumber: string | undefined;
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
                        {type.replaceAll('_', ' ')}
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
                            {connectionString !== undefined && connectionString !== '' ? (
                                <li style={{ marginLeft: 'var(--spectrum-global-dimension-size-200)' }}>
                                    Connection string:{' '}
                                    <pre style={{ margin: 0, display: 'inline' }}>{connectionString}</pre>
                                </li>
                            ) : null}

                            {serialNumber !== undefined && serialNumber !== '' ? (
                                <li style={{ marginLeft: 'var(--spectrum-global-dimension-size-200)' }}>
                                    Serial number: <pre style={{ margin: 0, display: 'inline' }}>{serialNumber}</pre>
                                </li>
                            ) : null}
                            <li style={{ marginLeft: 'var(--spectrum-global-dimension-size-200)' }}>
                                ID: <pre style={{ margin: 0, display: 'inline' }}>{id}</pre>
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
    const { data: projectRobots } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots', {
        params: { path: { project_id } },
    });

    const { data: onlineProjectRobots } = $api.useQuery('get', '/api/projects/{project_id}/robots/online', {
        params: { path: { project_id } },
        suspense: false,
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
                const onlineRobot = onlineProjectRobots?.find((r) => r.id === robot.id);

                const status = onlineRobot?.connection_status ?? 'unknown';

                const to = paths.project.robots.show({
                    project_id,
                    robot_id: robot.id,
                });

                return (
                    <NavLink key={robot.id} to={to}>
                        {({ isActive }) => {
                            return (
                                <RobotListItem
                                    id={robot.id}
                                    name={robot.name}
                                    connectionString={robot.connection_string}
                                    serialNumber={robot.serial_number}
                                    type={robot.type}
                                    status={onlineProjectRobots === undefined ? 'unknown' : status}
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
