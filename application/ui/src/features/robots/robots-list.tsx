import { Grid, StatusLight } from '@adobe/react-spectrum';
import { ActionButton, Flex, Heading, Item, Menu, MenuTrigger, View } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';
import { NavLink } from 'react-router-dom';

import { $api } from '../../api/client';
import { paths } from '../../router';
import { useProjectId } from '../projects/use-project';
import RobotArm from './../../assets/robot-arm.png';

const MenuActions = () => {
    return (
        <MenuTrigger>
            <ActionButton isQuiet UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }}>
                <MoreMenu />
            </ActionButton>
            <Menu>
                <Item>Edit</Item>
                <Item>Disconnect</Item>
                <Item>Delete</Item>
            </Menu>
        </MenuTrigger>
    );
};

const ConnectionStatus = ({ status }: { status: 'connected' | 'disconnected' }) => {
    return (
        <StatusLight
            variant={status === 'connected' ? 'positive' : 'negative'}
            UNSAFE_style={{
                background: 'var(--spectrum-global-color-gray-100)',
                borderRadius: '4px',
                paddingRight: '1em',
                scale: 0.7,
                transformOrigin: 'top right',
            }}
        >
            {status === 'connected' ? 'Connected' : 'Disconnected'}
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
            backgroundColor={'gray-50'}
            padding='size-200'
            UNSAFE_style={{
                border: `1px solid ${isActive ? 'var(--energy-blue)' : 'var(--spectrum-global-color-gray-200)'}`,
            }}
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
                        <MenuActions />
                    </View>
                </Flex>
            </Flex>
        </View>
    );
};

export const RobotsList = () => {
    const { project_id } = useProjectId();
    const { data: robots } = $api.useSuspenseQuery('get', '/api/hardware/robots');

    return (
        <Flex direction='column' gap='size-100'>
            {(robots ?? []).map((robot) => {
                const hardwareRobot = robot;

                const to = paths.project.robotConfiguration.show({
                    project_id,
                    robot_id: robot.serial_id,
                });

                return (
                    <NavLink key={robot.serial_id} to={to}>
                        {({ isActive }) => {
                            return (
                                <RobotListItem
                                    id={robot.serial_id}
                                    name={hardwareRobot?.device_name ?? ''}
                                    port={hardwareRobot?.port}
                                    // TODO configure using some project endpoint
                                    type={'Leader'}
                                    // Fake connected mode for now
                                    status={hardwareRobot?.port === '/dev/ttyACM1' ? 'connected' : 'disconnected'}
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
