import { Grid, StatusLight } from '@adobe/react-spectrum';
import { ActionButton, Button, Flex, Heading, Item, Menu, MenuTrigger, View } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';
import { NavLink, useParams } from 'react-router-dom';

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

const RobotListItem = ({ id, name, status }: { id: string; name: string; status: 'connected' | 'disconnected' }) => {
    const { project_id } = useProjectId();
    return (
        <NavLink to={paths.project.robotConfiguration.show({ project_id: project_id, robot_id: id })}>
            {({ isActive, isPending, isTransitioning }) => {
                return (
                    <View
                        backgroundColor={'gray-50'}
                        padding='size-200'
                        UNSAFE_style={
                            isActive
                                ? {
                                      border: '1px solid var(--energy-blue)',
                                  }
                                : {
                                      border: '1px solid var(--spectrum-global-color-gray-200)',
                                  }
                        }
                    >
                        <Flex justifyContent={'space-between'} direction='column' gap='size-100'>
                            <Grid
                                areas={['icon name menu', 'icon type menu']}
                                columns={['auto', '1fr']}
                                columnGap={'size-100'}
                            >
                                <View gridArea={'icon'} padding='size-100'>
                                    <img src={RobotArm} style={{ maxWidth: '32px' }} />
                                </View>
                                <Heading level={2} gridArea='name' UNSAFE_style={{ color: 'var(--energy-blue)' }}>
                                    {name}
                                </Heading>
                                <View
                                    gridArea='type'
                                    UNSAFE_style={{
                                        fontSize: '14px',
                                    }}
                                >
                                    {name} Type_SO101_Leader
                                </View>
                                <View gridArea='menu'>
                                    <MenuActions />
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
                                            <pre style={{ margin: 0, display: 'inline' }}>tty.usb.modem.5787345555</pre>
                                        </li>
                                        <li style={{ marginLeft: 'var(--spectrum-global-dimension-size-200)' }}>
                                            ID:{' '}
                                            <pre style={{ margin: 0, display: 'inline' }}>Robot_id_set_by_the_user</pre>
                                        </li>
                                        <li style={{ marginLeft: 'var(--spectrum-global-dimension-size-200)' }}>
                                            Calibrated: 09/02/2025 7:34 PM
                                        </li>
                                    </ul>
                                </View>
                                <View alignSelf={'end'}>
                                    {status === 'connected' ? (
                                        <StatusLight
                                            variant='positive'
                                            UNSAFE_style={{
                                                background: 'var(--spectrum-global-color-gray-100)',
                                                borderRadius: '4px',
                                                paddingRight: '1em',
                                                scale: 0.7,
                                                transformOrigin: 'bottom right',
                                            }}
                                        >
                                            Connected
                                        </StatusLight>
                                    ) : (
                                        <StatusLight
                                            variant='negative'
                                            UNSAFE_style={{
                                                background: 'var(--spectrum-global-color-gray-100)',
                                                borderRadius: '4px',
                                                paddingRight: '1em',
                                                scale: 0.7,
                                                transformOrigin: 'bottom right',
                                            }}
                                        >
                                            Disconnected
                                        </StatusLight>
                                    )}
                                </View>
                            </Flex>
                        </Flex>
                    </View>
                );
            }}
        </NavLink>
    );
};

export const RobotsList = () => {
    const { project_id } = useProjectId();
    const { data: robots } = $api.useSuspenseQuery('get', '/api/hardware/robots');
    const { data: cameras } = $api.useSuspenseQuery('get', '/api/hardware/cameras');
    const { data: calibrations } = $api.useSuspenseQuery('get', '/api/hardware/calibrations');

    console.log({ robots, cameras, calibrations });

    return (
        <Flex direction={'column'} gap='size-200'>
            <Flex justifyContent={'space-between'}>
                <Heading level={4} marginY='size-100'>
                    Robot list
                </Heading>

                <Button href={paths.project.robotConfiguration.new({ project_id: project_id })} variant='accent'>
                    Add
                </Button>
            </Flex>

            <Flex direction='column' gap='size-100'>
                <RobotListItem id='0' name='Follower robot' status='connected' />
                <RobotListItem id='1' name='Follower robot' status='disconnected' />
                <RobotListItem id='2' name='Leader robot' status='disconnected' />
            </Flex>
        </Flex>
    );
};
