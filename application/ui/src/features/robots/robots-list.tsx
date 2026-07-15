import { Grid, StatusLight } from '@adobe/react-spectrum';
import { ActionButton, Button, Flex, Heading, Icon, Item, Menu, MenuTrigger, toast, View } from '@geti-ui/ui';
import { Add, MoreMenu } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';
import { NavLink } from 'react-router-dom';

import { $api, fetchClient } from '../../api/client';
import { getApiErrorMessage, isResourceInUseError } from '../../api/errors';
import { paths } from '../../router';
import { useProjectId } from '../projects/use-project';
import RobotArm from './../../assets/robot-arm.png';
import { SchemaRobot } from './robot-types';

import classes from './robots-list.module.css';

const exportCalibration = async (project_id: string, robot: SchemaRobot) => {
    if (robot.active_calibration_id === null || robot.active_calibration_id === undefined) {
        return;
    }

    const { data, error } = await fetchClient.GET(
        '/api/projects/{project_id}/robots/{robot_id}/calibrations/{calibration_id}',
        { params: { path: { project_id, robot_id: robot.id, calibration_id: robot.active_calibration_id } } }
    );

    if (error || !data) {
        toast.negative(getApiErrorMessage(error) ?? 'Failed to export calibration.');
        return;
    }

    const calibration = Object.fromEntries(
        Object.entries(data.values).map(([jointName, value]) => [
            jointName,
            {
                id: value.id,
                drive_mode: value.drive_mode,
                homing_offset: value.homing_offset,
                range_min: value.range_min,
                range_max: value.range_max,
            },
        ])
    );
    const downloadUrl = URL.createObjectURL(
        new Blob([JSON.stringify(calibration, null, 4)], { type: 'application/json' })
    );
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `${robot.name}-calibration.json`;
    link.click();
    URL.revokeObjectURL(downloadUrl);
};

const MenuActions = ({ robot }: { robot: SchemaRobot }) => {
    const { project_id } = useProjectId();
    const deleteRobotMutation = $api.useMutation('delete', '/api/projects/{project_id}/robots/{robot_id}', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/robots', { params: { path: { project_id } } }],
                ['get', '/api/projects/{project_id}/robots/online', { params: { path: { project_id } } }],
            ],
        },
    });

    const editPath = paths.project.robots.edit({ project_id, robot_id: robot.id });
    const isSO101 = robot.type === 'SO101_Follower' || robot.type === 'SO101_Leader';

    return (
        <MenuTrigger>
            <ActionButton
                aria-label={`Actions for ${robot.name}`}
                isQuiet
                UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }}
            >
                <MoreMenu />
            </ActionButton>
            <Menu
                selectionMode='single'
                disabledKeys={
                    robot.active_calibration_id === null || robot.active_calibration_id === undefined
                        ? ['export-calibration']
                        : []
                }
                onAction={async (action) => {
                    if (action === 'delete') {
                        await deleteRobotMutation.mutateAsync(
                            { params: { path: { project_id, robot_id: robot.id } } },
                            {
                                onError: (error) => {
                                    if (isResourceInUseError(error)) {
                                        toast.info(
                                            getApiErrorMessage(error) ?? 'This robot is in use and cannot be deleted.'
                                        );
                                        return;
                                    }
                                    toast.negative(getApiErrorMessage(error) ?? 'Failed to delete robot.');
                                },
                            }
                        );
                    }
                    if (action === 'export-calibration') {
                        try {
                            await exportCalibration(project_id, robot);
                        } catch {
                            toast.negative('Failed to export calibration.');
                        }
                    }
                }}
            >
                <Item key='edit' href={editPath}>
                    Edit
                </Item>
                {isSO101 ? <Item key='export-calibration'>Export calibration</Item> : null}
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
    robot,
    status,
    isActive,
}: {
    robot: SchemaRobot;
    status: 'online' | 'offline' | 'unknown';
    isActive: boolean;
}) => {
    const payload = robot.payload;
    const connectionString =
        ('connection_string' in payload ? payload.connection_string : undefined) ??
        ('connection_string_left' in payload && 'connection_string_right' in payload
            ? `${payload.connection_string_left} | ${payload.connection_string_right}`
            : undefined);
    const serialNumber = 'serial_number' in robot.payload ? robot.payload.serial_number : undefined;

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
                        {robot.name}
                    </Heading>
                    <View gridArea='type' UNSAFE_style={{ fontSize: '14px' }}>
                        {robot.type.replaceAll('_', ' ')}
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
                                ID: <pre style={{ margin: 0, display: 'inline' }}>{robot.id}</pre>
                            </li>
                        </ul>
                    </View>
                    <View alignSelf={'end'}>
                        <MenuActions robot={robot} />
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
                                    robot={robot}
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
