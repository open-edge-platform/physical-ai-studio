import { Suspense } from 'react';

import {
    ActionButton,
    Button,
    Divider,
    Flex,
    Grid,
    Heading,
    Icon,
    Item,
    Loading,
    Menu,
    MenuTrigger,
    minmax,
    View,
} from '@geti/ui';
import { Add, MoreMenu } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { NavLink, Outlet, useParams } from 'react-router-dom';

import { $api } from '../../api/client';
import { SchemaProjectCamera } from '../../api/types';
import { useProjectId } from '../../features/projects/use-project';
import { ConnectionStatus } from '../../features/robots/robots-list';
import { paths } from '../../router';
import { ReactComponent as CameraIcon } from './../../assets/camera.svg';

import classes from './../../features/robots/robots-list.module.scss';

const MenuActions = ({ camera_id }: { camera_id: string }) => {
    const { project_id } = useProjectId();
    const deleteCameraMutation = $api.useMutation('delete', '/api/projects/{project_id}/cameras/{camera_id}');

    return (
        <MenuTrigger>
            <ActionButton isQuiet UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }}>
                <MoreMenu />
            </ActionButton>
            <Menu
                selectionMode='single'
                onAction={(action) => {
                    if (action === 'delete') {
                        deleteCameraMutation.mutate({ params: { path: { project_id, camera_id } } });
                    }
                }}
            >
                <Item href={paths.project.cameras.edit({ project_id, camera_id })}>Edit</Item>
                <Item key='delete'>Delete</Item>
            </Menu>
        </MenuTrigger>
    );
};

const CameraListItem = ({
    status,
    camera,
    isActive,
}: {
    status: 'connected' | 'disconnected';
    camera: SchemaProjectCamera;
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
            <Flex direction={'column'} justifyContent={'space-between'} gap={'size-50'}>
                <Grid
                    areas={['icon name status', 'parameters parameters menu']}
                    columns={['auto', '1fr']}
                    gap={'size-100'}
                >
                    <View gridArea={'icon'} padding='size-100'>
                        <CameraIcon style={{ width: '32px', height: '32px' }} />
                    </View>
                    <View gridArea='name'>
                        <Heading level={2} UNSAFE_style={isActive ? { color: 'var(--energy-blue)' } : {}}>
                            {camera.name}
                        </Heading>
                        <View UNSAFE_style={{ fontSize: '14px' }}>
                            {camera.payload.width} x {camera.payload.height} @ {camera.payload.fps}
                        </View>
                    </View>
                    <View gridArea='status'>
                        <ConnectionStatus status={status} />
                    </View>
                    <View gridArea='menu' alignSelf={'end'} justifySelf={'end'}>
                        <MenuActions camera_id={camera.id ?? 'undefined'} />
                    </View>
                    <View gridArea='parameters'>
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
                                {camera.driver}: {camera.hardware_name}
                            </li>
                            <li style={{ marginLeft: 'var(--spectrum-global-dimension-size-200)' }}>
                                {camera.fingerprint}
                            </li>
                        </ul>
                    </View>
                </Grid>
            </Flex>
        </View>
    );
};

export const CamerasList = () => {
    const { project_id = '' } = useParams<{ project_id: string }>();
    const { data: hardwareCameras } = $api.useSuspenseQuery('get', '/api/hardware/cameras');
    const { data: projectCameras } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras', {
        params: { path: { project_id } },
    });

    return (
        <Flex direction='column' gap='size-100'>
            {/* TODO:  */}
            <View isHidden>
                <Flex justifyContent={'space-between'} alignItems={'end'}>
                    <span>Step 2: setup cameras</span>
                    <Button>Next</Button>
                </Flex>
                <Divider size='S' marginY='size-200' />
            </View>
            <Button
                variant='secondary'
                href={paths.project.cameras.new({ project_id })}
                UNSAFE_className={classes.addNewRobotButton}
            >
                <Icon marginEnd='size-50'>
                    <Add />
                </Icon>
                Configure new camera
            </Button>

            <Flex direction='column' gap='size-100'>
                {projectCameras.map((camera) => {
                    const hardwareCamera = hardwareCameras.find((hardware) => {
                        return hardware.fingerprint === camera.fingerprint;
                    });
                    const to = paths.project.cameras.show({ project_id, camera_id: camera.id ?? 'undefined' });

                    return (
                        <NavLink key={camera.id} to={to}>
                            {({ isActive }) => {
                                return (
                                    <CameraListItem
                                        camera={camera}
                                        isActive={isActive}
                                        status={hardwareCamera !== undefined ? 'connected' : 'disconnected'}
                                    />
                                );
                            }}
                        </NavLink>
                    );
                })}
            </Flex>
        </Flex>
    );
};

export const Layout = () => {
    return (
        <Grid gap='size-200' areas={['camera controls']} columns={[minmax('size-6000', 'auto'), '1fr']} height={'100%'}>
            <View gridArea='camera' backgroundColor={'gray-100'} padding='size-400'>
                <CamerasList />
            </View>
            <View gridArea='controls' backgroundColor={'gray-50'} padding='size-400'>
                <Suspense
                    fallback={
                        <Grid width='100%' height='100%'>
                            <Loading mode='inline' />
                        </Grid>
                    }
                >
                    <Outlet />
                </Suspense>
            </View>
        </Grid>
    );
};
