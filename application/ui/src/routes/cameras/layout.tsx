import { Suspense } from 'react';

import { Button, Flex, Grid, Heading, Loading, View } from '@geti/ui';
import { NavLink, Outlet, useParams } from 'react-router-dom';

import { $api } from '../../api/client';
import { paths } from '../../router';

const CameraListItem = ({ id, cameraId, name, type }: { id: string; cameraId: string; name: string; type: string }) => {
    const { project_id = '' } = useParams<{ project_id: string }>();

    return (
        <NavLink to={paths.project.cameras.show({ project_id, camera_id: id })}>
            {({ isActive }) => {
                return (
                    <View
                        backgroundColor={'gray-100'}
                        padding='size-200'
                        UNSAFE_style={
                            isActive
                                ? {
                                      border: '2px solid var(--energy-blue)',
                                  }
                                : {}
                        }
                    >
                        <Flex direction={'column'} justifyContent={'space-between'} gap={'size-50'}>
                            <Flex justifyContent={'space-between'} gap={'size-100'}>
                                <span>{name}</span>
                            </Flex>
                            <Flex justifyContent={'space-between'} gap={'size-100'}>
                                <span>{cameraId}</span>
                                <span>{type}</span>
                            </Flex>
                        </Flex>
                    </View>
                );
            }}
        </NavLink>
    );
};

export const CamerasList = () => {
    const { project_id = '' } = useParams<{ project_id: string }>();
    const { data: cameras } = $api.useSuspenseQuery('get', '/api/hardware/cameras');

    return (
        <Flex direction={'column'} gap='size-200'>
            <Flex justifyContent={'space-between'}>
                <Heading level={4} marginY='size-100'>
                    Camera list
                </Heading>

                <Button href={paths.project.cameras.new({ project_id })} variant='accent' isDisabled>
                    Add
                </Button>
            </Flex>

            <Flex direction='column' gap='size-100'>
                <CameraListItem id={'webcam'} name={'Webcam tests'} cameraId={''} type={''} />
                <CameraListItem id={'overview'} name={'Overview tests'} cameraId={''} type={''} />
                {cameras.map((camera, idx) => {
                    return (
                        <CameraListItem
                            key={camera.id}
                            id={`${idx}`}
                            name={camera.name}
                            cameraId={camera.id}
                            type={camera.type}
                        />
                    );
                })}
            </Flex>
        </Flex>
    );
};

export const Layout = () => {
    return (
        <Grid
            gap='size-200'
            areas={['header header', 'camera controls']}
            //areas={['header header', 'controls cameras']}
            columns={['size-5000', '1fr']}
            rows={['auto', '1fr']}
            height={'100%'}
            UNSAFE_style={{ padding: 'var(--spectrum-global-dimension-size-400)' }}
        >
            <header style={{ gridArea: 'header' }}>
                <Heading level={2}>Cameras</Heading>
            </header>
            <View gridArea='camera' backgroundColor={'gray-200'} padding='size-200'>
                <CamerasList />
            </View>
            <View gridArea='controls' backgroundColor={'gray-200'} padding='size-200'>
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
