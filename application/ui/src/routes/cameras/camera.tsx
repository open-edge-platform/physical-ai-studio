import { useState } from 'react';

import { Button, Flex, Grid, Heading, Icon, Loading, minmax, repeat, Text, View } from '@geti/ui';
import { Play, Close as Stop } from '@geti/ui/icons';

import { $api } from '../../api/client';
import { Stream } from '../../components/stream/stream';
import { useWebRTCConnection, WebRTCConnectionProvider } from '../../components/stream/web-rtc-connection-provider';
import { ProjectCameraFeed } from '../../features/cameras/project-camera-feed';
import { WebsocketCamera } from '../../features/cameras/websocket-camera';
import { useProjectId } from '../../features/projects/use-project';
import { useCamera } from '../../features/robots/use-camera';

import classes from './camera.module.scss';

export const FullCameraView = ({ camera_id }: { camera_id: string }) => {
    const { project_id } = useProjectId();
    const cameraQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras/{camera_id}', {
        params: { path: { camera_id, project_id } },
    });

    const camera = cameraQuery.data;

    return (
        <WebRTCConnectionProvider camera={camera} key={camera_id}>
            <CameraView label={camera.fingerprint} />
        </WebRTCConnectionProvider>
    );
};

export const CameraView = ({ label }: { label: string }) => {
    const [size, setSize] = useState({ width: 300, height: 300 });
    const { start, status, stop } = useWebRTCConnection();

    return (
        <View maxHeight={'100%'} padding={'size-400'} backgroundColor={'gray-100'} height='100%'>
            <Grid areas={['canvas']} alignItems={'center'} justifyItems={'center'} height='100%'>
                <View gridArea='canvas' UNSAFE_className={classes.canvasContainer}>
                    <Stream setSize={setSize} size={size} />
                </View>
                {status === 'connecting' && (
                    <Grid gridArea='canvas' width='100%' height='100%'>
                        <Loading mode='inline' />
                    </Grid>
                )}
                <Flex justifyContent={'space-between'} gridArea='canvas' width='100%' alignSelf={'end'}>
                    <View padding='size-100'>
                        <Text>{label}</Text>
                    </View>
                    {status === 'connected' && (
                        <View padding='size-100'>
                            <Button style='fill' onPress={stop} aria-label={'Stop stream'}>
                                <Icon>
                                    <Stop width='32px' height='32px' />
                                </Icon>
                            </Button>
                        </View>
                    )}
                </Flex>
                {status === 'idle' && (
                    <Grid gridArea='canvas' width='100%' height='100%'>
                        <Button
                            onPress={start}
                            UNSAFE_className={classes.playButton}
                            aria-label={'Start stream'}
                            justifySelf={'center'}
                            alignSelf={'center'}
                        >
                            <Play width='32px' height='32px' />
                        </Button>
                    </Grid>
                )}
            </Grid>
        </View>
    );
};

export const Camera = () => {
    const actualCamera = useCamera();

    return <ProjectCameraFeed camera={actualCamera} />;
};

export const CameraOverview = () => {
    const { project_id } = useProjectId();
    //const { data: cameras } = $api.useSuspenseQuery('get', '/api/hardware/cameras');
    const { data: cameras } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras', {
        params: { path: { project_id } },
    });

    return (
        <Flex direction='column' gap='size-200'>
            <Heading level={3}>Cameras overview</Heading>

            <Grid
                columns={repeat('auto-fit', minmax('size-6000', '1fr'))}
                rows={repeat('auto-fit', minmax('size-6000', '1fr'))}
                gap='size-400'
                width='100%'
            >
                {cameras.map((camera) => {
                    return (
                        <WebsocketCamera
                            camera={{
                                ...camera,
                                hardware_name: camera.hardware_name,
                                driver: camera.driver ?? '',
                                fingerprint: camera.fingerprint ?? '',
                                fps: camera.payload.fps,
                                width: camera.payload.width,
                                height: camera.payload.height,
                                payload: camera.payload,
                            }}
                        />
                    );
                })}
            </Grid>
        </Flex>
    );
};
