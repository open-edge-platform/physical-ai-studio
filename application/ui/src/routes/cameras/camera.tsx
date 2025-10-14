import { useState } from 'react';

import { Button, Flex, Grid, Heading, Icon, Loading, minmax, repeat, Text, View } from '@geti/ui';
import { Play, Close as Stop } from '@geti/ui/icons';
import { useParams } from 'react-router-dom';

import { $api } from '../../api/client';
import { Stream } from '../../components/stream/stream';
import { useWebRTCConnection, WebRTCConnectionProvider } from '../../components/stream/web-rtc-connection-provider';

import classes from './camera.module.scss';

const CameraView = ({ label }: { label: string }) => {
    const [size, setSize] = useState({ width: 300, height: 300 });
    const { start, status, stop } = useWebRTCConnection();

    return (
        <View maxHeight={'100%'} padding={'size-400'} backgroundColor={'gray-50'}>
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
    const { data: cameras } = $api.useSuspenseQuery('get', '/api/hardware/cameras');
    const params = useParams<{ project_id: string; camera_id: string }>();
    const camera = cameras.at(Number(params.camera_id));

    return (
        <Flex direction='column' gap='size-200'>
            <Heading level={3}>{camera?.name}</Heading>

            <Grid
                columns={repeat('auto-fit', minmax('size-6000', '1fr'))}
                rows={repeat('auto-fit', minmax('size-6000', '1fr'))}
                gap='size-400'
                width='100%'
            >
                {camera && (
                    <WebRTCConnectionProvider camera={camera}>
                        <CameraView label={camera.port_or_device_id} />
                    </WebRTCConnectionProvider>
                )}
            </Grid>
        </Flex>
    );
};

export const CameraOverview = () => {
    const { data: cameras } = $api.useSuspenseQuery('get', '/api/hardware/cameras');

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
                        <WebRTCConnectionProvider camera={camera} key={camera.port_or_device_id}>
                            <CameraView label={camera.port_or_device_id} />
                        </WebRTCConnectionProvider>
                    );
                })}
            </Grid>
        </Flex>
    );
};
