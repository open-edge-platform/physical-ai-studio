import { useEffect, useRef, useState } from 'react';

import { Button, Flex, Grid, Icon, Loading, minmax, repeat, Text, toast, View } from '@geti/ui';
import { Play, Close as Stop } from '@geti/ui/icons';
import { useSuspenseQuery } from '@tanstack/react-query';

import classes from './camera.module.scss';

const WebCamView = ({ device }: { device: MediaDeviceInfo }) => {
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const [status, setStatus] = useState<'idle' | 'connecting' | 'connected'>('idle');

    const start = async () => {
        setStatus('connecting');
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    deviceId: { exact: device.deviceId },
                },
                audio: false,
            });

            streamRef.current = stream;

            if (!videoRef.current) {
                return;
            }

            const videoEl = videoRef.current;
            videoEl.srcObject = streamRef.current;
            videoEl.autoplay = true;
            videoEl.muted = true;
            videoEl.playsInline = true;
            setStatus('connected');
        } catch (error) {
            toast({ message: `Failed to connect to camera: ${error}`, type: 'error' });
            setStatus('idle');
        }
    };

    const stop = () => {
        if (streamRef.current) {
            const stream = streamRef.current;
            stream.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
        }

        if (!videoRef.current || !videoRef.current.srcObject) {
            return;
        }

        // Clear the source object
        videoRef.current.srcObject = null;
        setStatus('idle');
    };

    useEffect(() => {
        return () => {
            if (status === 'connected') {
                stop();
            }
        };
    }, [status]);

    return (
        <View maxHeight={'100%'} padding={'size-400'} backgroundColor={'gray-100'}>
            <Grid areas={['canvas']} alignItems={'center'} justifyItems={'center'} height='100%'>
                <View gridArea='canvas' width='100%' height='100%'>
                    <video
                        ref={videoRef}
                        autoPlay
                        muted
                        playsInline
                        style={{
                            width: '100%',
                            height: '100%',
                            backgroundColor: 'var(--spectrum-global-color-gray-50)',
                        }}
                    />
                </View>
                {status === 'connecting' && (
                    <Grid gridArea='canvas' width='100%' height='100%'>
                        <Loading mode='inline' />
                    </Grid>
                )}
                <Flex justifyContent={'space-between'} gridArea='canvas' width='100%' alignSelf={'end'}>
                    <View padding='size-100'>
                        <Text>{device.label}</Text>
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

export const CameraWebcam = () => {
    const devicesQuery = useSuspenseQuery({
        queryKey: ['cameras'],
        queryFn: async () => {
            const allDevices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = allDevices.filter((d) => d.kind === 'videoinput');

            return videoDevices;
        },
    });

    return (
        <Grid
            columns={repeat('auto-fit', minmax('size-6000', '1fr'))}
            rows={repeat('auto-fit', minmax('size-6000', '1fr'))}
            gap='size-400'
            width='100%'
        >
            {devicesQuery.data.map((device) => {
                return <WebCamView device={device} key={device.deviceId} />;
            })}
        </Grid>
    );
};
