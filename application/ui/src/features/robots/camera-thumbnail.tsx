import { Button, Flex, Grid, Loading, View } from '@geti/ui';
import { Play } from '@geti/ui/icons';

import { $api } from '../../api/client';
import { useStreamToVideo } from '../../components/stream/stream';
import { useWebRTCConnection, WebRTCConnectionProvider } from '../../components/stream/web-rtc-connection-provider';
import RobotPicture from './../../assets/robot-picture.png';

const CameraView = () => {
    const { start, status } = useWebRTCConnection();
    const videoRef = useStreamToVideo();

    return (
        <Grid areas={['canvas']} alignItems={'center'} justifyItems={'center'}>
            <View gridArea='canvas'>
                {status === 'connected' && (
                    // eslint-disable-next-line jsx-a11y/media-has-caption
                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        controls={true}
                        style={{
                            borderRadius: '8px',
                            maxWidth: 'min(400px, 100%)',
                            maxHeight: '400px',
                        }}
                    />
                )}
            </View>
            <Grid
                gridArea='canvas'
                width='100%'
                height='100%'
                UNSAFE_style={{
                    placeContent: 'center',
                    placeItems: 'center',
                    padding: 'var(--spectrum-global-dimension-size-400)',
                }}
            >
                {status === 'connecting' && <Loading mode='inline' />}
                {status === 'idle' && (
                    <Button
                        onPress={start}
                        aria-label={'Start stream'}
                        justifySelf={'center'}
                        alignSelf={'center'}
                        UNSAFE_style={{
                            display: 'flex',
                            gap: 'var(--spectrum-global-dimension-size-400)',
                            alignItems: 'center',
                            padding: 'var(--spectrum-global-dimension-size-400)',
                            position: 'relative',
                            borderRadius: 'var(--spectrum-alias-border-radius-medium)',
                        }}
                    >
                        <Play width='32px' height='32px' />
                    </Button>
                )}
            </Grid>
        </Grid>
    );
};

export const CameraThumbnail = ({ name, fingerprint }: { name: string; fingerprint: string | null }) => {
    const availableCamerasQuery = $api.useQuery('get', '/api/hardware/cameras');
    const availableCamera = availableCamerasQuery.data?.find(
        ({ port_or_device_id }) => port_or_device_id === fingerprint
    );

    const ratio =
        (availableCamera?.default_stream_profile?.width ?? 1) / (availableCamera?.default_stream_profile?.height ?? 1);

    const defaultSize = 300;
    return (
        <Grid
            UNSAFE_style={{
                background: 'var(--spectrum-global-color-gray-100)',
                border: '1px solid var(--spectrum-global-color-gray-400)',
                borderRadius: '8px',
                padding: 'var(--spectrum-global-dimension-size-150)',
            }}
            areas={['camera']}
            width={availableCamera?.default_stream_profile?.width}
            maxWidth={defaultSize * ratio}
        >
            <Flex gridArea='camera' justifyContent='center' alignItems='center'>
                {availableCamera ? (
                    <WebRTCConnectionProvider camera={availableCamera}>
                        <CameraView />
                    </WebRTCConnectionProvider>
                ) : (
                    <img
                        alt='Robot camera placeholder'
                        src={RobotPicture}
                        style={{ borderRadius: '8px', width: '100%' }}
                    />
                )}
            </Flex>

            {name && (
                <Flex gridArea='camera' alignItems='start' justifyContent='end'>
                    <span
                        style={{
                            background: 'var(--spectrum-global-color-gray-300)',
                            color: '#E3E3E5',
                            padding: 'var(--spectrum-global-dimension-size-50)',
                            borderRadius: '8px',
                            marginRight: '-4px',
                            marginTop: '-4px',
                            fontSize: '12px',
                            position: 'relative',
                        }}
                    >
                        {name}
                    </span>
                </Flex>
            )}
        </Grid>
    );
};
