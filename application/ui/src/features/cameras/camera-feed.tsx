import { Flex, Grid, Heading, minmax, repeat } from '@geti/ui';

import { WebsocketCamera } from './websocket-camera';

type CameraFeedProps = {
    name: string;
    hardware_name: string;
    driver: string;
    fingerprint: string;
    fps: number;
    width: number;
    height: number;
};

export const CameraFeed = ({ camera, empty = false }: { camera: CameraFeedProps; empty?: boolean }) => {
    return (
        <Flex direction='column' gap='size-200'>
            {empty === false && camera && (
                <Flex gap='size-100' direction='column'>
                    <Heading level={3}>{camera.name}</Heading>
                    <Flex gap='size-100'>
                        <span style={{ fontSize: '10px', fontWeight: 'bold' }}>
                            <Flex gap='size-150'>
                                <span
                                    style={{
                                        backgroundColor: 'var(--spectrum-global-color-gray-300)',
                                        padding: '4px',
                                        borderRadius: '2px',
                                    }}
                                >
                                    {camera.hardware_name}
                                </span>
                                <span
                                    style={{
                                        backgroundColor: 'var(--spectrum-global-color-gray-300)',
                                        padding: '4px',
                                        borderRadius: '2px',
                                    }}
                                >
                                    {camera.fingerprint}
                                </span>
                                <span
                                    style={{
                                        backgroundColor: 'var(--spectrum-global-color-gray-300)',
                                        padding: '4px',
                                        borderRadius: '2px',
                                    }}
                                >
                                    {camera.width} x {camera.height} @ {camera.fps}
                                </span>
                            </Flex>
                        </span>
                    </Flex>
                </Flex>
            )}

            <Grid
                columns={repeat('auto-fit', minmax('size-6000', '1fr'))}
                rows={repeat('auto-fit', minmax('size-6000', '1fr'))}
                gap='size-400'
                width='100%'
            >
                <WebsocketCamera camera={camera} />
            </Grid>
        </Flex>
    );
};
