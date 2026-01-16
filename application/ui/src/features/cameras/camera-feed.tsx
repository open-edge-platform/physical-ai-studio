import { Flex, Grid, Heading, minmax, repeat } from '@geti/ui';

import { SchemaProjectCamera } from '../../api/types';
import { WebsocketCamera } from './websocket-camera';

export const CameraFeed = ({ camera, empty = false }: { camera: SchemaProjectCamera; empty?: boolean }) => {
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
                                    {camera.payload.width} x {camera.payload.height} @ {camera.payload.fps}
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
