import { ReactNode } from 'react';

import { Flex, Grid, Heading, minmax, repeat, View, Well } from '@geti/ui';

import { SchemaProjectCamera } from '../../api/types';
import { WebsocketCamera } from './websocket-camera';

const CameraWell = ({ children, aspectRatio }: { children: ReactNode; aspectRatio: number }) => {
    return (
        <Flex direction='column' alignContent='start' flex gap='size-30'>
            <Flex UNSAFE_style={{ aspectRatio }}>
                <Well flex UNSAFE_style={{ position: 'relative', overflow: 'hidden' }}>
                    <View
                        maxHeight='100%'
                        padding='size-400'
                        backgroundColor='gray-100'
                        height='100%'
                        position='relative'
                    >
                        {children}
                    </View>
                </Well>
            </Flex>
        </Flex>
    );
};

export const CameraFeed = ({ camera, empty = false }: { camera: SchemaProjectCamera; empty?: boolean }) => {
    const aspectRatio = camera.payload.width / camera.payload.height;

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
                <CameraWell aspectRatio={aspectRatio}>
                    <WebsocketCamera camera={camera} />
                </CameraWell>
            </Grid>
        </Flex>
    );
};
