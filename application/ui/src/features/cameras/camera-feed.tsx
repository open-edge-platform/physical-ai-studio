import { ReactNode } from 'react';

import { Flex, Grid, Heading, minmax, View } from '@geti-ui/ui';

import { SchemaProjectCamera } from '../../api/types';
import { WebsocketCamera } from './websocket-camera';

const CameraWell = ({
    children,
    width,
    height,
}: {
    children: ReactNode;
    width?: number | null;
    height?: number | null;
}) => {
    const aspectRatio = width && height ? width / height : undefined;

    return (
        <View
            UNSAFE_style={{ aspectRatio, position: 'relative', overflow: 'hidden' }}
            width='100%'
            height='100%'
            minHeight={0}
        >
            <View maxHeight='100%' height='100%' position='relative'>
                {children}
            </View>
        </View>
    );
};

const CameraHeading = ({ camera }: { camera: SchemaProjectCamera }) => {
    return (
        <Flex gap='size-100' direction='column'>
            <Heading level={3}>{camera.name}</Heading>
            <Flex gap='size-100'>
                <span style={{ fontSize: '10px', fontWeight: 'bold' }}>
                    <Flex gap='size-150'>
                        {camera.hardware_name && (
                            <span
                                style={{
                                    backgroundColor: 'var(--spectrum-global-color-gray-300)',
                                    padding: '4px',
                                    borderRadius: '2px',
                                }}
                            >
                                {camera.hardware_name}
                            </span>
                        )}
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
    );
};

export const CameraFeed = ({ camera, empty = false }: { camera: SchemaProjectCamera; empty?: boolean }) => {
    return (
        <Flex direction='column' gap='size-200' height='100%' minHeight={0}>
            {empty === false && camera && <CameraHeading camera={camera} />}

            <Grid
                columns={[minmax(0, '1fr')]}
                rows={[minmax(0, '1fr')]}
                gap='size-400'
                width='100%'
                flex={1}
                minHeight={0}
            >
                <CameraWell width={camera.payload.width} height={camera.payload.height}>
                    <WebsocketCamera camera={camera} />
                </CameraWell>
            </Grid>
        </Flex>
    );
};
