import { useCallback, useRef, useState } from 'react';

import { Flex, ProgressCircle, View, Well } from '@geti/ui';
import useWebSocket from 'react-use-websocket';
import { v4 as uuidv4 } from 'uuid';

import { SchemaProjectCamera } from '../../api/types';

const CAMERA_WS_URL = '/api/cameras/ws';

export const WebsocketCamera = ({ camera, empty = false }: { camera: SchemaProjectCamera; empty?: boolean }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isLoading, setIsLoading] = useState(true);
    const processingRef = useRef(false);
    const frameQueueRef = useRef<Blob | null>(null);

    const processFrame = useCallback(async (blobData: Blob) => {
        if (processingRef.current) {
            frameQueueRef.current = blobData;
            return;
        }

        processingRef.current = true;
        try {
            const bitmap = await createImageBitmap(blobData);
            const canvas = canvasRef.current;
            const ctx = canvas?.getContext('2d', { alpha: false });

            if (canvas && ctx) {
                ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
                setIsLoading(false);
            }

            bitmap.close();

            if (frameQueueRef.current) {
                const queuedBlob = frameQueueRef.current;
                frameQueueRef.current = null;
                processingRef.current = false;
                await processFrame(queuedBlob);
                return;
            }
        } catch (error) {
            console.error('Failed to process camera frame:', error);
        } finally {
            processingRef.current = false;
        }
    }, []);

    // WebSocket message handler
    const handleMessage = useCallback(
        (event: WebSocketEventMap['message']) => {
            try {
                if (event.data instanceof Blob) {
                    // Binary JPEG frame
                    void processFrame(event.data);
                } else {
                    console.info('Received unknown event', event.data);
                }
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        },
        [processFrame]
    );

    const [id] = useState(() => uuidv4());
    const params: SchemaProjectCamera = {
        id,
        name: camera.name,
        driver: camera.driver === 'webcam' ? 'usb_camera' : (camera.driver ?? ''),
        fingerprint: camera.fingerprint,
        hardware_name: camera.name,
        payload: {
            // stream_url: camera.fingerprint,
            fps: camera.payload.fps,
            width: camera.payload.width,
            height: camera.payload.height,
        },
    };

    useWebSocket(CAMERA_WS_URL, {
        queryParams: {
            camera: JSON.stringify(params),
        },
        shouldReconnect: () => true,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onMessage: handleMessage,
        onError: (error) => console.error('WebSocket error:', error),
        onClose: () => console.info('WebSocket closed'),
    });

    const aspectRatio = camera.payload.width / camera.payload.height;

    if (empty) {
        return (
            <View maxHeight='100%' height='100%' position='relative'>
                {isLoading && (
                    <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                        <ProgressCircle isIndeterminate />
                    </Flex>
                )}
                <canvas
                    ref={canvasRef}
                    width={camera.payload.width}
                    height={camera.payload.height}
                    style={{
                        display: isLoading ? 'none' : 'block',
                        objectFit: 'contain',
                        height: '100%',
                        width: '100%',
                    }}
                    aria-label={`Camera: ${camera.name}`}
                />
            </View>
        );
    }

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
                        {isLoading && (
                            <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                                <ProgressCircle isIndeterminate />
                            </Flex>
                        )}
                        <canvas
                            ref={canvasRef}
                            width={camera.payload.width}
                            height={camera.payload.height}
                            style={{
                                display: isLoading ? 'none' : 'block',
                                width: '100%',
                                height: '100%',
                                objectFit: 'contain',
                            }}
                            aria-label={`Live feed from ${camera.name}`}
                        />
                    </View>
                </Well>
            </Flex>
        </Flex>
    );
};
