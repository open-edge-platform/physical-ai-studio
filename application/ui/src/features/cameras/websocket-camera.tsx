import { useCallback, useRef, useState } from 'react';

import { Flex, ProgressCircle } from '@geti/ui';
import useWebSocket from 'react-use-websocket';

import { SchemaProjectCamera } from '../../api/types';

const CAMERA_WS_URL = '/api/cameras/ws';

export const WebsocketCamera = ({ camera }: { camera: SchemaProjectCamera }) => {
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

    useWebSocket(CAMERA_WS_URL, {
        queryParams: {
            camera: JSON.stringify({
                ...camera,
                // Prevent the stream from resetting anytime the user changes the camera name
                name: camera.hardware_name ?? '_',
            }),
        },
        shouldReconnect: () => true,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onMessage: handleMessage,
        onError: (error) => console.error('WebSocket error:', error),
        onClose: () => console.info('WebSocket closed'),
    });

    return (
        <>
            {isLoading && (
                <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                    <ProgressCircle isIndeterminate />
                </Flex>
            )}
            <canvas
                ref={canvasRef}
                width={Number(camera.payload?.width)}
                height={Number(camera.payload?.height)}
                style={{
                    display: isLoading ? 'none' : 'block',
                    objectFit: 'contain',
                    height: '100%',
                    width: '100%',
                }}
                aria-label={`Camera: ${camera.name}`}
            />
        </>
    );
};
