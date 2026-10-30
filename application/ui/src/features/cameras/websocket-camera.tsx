import { useMemo, useRef, useState } from 'react';

import useWebSocket from 'react-use-websocket';

import { fetchClient } from '../../api/client';
import { SchemaProjectCamera } from '../../api/types';
import { CameraView, FrameSource } from './camera-view';

const CAMERA_WS_URL = fetchClient.PATH('/api/cameras/ws');

/** A standalone camera socket, for the camera form. Pages with a runtime session use the session's frames. */
export const useCameraSocket = (camera: SchemaProjectCamera) => {
    const listeners = useRef(new Set<(jpeg: Blob) => void>()).current;
    const [error, setError] = useState<string | null>(null);

    useWebSocket(CAMERA_WS_URL, {
        queryParams: {
            camera: JSON.stringify({
                ...camera,
                // Prevent the stream from resetting anytime the user changes the camera name
                name: camera.hardware_name ?? '_',
            }),
        },
        share: true,
        shouldReconnect: () => error === null,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onOpen: () => setError(null),
        onMessage: (event: MessageEvent) => {
            if (event.data instanceof Blob) {
                listeners.forEach((listener) => listener(event.data));
                return;
            }
            const message = JSON.parse(event.data) as { event?: string; message?: string };
            if (message.event === 'error') {
                setError(message.message ?? 'The camera stream failed.');
            }
        },
        onError: (wsError) => console.error('WebSocket error:', wsError),
    });

    const source: FrameSource = useMemo(
        () => ({
            subscribe: (listener) => {
                listeners.add(listener);
                return () => {
                    listeners.delete(listener);
                };
            },
        }),
        [listeners]
    );

    return { source, error };
};

export const WebsocketCamera = ({ camera }: { camera: SchemaProjectCamera }) => {
    const { source, error } = useCameraSocket(camera);

    return (
        <CameraView
            source={source}
            label={`Camera: ${camera.name}`}
            width={camera.payload?.width}
            height={camera.payload?.height}
            error={error}
        />
    );
};
