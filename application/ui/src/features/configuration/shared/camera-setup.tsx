import { useEffect, useRef } from 'react';

import { Flex, Heading, Item, Key, Picker } from '@geti/ui';

import { SchemaCamera, SchemaCameraConfigInput } from '../../../api/openapi-spec';
import { WebRTCConnection } from '../../../components/stream/web-rtc-connection';

export const CameraPreview = ({ camera }: { camera: SchemaCameraConfigInput }) => {
    const size = { width: 240, height: 180 };

    const cameraRef = useRef<SchemaCameraConfigInput | null>(null);
    const webRTCConnectionRef = useRef<WebRTCConnection | null>(null);
    const videoRef = useRef<HTMLVideoElement>(null);

    useEffect(() => {
        if (cameraRef.current !== camera) {
            cameraRef.current = camera;

            const webRTCConnection = new WebRTCConnection(camera);
            webRTCConnectionRef.current = webRTCConnection;
            const unsubscribe = webRTCConnection.subscribe((event) => {
                if (event.type === 'status_change' && event.status === 'connected') {
                    const peerConnection = webRTCConnection?.getPeerConnection();
                    if (!peerConnection) {
                        return;
                    }
                    const receivers = peerConnection.getReceivers() ?? [];
                    const stream = new MediaStream(receivers.map((receiver) => receiver.track));

                    if (videoRef.current && videoRef.current.srcObject !== stream) {
                        videoRef.current.srcObject = stream;
                    }
                }
            });
            webRTCConnectionRef.current.start();

            return () => {
                if (cameraRef.current !== camera) {
                    unsubscribe();
                    webRTCConnection.stop(); // Ensure connection is closed on unmount
                    webRTCConnectionRef.current = null;
                }
            };
        }
    }, [camera, cameraRef, webRTCConnectionRef]);

    return (
        // eslint-disable-next-line jsx-a11y/media-has-caption
        <video
            ref={videoRef}
            autoPlay
            playsInline
            width={size.width}
            height={size.height}
            controls={false}
            style={{
                background: 'var(--spectrum-global-color-gray-200)',
            }}
        />
    );
};

interface CameraSetupProps {
    camera: SchemaCameraConfigInput;
    availableCameras: SchemaCamera[];
    updateCamera: (name: string, id: string, oldId: string, driver: string, oldDriver: string) => void;
}
export const CameraSetup = ({ camera, availableCameras, updateCamera }: CameraSetupProps) => {
    const camerasConnectedOfType = availableCameras.filter((m) => m.driver === camera.driver);
    const makeKey = (cam: SchemaCamera) => `${cam.driver}%${cam.fingerprint}`;

    const onSelection = (key: Key | null) => {
        if (key) {
            const [driver, id] = String(key).split('%');
            updateCamera(camera.name, String(id), camera.fingerprint ?? '', driver, camera.driver);
        }
    };

    return (
        <Flex direction={'column'} flex={1}>
            <Heading>{camera.name}</Heading>
            <Picker selectedKey={`${camera.driver}%${camera.fingerprint}`} onSelectionChange={onSelection}>
                {camerasConnectedOfType.map((cam) => (
                    <Item key={makeKey(cam)}>{cam.name}</Item>
                ))}
            </Picker>
        </Flex>
    );
};
