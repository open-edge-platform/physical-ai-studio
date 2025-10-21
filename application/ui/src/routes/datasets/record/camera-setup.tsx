import { useEffect, useState } from 'react';

import { Flex, Heading, Item, Key, Picker, ProgressCircle } from '@geti/ui';
import useWebSocket from 'react-use-websocket';

import { API_BASE_URL } from '../../../api/client';
import { SchemaCamera, SchemaCameraConfig } from '../../../api/openapi-spec';
import { useWebRTCConnection, WebRTCConnectionProvider } from '../../../components/stream/web-rtc-connection-provider';
import { Stream } from '../../../components/stream/stream';

export const CameraView = ({ label }: { label: string }) => {
    const [size, setSize] = useState({ width: 280, height: 180 });
    const { start, status, stop, webRTCConnectionRef } = useWebRTCConnection();
    //useEffect(() => {
    //    console.log(status);
    //    console.log(webRTCConnectionRef.current)
    //    if (status === "idle" && webRTCConnectionRef.current !== null) {
    //        console.log("starting...")
    //        start();
    //    }
    //}, [status, webRTCConnectionRef.current])

    if (status === "connected") {
        return (
            <Stream setSize={setSize} size={size} />
        )
    }
    return <></>
}

const CameraPreview = ({ camera }: { camera: SchemaCameraConfig }) => {

    const cameraConfig: SchemaCamera = {
        driver: camera.driver,
        name: camera.name,
        port_or_device_id: camera.port_or_device_id,
        default_stream_profile: {
            fps: camera.fps,
            height: camera.height,
            width: camera.width,
        },
    };

    return (
        <WebRTCConnectionProvider camera={cameraConfig} autoplay={true}>
            <CameraView label={camera.name} />
        </WebRTCConnectionProvider>
    );
};

interface CameraSetupProps {
    camera: SchemaCameraConfig;
    availableCameras: SchemaCamera[];
    updateCamera: (name: string, id: string, oldId: string, driver: string, oldDriver: string) => void;
}
export const CameraSetup = ({ camera, availableCameras, updateCamera }: CameraSetupProps) => {
    const camerasConnectedOfType = availableCameras.filter((m) => m.driver === camera.driver);
    const makeKey = (cam: SchemaCamera) => `${cam.driver}%${cam.port_or_device_id}`;

    const onSelection = (key: Key | null) => {
        if (key) {
            const [driver, id] = String(key).split('%');
            updateCamera(camera.name, String(id), camera.port_or_device_id ?? '', driver, camera.driver);
        }
    };

    return (
        <Flex direction={'column'} flex={1}>
            <Heading>{camera.name}</Heading>
            {/*<CameraPreview key={camera.port_or_device_id} camera={camera} />*/}
            <Picker selectedKey={`${camera.driver}%${camera.port_or_device_id}`} onSelectionChange={onSelection}>
                {camerasConnectedOfType.map((cam) => (
                    <Item key={makeKey(cam)}>{cam.name}</Item>
                ))}
            </Picker>
        </Flex>
    );
};
