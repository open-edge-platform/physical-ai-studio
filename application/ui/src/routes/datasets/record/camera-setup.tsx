import { Flex, Heading, Item, Key, Picker, ProgressCircle } from '@geti/ui';

import { SchemaCamera, SchemaCameraConfig } from '../../../api/openapi-spec';
import { useState } from 'react';
import useWebSocket from 'react-use-websocket';
import { API_BASE_URL } from '../../../api/client';

const CameraPreview = ({ camera }: { camera: SchemaCameraConfig }) => {
    const [image, setImage] = useState<string>();
    const { sendJsonMessage } = useWebSocket(`${API_BASE_URL}/api/cameras/offer/camera/ws`, {
        onOpen: () => {
            if (camera.port_or_device_id !== ""){
                console.log(camera);
                sendJsonMessage(camera);
            }
        },
        onMessage: (message) => {
            setImage(message.data);
        },
    });

    if (image) {
        return (
            <img
                alt='Preview Camera'
                src={`data:image/jpg;base64,${image}`}
                style={{ flex: 1, maxWidth: '280px', paddingBottom: '10px' }}
            />
        );
    } else {
        return (
            <Flex
                width={280}
                height={(280 / camera.width) * camera.height + 10}
                justifyContent={'center'}
                alignItems={'center'}
            >
                {camera.port_or_device_id !== "" && <ProgressCircle isIndeterminate />}
            </Flex>
        );
    }
};



interface CameraSetupProps {
    camera: SchemaCameraConfig;
    availableCameras: SchemaCamera[];
    updateCamera: (name: string, id: string, oldId: string) => void;
}
export const CameraSetup = ({ camera, availableCameras, updateCamera }: CameraSetupProps) => {
    const camerasConnectedOfType = availableCameras.filter((m) => m.type === camera.type);
    console.log(camera);

    const onSelection = (key: Key | null) => {
        if (key) {
            updateCamera(camera.name, String(key), camera.id ?? '');
        }
    };

    return (
        <Flex direction={'column'} flex={1}>
            <Heading>{camera.name}</Heading>
            <CameraPreview key={camera.port_or_device_id} camera={camera} />
            <Picker selectedKey={camera.port_or_device_id} onSelectionChange={onSelection}>
                {camerasConnectedOfType.map((cam) => (
                    <Item key={cam.id}>{cam.name}</Item>
                ))}
            </Picker>
        </Flex>
    );
};
