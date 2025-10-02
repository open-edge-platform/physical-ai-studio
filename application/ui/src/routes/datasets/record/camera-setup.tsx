import { Flex, Heading, Item, Key, Picker } from '@geti/ui';

import { SchemaCamera, SchemaCameraConfig } from '../../../api/openapi-spec';

interface CameraSetupProps {
    camera: SchemaCameraConfig;
    availableCameras: SchemaCamera[];
    updateCamera: (name: string, id: string, oldId: string, driver: string, oldDriver: string) => void;
}
export const CameraSetup = ({ camera, availableCameras, updateCamera }: CameraSetupProps) => {
    const camerasConnectedOfType = availableCameras;
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
            <img
                alt='Preview Camera'
                src={`/api/hardware/camera_feed?id=${camera.port_or_device_id}&driver=${camera.driver}`}
                style={{ flex: 1, maxWidth: '280px', paddingBottom: '10px' }}
            />
            <Picker selectedKey={camera.port_or_device_id} onSelectionChange={onSelection}>
                {camerasConnectedOfType.map((cam) => (
                    <Item key={makeKey(cam)}>{cam.name}</Item>
                ))}
            </Picker>
        </Flex>
    );
};
