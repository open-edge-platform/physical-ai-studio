import { Flex, Heading, Item, Key, Picker } from '@geti/ui';

import { SchemaCamera, SchemaCameraConfig } from '../../../api/openapi-spec';

interface CameraSetupProps {
    camera: SchemaCameraConfig;
    availableCameras: SchemaCamera[];
    updateCamera: (name: string, id: string, oldId: string) => void;
}
export const CameraSetup = ({ camera, availableCameras, updateCamera }: CameraSetupProps) => {
    const camerasConnectedOfType = availableCameras.filter((m) => m.type === camera.type);

    const onSelection = (key: Key | null) => {
        if (key) {
            updateCamera(camera.name, String(key), camera.id);
        }
    };

    return (
        <Flex direction={'column'} flex={1}>
            <Heading>{camera.name}</Heading>
            <img
                alt='Preview Camera'
                src={`/api/hardware/camera_feed?id=${camera.id}&type=${camera.type}`}
                style={{ flex: 1, maxWidth: '280px', paddingBottom: '10px' }}
            />
            <Picker selectedKey={camera.id} onSelectionChange={onSelection}>
                {camerasConnectedOfType.map((cam) => (
                    <Item key={cam.id}>{cam.name}</Item>
                ))}
            </Picker>
        </Flex>
    );
};
