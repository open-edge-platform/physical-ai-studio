import { Item, Picker, Text } from '@geti/ui';

import { CameraDriver } from '../provider';
import { useAvailableCameras } from './use-camera-data';

interface CameraPickerProps {
    driver: CameraDriver;
    selectedFingerprint: string | undefined;
    onSelect: (camera: { fingerprint: string; name: string }) => void;
}

export const CameraPicker = ({ driver, selectedFingerprint, onSelect }: CameraPickerProps) => {
    const availableCameras = useAvailableCameras(driver);

    return (
        <Picker
            label='Camera'
            width='100%'
            selectedKey={selectedFingerprint}
            onSelectionChange={(key) => {
                const selected = availableCameras.find(({ fingerprint }) => fingerprint === key);
                if (selected) {
                    onSelect({ fingerprint: selected.fingerprint, name: selected.name });
                }
            }}
        >
            {availableCameras.map((camera) => (
                <Item textValue={camera.fingerprint} key={camera.fingerprint}>
                    <Text>{camera.name}</Text>
                    <Text slot='description'>
                        {camera.fingerprint} ({camera.driver})
                    </Text>
                </Item>
            ))}
        </Picker>
    );
};
