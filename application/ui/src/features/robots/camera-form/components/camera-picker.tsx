import { Item, Picker, Text } from '@geti-ui/ui';

import { CameraFingerprint, fingerprintKey, formatFingerprint } from '../../../cameras/fingerprint';
import { CameraDriver } from '../provider';
import { useAvailableCameras } from './use-camera-data';

interface CameraPickerProps {
    driver: CameraDriver;
    selectedFingerprint: CameraFingerprint | null | undefined;
    onSelect: (camera: { fingerprint: CameraFingerprint; name: string }) => void;
}

export const CameraPicker = ({ driver, selectedFingerprint, onSelect }: CameraPickerProps) => {
    const availableCameras = useAvailableCameras(driver);

    return (
        <Picker
            label='Camera'
            width='100%'
            selectedKey={fingerprintKey(selectedFingerprint)}
            onSelectionChange={(key) => {
                const selected = availableCameras.find(
                    ({ fingerprint }) => fingerprintKey(fingerprint) === String(key)
                );
                if (selected) {
                    onSelect({ fingerprint: selected.fingerprint, name: selected.name });
                }
            }}
        >
            {availableCameras.map((camera) => (
                <Item
                    textValue={formatFingerprint(camera.fingerprint)}
                    key={fingerprintKey(camera.fingerprint) ?? camera.name}
                >
                    <Text>{camera.name}</Text>
                    <Text slot='description'>
                        {formatFingerprint(camera.fingerprint)} ({camera.driver})
                    </Text>
                </Item>
            ))}
        </Picker>
    );
};
