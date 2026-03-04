import { Item, Picker } from '@geti/ui';

import { CameraDriver } from '../provider';
import { useSupportedFormats } from './use-camera-data';

interface FpsPickerProps {
    driver: CameraDriver;
    fingerprint: string | undefined;
    width: number | undefined;
    height: number | undefined;
    selectedFps: number | undefined;
    onSelect: (fps: number) => void;
}

export const FpsPicker = ({ driver, fingerprint, width, height, selectedFps, onSelect }: FpsPickerProps) => {
    const supportedFormats = useSupportedFormats(driver, fingerprint);

    const resolution = supportedFormats.find((r) => r.width === width && r.height === height);
    const supportedFps = resolution?.fps ?? [];
    const isDisabled = fingerprint === undefined;

    return (
        <Picker
            label='Frames per second (FPS)'
            width='100%'
            isDisabled={isDisabled}
            selectedKey={selectedFps !== undefined ? String(selectedFps) : undefined}
            onSelectionChange={(key) => {
                if (key !== null) {
                    onSelect(Number(key));
                }
            }}
        >
            {supportedFps.map((fps) => (
                <Item key={fps}>{`${fps}`}</Item>
            ))}
        </Picker>
    );
};
