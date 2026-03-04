import { Item, Picker } from '@geti/ui';

import { CameraDriver } from '../provider';
import { useSupportedFormats } from './use-camera-data';

interface Resolution {
    width: number;
    height: number;
    fps: number[];
}

interface ResolutionPickerProps {
    driver: CameraDriver;
    fingerprint: string | undefined;
    selectedResolution: { width: number | undefined; height: number | undefined } | undefined;
    onSelect: (resolution: Resolution) => void;
}

export const ResolutionPicker = ({ driver, fingerprint, selectedResolution, onSelect }: ResolutionPickerProps) => {
    const supportedResolutions = useSupportedFormats(driver, fingerprint);

    const selectedKey =
        selectedResolution?.width && selectedResolution?.height
            ? `${selectedResolution.width}_${selectedResolution.height}`
            : undefined;

    const isDisabled = fingerprint === undefined;

    return (
        <Picker
            label='Resolution'
            width='100%'
            selectedKey={selectedKey}
            isDisabled={isDisabled}
            onSelectionChange={(key) => {
                const found = supportedResolutions.find(({ width, height }) => `${width}_${height}` === key);

                if (found) {
                    onSelect(found);
                }
            }}
        >
            {supportedResolutions.map(({ width, height }) => (
                <Item key={`${width}_${height}`}>{`${width} x ${height}`}</Item>
            ))}
        </Picker>
    );
};
