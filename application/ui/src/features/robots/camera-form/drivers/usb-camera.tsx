import { Flex } from '@geti/ui';

import { SchemaUsbCameraInput } from '../../../../api/openapi-spec';
import { CameraPicker } from '../components/camera-picker';
import { FpsPicker } from '../components/fps-picker';
import { NameField } from '../components/name-field';
import { ResolutionPicker } from '../components/resolution-picker';
import { useCameraFormFields } from '../components/use-camera-form-fields';
import { DriverFormSchema } from '../provider';

export const initialUsbCameraState: DriverFormSchema<'usb_camera'> = {
    driver: 'usb_camera',
};

export const validateUsbCamera = (formData: DriverFormSchema<'usb_camera'>): formData is SchemaUsbCameraInput => {
    return (
        !!formData.name &&
        !!formData.fingerprint &&
        !!formData.payload?.width &&
        !!formData.payload?.height &&
        !!formData.payload?.fps
    );
};

export const USBCameraFormFields = () => {
    const { formData, updateField, updatePayload } = useCameraFormFields('usb_camera');

    return (
        <Flex gap='size-100' alignItems='end' direction='column'>
            <NameField value={formData.name ?? ''} onChange={(name) => updateField('name', name)} />
            <CameraPicker
                driver='usb_camera'
                selectedFingerprint={formData.fingerprint}
                onSelect={({ fingerprint, name }) => {
                    updateField('fingerprint', fingerprint);
                    updateField('hardware_name', name);
                }}
            />
            <ResolutionPicker
                driver='usb_camera'
                fingerprint={formData.fingerprint}
                selectedResolution={{
                    width: formData.payload?.width,
                    height: formData.payload?.height,
                }}
                onSelect={({ width, height, fps }) => {
                    const newFps = fps.find((f) => f === formData.payload?.fps) ?? fps.at(-1) ?? 30;
                    updatePayload({ width, height, fps: newFps });
                }}
            />
            <FpsPicker
                driver='usb_camera'
                fingerprint={formData.fingerprint}
                width={formData.payload?.width}
                height={formData.payload?.height}
                selectedFps={formData.payload?.fps}
                onSelect={(fps) => updatePayload({ fps })}
            />
        </Flex>
    );
};
