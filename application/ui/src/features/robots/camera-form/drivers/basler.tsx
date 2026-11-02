import { Flex } from '@geti-ui/ui';

import { SchemaBaslerCameraInput } from '../../../../api/openapi-spec';
import { CameraPicker } from '../components/camera-picker';
import { FpsPicker } from '../components/fps-picker';
import { NameField } from '../components/name-field';
import { ResolutionPicker } from '../components/resolution-picker';
import { useCameraFormFields } from '../components/use-camera-form-fields';
import { DriverFormSchema } from '../provider';

export const initialBaslerState: DriverFormSchema<'basler'> = {
    driver: 'basler',
    payload: {
        fps: 30,
        is_mono: false,
    },
};

export const validateBasler = (formData: DriverFormSchema<'basler'>): formData is SchemaBaslerCameraInput => {
    return (
        !!formData.name &&
        !!formData.fingerprint &&
        !!formData.payload?.width &&
        !!formData.payload?.height &&
        !!formData.payload?.fps &&
        formData.payload?.is_mono !== undefined
    );
};

export const BaslerFormFields = () => {
    const { formData, updateField, updatePayload } = useCameraFormFields('basler');

    return (
        <Flex gap='size-100' alignItems='end' direction='column'>
            <NameField value={formData.name ?? ''} onChange={(name) => updateField('name', name)} />
            <CameraPicker
                driver='basler'
                selectedFingerprint={formData.fingerprint}
                onSelect={({ fingerprint, name }) => {
                    updateField('fingerprint', fingerprint);
                    updateField('hardware_name', name);
                }}
            />
            <ResolutionPicker
                driver='basler'
                fingerprint={formData.fingerprint}
                selectedResolution={{
                    width: formData.payload?.width ?? undefined,
                    height: formData.payload?.height ?? undefined,
                }}
                onSelect={({ width, height, fps }) => {
                    const newFps = fps.find((f) => f === formData.payload?.fps) ?? fps[0] ?? 30;
                    updatePayload({ width, height, fps: newFps });
                }}
            />
            <FpsPicker
                driver='basler'
                fingerprint={formData.fingerprint}
                width={formData.payload?.width ?? undefined}
                height={formData.payload?.height ?? undefined}
                selectedFps={formData.payload?.fps}
                onSelect={(fps) => updatePayload({ fps })}
            />
        </Flex>
    );
};
