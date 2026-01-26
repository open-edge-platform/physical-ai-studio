import { Flex } from '@geti/ui';

import { SchemaGenicamCameraInput } from '../../../../api/openapi-spec';
import { CameraPicker } from '../components/camera-picker';
import { FpsPicker } from '../components/fps-picker';
import { NameField } from '../components/name-field';
import { ResolutionPicker } from '../components/resolution-picker';
import { useCameraFormFields } from '../components/use-camera-form-fields';
import { DriverFormSchema } from '../provider';

export const initialGenicamState: DriverFormSchema<'genicam'> = {
    driver: 'genicam',
    payload: {
        fps: 30,
    },
};

export const validateGenicam = (formData: DriverFormSchema<'genicam'>): formData is SchemaGenicamCameraInput => {
    return !!formData.name && !!formData.fingerprint && !!formData.payload?.fps;
};

export const GenicamFormFields = () => {
    const { formData, updateField, updatePayload } = useCameraFormFields('genicam');

    return (
        <Flex gap='size-100' alignItems='end' direction='column'>
            <NameField value={formData.name ?? ''} onChange={(name) => updateField('name', name)} />
            <CameraPicker
                driver='genicam'
                selectedFingerprint={formData.fingerprint}
                onSelect={({ fingerprint, name }) => {
                    updateField('fingerprint', fingerprint);
                    updateField('hardware_name', name);
                }}
            />
            <ResolutionPicker
                driver='genicam'
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
                driver='genicam'
                fingerprint={formData.fingerprint}
                width={formData.payload?.width ?? undefined}
                height={formData.payload?.height ?? undefined}
                selectedFps={formData.payload?.fps}
                onSelect={(fps) => updatePayload({ fps })}
            />
        </Flex>
    );
};
