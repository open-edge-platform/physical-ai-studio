import { Flex } from '@geti/ui';

import { SchemaBaslerCameraInput } from '../../../../api/openapi-spec';
import { CameraPicker } from '../components/camera-picker';
import { NameField } from '../components/name-field';
import { useCameraFormFields } from '../components/use-camera-form-fields';
import { DriverFormSchema } from '../provider';

export const initialBaslerState: DriverFormSchema<'basler'> = {
    driver: 'basler',
};

export const validateBasler = (formData: DriverFormSchema<'basler'>): formData is SchemaBaslerCameraInput => {
    return !!formData.name && !!formData.fingerprint;
};

export const BaslerFormFields = () => {
    const { formData, updateField } = useCameraFormFields('basler');

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
        </Flex>
    );
};
