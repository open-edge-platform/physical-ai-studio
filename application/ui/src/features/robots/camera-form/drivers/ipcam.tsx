import { Flex, TextField } from '@geti/ui';

import { SchemaIpCameraInput } from '../../../../api/openapi-spec';
import { NameField } from '../components/name-field';
import { useCameraFormFields } from '../components/use-camera-form-fields';
import { DriverFormSchema } from '../provider';

export const initialIpCamState: DriverFormSchema<'ipcam'> = {
    driver: 'ipcam',
    hardware_name: null,
    payload: {
        fps: 30,
        width: 480,
        height: 480,
        stream_url: '',
    },
};

export const validateIpCam = (formData: DriverFormSchema<'ipcam'>): formData is SchemaIpCameraInput => {
    return (
        !!formData.name &&
        !!formData.fingerprint &&
        !!formData.payload?.width &&
        !!formData.payload?.height &&
        !!formData.payload?.fps &&
        !!formData.payload?.stream_url
    );
};

export const IpCamFormFields = () => {
    const { formData, updateField, updatePayload } = useCameraFormFields('ipcam');

    return (
        <Flex gap='size-100' alignItems='end' direction='column'>
            <NameField value={formData.name ?? ''} onChange={(name) => updateField('name', name)} />
            <TextField
                isRequired
                label='Stream URL'
                width='100%'
                value={formData.fingerprint ?? ''}
                onChange={(url) => {
                    updateField('fingerprint', url);
                    updatePayload({ stream_url: url });
                }}
            />
        </Flex>
    );
};
