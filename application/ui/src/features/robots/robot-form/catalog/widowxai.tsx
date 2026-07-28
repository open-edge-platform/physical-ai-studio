import { Flex, TextField } from '@geti-ui/ui';

import type { SchemaTrossenSingleArmPayload } from '../../../../api/openapi-spec';
import { useCatalogIdentifyMutation } from '../../robot-catalog.hooks';
import type { SchemaRobot, SchemaRobotInput, SchemaRobotType } from '../../robot-types';
import { useRobotFormFields } from '../provider';
import { IdentifyRobot } from './actions';

export interface WidowxFormData {
    name: string;
    payload: SchemaTrossenSingleArmPayload;
}

export const getInitialWidowxFormData = (robot?: SchemaRobot): WidowxFormData => ({
    name: robot?.name ?? '',
    payload:
        robot && 'connection_string' in robot.payload ? robot.payload : { connection_string: '', serial_number: '' },
});

export const buildWidowxBody = (
    formData: WidowxFormData,
    schemaType: SchemaRobotType,
    robot_id: string
): SchemaRobotInput | null => {
    if (!formData.payload.connection_string) {
        return null;
    }

    return {
        id: robot_id,
        name: formData.name,
        type: schemaType,
        payload: formData.payload,
    } as SchemaRobotInput;
};

export const WidowxAIFormFields = () => {
    const { formData, updateField } = useRobotFormFields<WidowxFormData>();
    const identifyMutation = useCatalogIdentifyMutation();

    return (
        <Flex gap='size-100' justifyContent={'space-between'} alignItems={'end'}>
            <TextField
                isRequired
                label='Robot IP address'
                width='100%'
                value={formData.payload.connection_string}
                onChange={(connection_string) => {
                    updateField('payload', { ...formData.payload, connection_string, serial_number: '' });
                }}
                placeholder='192.168.1.2'
            />
            <Flex gap='size-100'>
                <IdentifyRobot identifyMutation={identifyMutation} />
            </Flex>
        </Flex>
    );
};
