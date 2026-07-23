import { Flex, TextField } from '@geti-ui/ui';
import { v4 as uuidv4 } from 'uuid';

import type { SchemaTrossenSingleArmPayload } from '../../../../api/openapi-spec';
import type { SchemaRobot, SchemaRobotInput, SchemaRobotType } from '../../robot-types';
import { useRobotFormFields } from '../provider';
import { IdentifyRobot, useIdentifyMutation } from './actions';

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
    const { formData, updateField, activeType } = useRobotFormFields<WidowxFormData>();

    const identifyMutation = useIdentifyMutation();
    const identifyRobot = buildWidowxBody(formData, activeType, uuidv4());

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
                <IdentifyRobot identifyMutation={identifyMutation} robot={identifyRobot} />
            </Flex>
        </Flex>
    );
};
