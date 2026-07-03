import { Flex, TextField } from '@geti-ui/ui';
import { v4 as uuidv4 } from 'uuid';

import type { SchemaRobot, SchemaRobotInput, SchemaRobotType } from '../../robot-types';
import { useRobotFormFields } from '../provider';
import { IdentifyRobot, useIdentifyMutation } from './actions';

export interface WidowxFormData {
    name: string;
    connection_string: string;
    serial_number: string;
}

export const getInitialWidowxFormData = (robot?: SchemaRobot): WidowxFormData => ({
    name: robot?.name ?? '',
    connection_string: robot && 'connection_string' in robot.payload ? robot.payload.connection_string : '',
    serial_number: robot?.payload?.serial_number ?? '',
});

export const buildWidowxBody = (
    formData: WidowxFormData,
    schemaType: SchemaRobotType,
    robot_id: string
): SchemaRobotInput | null => {
    if (!formData.connection_string) {
        return null;
    }

    return {
        id: robot_id,
        name: formData.name,
        type: schemaType,
        payload: {
            connection_string: formData.connection_string,
            serial_number: formData.serial_number ?? '',
        },
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
                value={formData.connection_string}
                onChange={(connection_string) => {
                    updateField('connection_string', connection_string);
                    updateField('serial_number', '');
                }}
                placeholder='192.168.1.2'
            />
            <Flex gap='size-100'>
                <IdentifyRobot identifyMutation={identifyMutation} robot={identifyRobot} />
            </Flex>
        </Flex>
    );
};
