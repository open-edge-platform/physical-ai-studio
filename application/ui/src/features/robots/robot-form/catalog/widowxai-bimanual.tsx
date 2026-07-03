import { Flex, TextField, View } from '@geti-ui/ui';
import { v4 as uuidv4 } from 'uuid';

import type { SchemaRobot, SchemaRobotInput, SchemaRobotType } from '../../robot-types';
import { useRobotFormFields } from '../provider';
import { IdentifyRobot, useIdentifyMutation } from './actions';
import { buildWidowxBody } from './widowxai';

export interface BimanualFormData {
    name: string;
    connection_string_left: string;
    connection_string_right: string;
    serial_number: string;
}

export const getInitialBimanualFormData = (robot?: SchemaRobot): BimanualFormData => ({
    name: robot?.name ?? '',
    connection_string_left:
        robot && 'connection_string_left' in robot.payload ? robot.payload.connection_string_left : '',
    connection_string_right:
        robot && 'connection_string_right' in robot.payload ? robot.payload.connection_string_right : '',
    serial_number: robot?.payload?.serial_number ?? '',
});

export const buildBimanualBody = (
    formData: BimanualFormData,
    schemaType: SchemaRobotType,
    robot_id: string
): SchemaRobotInput | null => {
    if (!formData.connection_string_left || !formData.connection_string_right) {
        return null;
    }

    return {
        id: robot_id,
        name: formData.name,
        type: schemaType,
        payload: {
            connection_string_left: formData.connection_string_left,
            connection_string_right: formData.connection_string_right,
            serial_number: formData.serial_number ?? '',
        },
    } as SchemaRobotInput;
};

export const BiManualWidowxAIFormFields = () => {
    const { formData, updateField } = useRobotFormFields<BimanualFormData>();

    const identifyMutation = useIdentifyMutation();
    const leftIdentifyRobot = buildWidowxBody(
        { name: formData.name, connection_string: formData.connection_string_left, serial_number: '' },
        'Trossen_WidowXAI_Follower',
        uuidv4()
    );
    const rightIdentifyRobot = buildWidowxBody(
        { name: formData.name, connection_string: formData.connection_string_right, serial_number: '' },
        'Trossen_WidowXAI_Follower',
        uuidv4()
    );

    return (
        <>
            <Flex direction='column' gap='size-100' width='100%'>
                <Flex gap='size-100' justifyContent={'space-between'} alignItems={'end'}>
                    <TextField
                        isRequired
                        label='Left arm IP address'
                        width='100%'
                        value={formData.connection_string_left}
                        onChange={(connection_string_left) => {
                            updateField('connection_string_left', connection_string_left);
                            updateField('serial_number', '');
                        }}
                        placeholder='192.168.1.2'
                    />
                    <View>
                        <IdentifyRobot identifyMutation={identifyMutation} robot={leftIdentifyRobot} />
                    </View>
                </Flex>
            </Flex>

            <Flex gap='size-100' justifyContent={'space-between'} alignItems={'end'}>
                <TextField
                    isRequired
                    label='Right arm IP address'
                    width='100%'
                    value={formData.connection_string_right}
                    onChange={(connection_string_right) => {
                        updateField('connection_string_right', connection_string_right);
                        updateField('serial_number', '');
                    }}
                    placeholder='192.168.1.3'
                />
                <View>
                    <IdentifyRobot identifyMutation={identifyMutation} robot={rightIdentifyRobot} />
                </View>
            </Flex>
        </>
    );
};
