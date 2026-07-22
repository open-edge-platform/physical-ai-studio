import { Flex, TextField, View } from '@geti-ui/ui';
import { v4 as uuidv4 } from 'uuid';

import type { SchemaTrossenBimanualPayload } from '../../../../api/openapi-spec';
import type { SchemaRobot, SchemaRobotInput, SchemaRobotType } from '../../robot-types';
import { useRobotFormFields } from '../provider';
import { IdentifyRobot, useIdentifyMutation } from './actions';
import { buildWidowxBody } from './widowxai';

export interface BimanualFormData {
    name: string;
    payload: SchemaTrossenBimanualPayload;
}

export const getInitialBimanualFormData = (robot?: SchemaRobot): BimanualFormData => ({
    name: robot?.name ?? '',
    payload:
        robot && 'connection_string_left' in robot.payload
            ? robot.payload
            : { connection_string_left: '', connection_string_right: '', serial_number: '' },
});

export const buildBimanualBody = (
    formData: BimanualFormData,
    schemaType: SchemaRobotType,
    robot_id: string
): SchemaRobotInput | null => {
    if (!formData.payload.connection_string_left || !formData.payload.connection_string_right) {
        return null;
    }

    return {
        id: robot_id,
        name: formData.name,
        type: schemaType,
        payload: formData.payload,
    } as SchemaRobotInput;
};

export const BiManualWidowxAIFormFields = () => {
    const { formData, updateField } = useRobotFormFields<BimanualFormData>();

    const identifyMutation = useIdentifyMutation();
    const leftIdentifyRobot = buildWidowxBody(
        {
            name: formData.name,
            payload: { connection_string: formData.payload.connection_string_left, serial_number: '' },
        },
        'Trossen_WidowXAI_Follower',
        uuidv4()
    );
    const rightIdentifyRobot = buildWidowxBody(
        {
            name: formData.name,
            payload: { connection_string: formData.payload.connection_string_right, serial_number: '' },
        },
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
                        value={formData.payload.connection_string_left}
                        onChange={(connection_string_left) => {
                            updateField('payload', {
                                ...formData.payload,
                                connection_string_left,
                                serial_number: '',
                            });
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
                    value={formData.payload.connection_string_right}
                    onChange={(connection_string_right) => {
                        updateField('payload', {
                            ...formData.payload,
                            connection_string_right,
                            serial_number: '',
                        });
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
